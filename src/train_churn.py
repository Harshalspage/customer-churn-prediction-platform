
from pathlib import Path

import json
import warnings


import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "telco_churn_features.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def main():
    print("Loading modeling dataset...")
    df = pd.read_csv(DATA_PATH)

    target = "churn_value"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    df = df.dropna(subset=[target]).copy()
    df[target] = df[target].astype(int)

    if not set(df[target].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

        # Exclude identifiers, target, and redundant engineered fields.
    excluded = [
        "customerid",
        target,
        "tenure_band",
        "is_month_to_month",
        "is_paperless_billing",
    ]

    X = df.drop(
        columns=[c for c in excluded if c in df.columns]
    ).copy()
    y = df[target]

    if "customerid" in X.columns:
        raise ValueError(
            "Data leakage risk: customerid must not be a model feature."
        )
        
    # Remove constant columns, if any.
    constant_cols = [
        col for col in X.columns if X[col].nunique(dropna=False) <= 1
    ]
    X = X.drop(columns=constant_cols)

    # Explicitly treat categorical fields as strings.
    categorical_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_cols = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    # Stratification preserves the approximate churn proportion.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Features:", X.shape[1])
    print("Training churn rate:", round(y_train.mean(), 4))
    print("Test churn rate:", round(y_test.mean(), 4))

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )),
    ])

    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # Handle class imbalance with training data only.
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    class_weight_ratio = negative_count / positive_count

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=20,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=class_weight_ratio,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    print("\nTraining LightGBM...")
    pipeline.fit(X_train, y_train)

    # Evaluate on the untouched test set.
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_pr_auc": float(average_precision_score(y_test, probabilities)),
        "precision_at_0_50": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall_at_0_50": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_at_0_50": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "classification_threshold": 0.50,
        "test_rows": int(len(y_test)),
        "positive_class_weight_ratio": float(class_weight_ratio),
        "constant_columns_removed": constant_cols,
        "input_features": X.columns.tolist(),
    }

    print("\n--- TEST METRICS ---")
    print(json.dumps(metrics, indent=2))
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(
        y_test, predictions, target_names=["Stayed", "Churned"],
        zero_division=0
    ))

    with open(REPORT_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "actual_churn": y_test.to_numpy(),
        "predicted_probability": probabilities,
        "predicted_churn_at_0_50": predictions,
    }, index=X_test.index).to_csv(
        REPORT_DIR / "test_predictions.csv", index_label="row_index"
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stayed", "Churned"],
    )
    disp.plot()
    plt.title("LightGBM Churn — Test Set")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "churn_confusion_matrix.png", dpi=150
    )
    plt.close()

    # Save full pipeline so preprocessing is retained at prediction time.
    model_path = MODEL_DIR / "lightgbm_churn_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    
    # Optional SHAP explanations.
    # SHAP may be unavailable because of Windows security restrictions.
    print("\nGenerating SHAP explanations...")

    try:
        import shap

        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        fitted_model = pipeline.named_steps["model"]

        X_test_transformed = fitted_preprocessor.transform(X_test)
        feature_names = fitted_preprocessor.get_feature_names_out()

        # Sample up to 200 test records.
        rng = np.random.default_rng(RANDOM_STATE)
        sample_size = min(200, X_test.shape[0])
        sample_indices = rng.choice(
            X_test.shape[0],
            size=sample_size,
            replace=False,
        )
        X_shap = X_test_transformed[sample_indices]

        explainer = shap.TreeExplainer(fitted_model)
        shap_values = explainer.shap_values(X_shap)

        if isinstance(shap_values, list):
            shap_values = (
                shap_values[1]
                if len(shap_values) > 1
                else shap_values[0]
            )

        shap_values = np.asarray(shap_values)

        importance = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        importance.to_csv(
            REPORT_DIR / "shap_feature_importance.csv",
            index=False,
        )

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_shap,
            feature_names=feature_names,
            max_display=20,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / "shap_summary.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print("\nTop 15 SHAP features:")
        print(importance.head(15).to_string(index=False))
        print("SHAP importance saved.")

    except Exception as exc:
        print(f"SHAP explanations skipped: {exc}")
        print("Model training and evaluation outputs remain available.")

    print("\n--- SAVED OUTPUTS ---")
    print("Model:", model_path)
    print("Metrics:", REPORT_DIR / "model_metrics.json")
    print("Predictions:", REPORT_DIR / "test_predictions.csv")
    print("SHAP importance:", REPORT_DIR / "shap_feature_importance.csv")
    print("Figures:", FIGURE_DIR)
    print("\nTraining completed.")

if __name__ == "__main__":
    main()
