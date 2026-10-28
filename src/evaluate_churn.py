
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
FIGURES_DIR = ROOT / "reports" / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load held-out test predictions
    df = pd.read_csv(PREDICTIONS_PATH)

    required = [
        "actual_churn",
        "predicted_probability",
        "predicted_churn_at_0_50",
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    y_true = df["actual_churn"].astype(int)
    probabilities = df["predicted_probability"].astype(float)

    if y_true.nunique() != 2:
        raise ValueError("Evaluation requires both churn classes.")

    if not probabilities.between(0, 1).all():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    # ---------------------------------
    # 1. Confusion matrix
    # ---------------------------------
    threshold = 0.50
    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stayed", "Churned"],
    )
    display.plot(values_format="d")
    plt.title("Churn Model Confusion Matrix (Threshold 0.50)")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "evaluation_confusion_matrix.png",
        dpi=150,
    )
    plt.close()

    # ---------------------------------
    # 2. ROC curve
    # ---------------------------------
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # ---------------------------------
    # 3. Precision-recall curve
    # ---------------------------------
    precision_values, recall_values, _ = precision_recall_curve(
        y_true, probabilities
    )
    pr_auc = average_precision_score(y_true, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(
        recall_values,
        precision_values,
        label=f"Average precision = {pr_auc:.3f}",
    )
    plt.axhline(
        y_true.mean(),
        linestyle="--",
        label=f"Churn prevalence = {y_true.mean():.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()

    # ---------------------------------
    # 4. Threshold comparison
    # ---------------------------------
    threshold_rows = []

    for cutoff in np.arange(0.10, 0.91, 0.05):
        predicted = (probabilities >= cutoff).astype(int)

        threshold_rows.append({
            "threshold": round(float(cutoff), 2),
            "precision": precision_score(
                y_true, predicted, zero_division=0
            ),
            "recall": recall_score(
                y_true, predicted, zero_division=0
            ),
            "f1": f1_score(
                y_true, predicted, zero_division=0
            ),
            "customers_flagged": int(predicted.sum()),
            "true_positives": int(
                ((y_true == 1) & (predicted == 1)).sum()
            ),
            "false_positives": int(
                ((y_true == 0) & (predicted == 1)).sum()
            ),
            "false_negatives": int(
                ((y_true == 1) & (predicted == 0)).sum()
            ),
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(
        ROOT / "reports" / "threshold_comparison.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        marker="o",
        label="Precision",
    )
    plt.plot(
        threshold_df["threshold"],
        threshold_df["recall"],
        marker="o",
        label="Recall",
    )
    plt.plot(
        threshold_df["threshold"],
        threshold_df["f1"],
        marker="o",
        label="F1 score",
    )
    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.title("Classification Metrics by Threshold")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "threshold_comparison.png",
        dpi=150,
    )
    plt.close()

    # ---------------------------------
    # 5. Metrics and report
    # ---------------------------------
    metrics = {
        "test_rows": int(len(df)),
        "churn_prevalence": float(y_true.mean()),
        "roc_auc": float(roc_auc),
        "average_precision": float(pr_auc),
        "accuracy_at_0_50": float(
            accuracy_score(y_true, predictions)
        ),
        "precision_at_0_50": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall_at_0_50": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_at_0_50": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "confusion_matrix": cm.tolist(),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n--- HELD-OUT TEST EVALUATION ---")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Stayed", "Churned"],
            zero_division=0,
        )
    )

    print("\nThreshold comparison around 0.50:")
    print(
        threshold_df.iloc[
            (threshold_df["threshold"] - 0.50).abs().argsort()[:3]
        ].sort_values("threshold").to_string(index=False)
    )

    print("\nSaved evaluation figures to:", FIGURES_DIR)
    print("Saved threshold comparison to: reports/threshold_comparison.csv")


if __name__ == "__main__":
    main()
