
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "telco_churn_scoring_data.csv"
MODEL_PATH = ROOT / "models" / "lightgbm_churn_pipeline.joblib"
OUTPUT_PATH = ROOT / "reports" / "customer_churn_risk_with_ids.csv"

FEATURE_COLUMNS = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "service_count",
]


def main():
    print("Loading scoring data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Scoring dataset not found: {DATA_PATH}\n"
            "Update src/eda.py first, then run it to create the scoring dataset."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH, dtype={"customerid": "string"})

    required_columns = ["customerid"] + FEATURE_COLUMNS
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required scoring columns: {missing}")

    # Validate IDs before prediction.
    ids = df["customerid"].str.strip()

    if ids.isna().any() or ids.eq("").any():
        raise ValueError("Missing or blank customer IDs found.")

    if ids.duplicated().any():
        raise ValueError("Duplicate customer IDs found in scoring data.")

    if df.empty:
        raise ValueError("Scoring dataset is empty.")

    # Keep IDs outside the model feature matrix.
    customer_ids = ids.reset_index(drop=True)
    X = df[FEATURE_COLUMNS].copy()

    if "customerid" in X.columns:
        raise ValueError("Customer ID must not be a model feature.")

    print("Loading trained pipeline...")
    pipeline = joblib.load(MODEL_PATH)

    # Confirm the saved model expects these features in this order.
    expected = list(
        pipeline.named_steps["preprocessor"].feature_names_in_
    )
    if expected != FEATURE_COLUMNS:
        raise ValueError(
            "Scoring features do not match the trained model.\n"
            f"Model expects: {expected}\n"
            f"Scoring provides: {FEATURE_COLUMNS}\n"
            "Check the training and preprocessing scripts."
        )

    print("Predicting churn probabilities...")
    probabilities = pipeline.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "customerid": customer_ids,
        "predicted_churn_probability": probabilities,
        "tenure_months": df["tenure_months"].to_numpy(),
        "contract": df["contract"].to_numpy(),
        "monthly_charges": df["monthly_charges"].to_numpy(),
        "tech_support": df["tech_support"].to_numpy(),
        "online_security": df["online_security"].to_numpy(),
        "payment_method": df["payment_method"].to_numpy(),
        "internet_service": df["internet_service"].to_numpy(),
    })

    results["risk_segment"] = pd.cut(
        results["predicted_churn_probability"],
        bins=[-0.001, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)

    def recommend_action(row):
        reasons = []
        actions = []

        if row["risk_segment"] == "High":
            reasons.append("High predicted churn risk")
            actions.append("Prioritize proactive retention outreach")
        elif row["risk_segment"] == "Medium":
            reasons.append("Moderate predicted churn risk")
            actions.append("Consider a customer check-in")
        else:
            reasons.append("Low predicted churn risk")
            actions.append("Continue regular engagement")

        if row["contract"] == "Month-to-month":
            reasons.append("Month-to-month contract")
            actions.append("Review contract needs and suitable plan options")

        if pd.notna(row["tenure_months"]) and row["tenure_months"] <= 12:
            reasons.append("Tenure of 12 months or less")
            actions.append("Offer onboarding or service guidance")

        if row["tech_support"] == "No":
            reasons.append("No current tech support service")
            actions.append("Check whether technical support would be useful")

        if row["online_security"] == "No":
            reasons.append("No current online security service")
            actions.append("Review security service needs")

        if row["payment_method"] == "Electronic check":
            reasons.append("Electronic check payment method")
            actions.append("Check for billing or payment experience issues")

        return pd.Series({
            "recommendation_reasons": "; ".join(reasons),
            "recommended_actions": "; ".join(dict.fromkeys(actions)),
        })

    recommendations = results.apply(recommend_action, axis=1)
    results = pd.concat([results, recommendations], axis=1)

    # Ensure every input customer has exactly one output prediction.
    if len(results) != len(df):
        raise ValueError("Scoring row count does not match input row count.")

    if results["customerid"].duplicated().any():
        raise ValueError("Duplicate customer IDs in output report.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = results.sort_values(
        "predicted_churn_probability", ascending=False
    ).reset_index(drop=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\n--- CUSTOMER RISK SUMMARY ---")
    print(f"Customers scored: {len(results):,}")
    print(f"Unique customer IDs: {results['customerid'].nunique():,}")
    print(
        "Average predicted churn probability: "
        f"{results['predicted_churn_probability'].mean():.2%}"
    )
    print("\nRisk segment counts:")
    print(
        results["risk_segment"]
        .value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
        .to_string()
    )
    print(f"\nSaved customer risk report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
