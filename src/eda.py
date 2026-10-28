
from pathlib import Path
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from sqlalchemy import create_engine, URL, text

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data" / "processed"
CHART_DIR = ROOT / "reports" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

required = ["DB_USER", "DB_PASSWORD"]
missing = [key for key in required if not os.getenv(key)]

if missing:
    raise ValueError(
        f"Missing {missing} in your project .env file"
    )

url = URL.create(
    "postgresql+psycopg2",
    username=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_NAME", "customer_churn_db"),
)

engine = create_engine(url, pool_pre_ping=True)


def main():
    print("Connecting to PostgreSQL...")

    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM public.telco_churn_raw"),
            conn,
        )

    print("\n--- DATA OVERVIEW ---")
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum().sort_values(
        ascending=False
    ).head(15))
    print("\nDuplicate customer IDs:",
          df["customerid"].duplicated().sum())

    # Convert relevant columns to numeric
    for col in [
        "tenure_months", "monthly_charges",
        "total_charges", "churn_value"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Do not silently invent historical spending.
    # Missing total_charges will be handled after inspection.
    print(
        "\nMissing total_charges before cleaning:",
        df["total_charges"].isna().sum()
    )

    # Validate identifiers and target
    if df["customerid"].isna().any():
        raise ValueError("Customer IDs contain missing values.")

    if df["customerid"].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    df = df.dropna(
        subset=[
            "tenure_months", "monthly_charges", "churn_value"
        ]
    ).copy()

    if not set(df["churn_value"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected churn target values.")

    df["churn_value"] = df["churn_value"].astype(int)

    print("\n--- CHURN DISTRIBUTION ---")
    print(df["churn_value"].value_counts().sort_index())
    print(f"Overall churn rate: {df['churn_value'].mean():.2%}")

    # Contract analysis
    contract_summary = (
        df.groupby("contract", dropna=False)
        .agg(
            customers=("customerid", "count"),
            churn_rate=("churn_value", "mean"),
            avg_monthly_charges=("monthly_charges", "mean"),
            avg_tenure=("tenure_months", "mean"),
        )
        .sort_values("churn_rate", ascending=False)
    )

    print("\n--- CHURN BY CONTRACT ---")
    print(contract_summary.round(3))
    contract_summary.to_csv(
        DATA_DIR / "contract_summary.csv"
    )

    # Tenure bands
    df["tenure_band"] = pd.cut(
        df["tenure_months"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0-12", "13-24", "25-48", "49+"],
    ).astype("string")

    # Count services explicitly marked Yes
    service_cols = [
        "phone_service", "multiple_lines", "internet_service",
        "online_security", "online_backup",
        "device_protection", "tech_support",
        "streaming_tv", "streaming_movies",
    ]

    available_services = [
        col for col in service_cols if col in df.columns
    ]

    df["service_count"] = sum(
        df[col].eq("Yes").astype(int)
        for col in available_services
    )

    df["is_month_to_month"] = (
        df["contract"].eq("Month-to-month").astype(int)
    )

    df["is_paperless_billing"] = (
        df["paperless_billing"].eq("Yes").astype(int)
    )

    # Save charts
    sns.set_theme()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df, x="contract", y="churn_value", errorbar=None
    )
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn rate")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "churn_by_contract.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df, x="tenure_band", y="churn_value",
        order=["0-12", "13-24", "25-48", "49+"],
        errorbar=None,
    )
    plt.title("Churn Rate by Tenure")
    plt.ylabel("Churn rate")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "churn_by_tenure.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="churn_label", y="monthly_charges")
    plt.title("Monthly Charges by Churn Status")
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "monthly_charges_by_churn.png", dpi=150
    )
    plt.close()

        # Keep customer identifiers in the scoring dataset,
    # but exclude them from model training features.
    excluded_from_model = [
        "customerid", "count", "country", "state", "city",
        "zip_code", "lat_long", "latitude", "longitude",
        "churn_label", "churn_reason", "churn_score", "cltv",
        "total_charges",
    ]

    # Preserve IDs alongside processed features for traceable scoring.
    scoring_df = df.copy()

    # This dataset retains customerid, but training code must exclude it.
    scoring_path = DATA_DIR / "telco_churn_scoring_data.csv"
    scoring_df.to_csv(scoring_path, index=False)

    # Keep the existing training dataset ID-free.
    model_df = df.drop(
        columns=[c for c in excluded_from_model if c in df.columns]
    ).copy()

    model_path = DATA_DIR / "telco_churn_features.csv"
    model_df.to_csv(model_path, index=False)

    print("\n--- PIPELINE COMPLETED ---")
    print("Scoring dataset:", scoring_path)
    print("Scoring dataset shape:", scoring_df.shape)
    print("Modeling dataset:", model_path)
    print("Modeling dataset shape:", model_df.shape)
    print("Charts saved in:", CHART_DIR)
    print("Contract summary saved in:", DATA_DIR)
    print("\nModeling columns:")
    print(model_df.columns.tolist())


if __name__ == "__main__":
    try:
        main()
    finally:
        engine.dispose()
