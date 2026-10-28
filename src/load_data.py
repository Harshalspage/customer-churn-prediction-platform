import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Telco_customer_churn.xlsx"

# Load connection settings
load_dotenv(PROJECT_ROOT / ".env")

required_vars = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing = [key for key in required_vars if not os.getenv(key)]

if missing:
    raise ValueError(f"Missing database settings in .env: {missing}")

# Safely construct the PostgreSQL connection URL
connection_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    host=os.environ["DB_HOST"],
    port=int(os.environ["DB_PORT"]),
    database=os.environ["DB_NAME"],
)

engine = create_engine(connection_url, pool_pre_ping=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place Telco_customer_churn.xlsx in the data folder."
        )

    # Read the Excel workbook
    df = pd.read_excel(DATA_PATH)

    # Standardize column names for PostgreSQL
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Basic validation
    if "customerid" not in df.columns:
        raise ValueError("Expected customer ID column is missing.")

    if df["customerid"].isna().any():
        raise ValueError("Missing customer IDs found.")

    if df["customerid"].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    if "churn_value" not in df.columns:
        raise ValueError("Expected churn_value target column is missing.")

    if not set(df["churn_value"].dropna().unique()).issubset({0, 1}):
        raise ValueError("churn_value must contain only 0 and 1.")

    # Load raw records into PostgreSQL.
    # Replace the table on reruns: this is an initial development pipeline.
    with engine.begin() as connection:
        df.to_sql(
            name="telco_churn_raw",
            con=connection,
            if_exists="replace",
            index=False,
            chunksize=1000,
            method="multi",
        )

        row_count = connection.execute(
            text("SELECT COUNT(*) FROM telco_churn_raw")
        ).scalar_one()

        distinct_customers = connection.execute(
            text("""
                SELECT COUNT(DISTINCT customerid)
                FROM telco_churn_raw
            """)
        ).scalar_one()

    print("Data loaded successfully!")
    print(f"Excel rows: {len(df):,}")
    print(f"Database rows: {row_count:,}")
    print(f"Unique customers: {distinct_customers:,}")
    print(f"Columns: {len(df.columns)}")
    print("Table: public.telco_churn_raw")

    engine.dispose()


if __name__ == "__main__":
    main()