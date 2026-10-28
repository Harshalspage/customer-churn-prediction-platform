
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lightgbm_churn_pipeline.joblib"
OUTPUT_CSV = ROOT / "reports" / "feature_importance.csv"
OUTPUT_PNG = ROOT / "reports" / "figures" / "feature_importance.png"


def main():
    print("Loading trained pipeline...")
    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            "Feature-name count does not match model importance count."
        )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    importance_df.to_csv(OUTPUT_CSV, index=False)

    top_features = importance_df.head(15).sort_values(
        "importance", ascending=True
    )

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("LightGBM split/gain importance setting")
    plt.ylabel("Model feature")
    plt.title("Top 15 Features Used by Churn Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))
    print(f"\nCSV saved: {OUTPUT_CSV}")
    print(f"Chart saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
