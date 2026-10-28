
from pathlib import Path

import pandas as pd
import streamlit as st


# ----------------------------
# Configuration
# ----------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "reports" / "customer_churn_risk_with_ids.csv"

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# Visual styling
# ----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.22);
        padding: 16px 18px;
        border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.88rem;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 10px;
        overflow: hidden;
    }
    .section-note {
        color: var(--text-color);
        opacity: 0.78;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["predicted_churn_probability"] = pd.to_numeric(
        df["predicted_churn_probability"], errors="coerce"
    )
    df["monthly_charges"] = pd.to_numeric(
        df["monthly_charges"], errors="coerce"
    )
    df["tenure_months"] = pd.to_numeric(
        df["tenure_months"], errors="coerce"
    )
    df = df.dropna(subset=["predicted_churn_probability"])
    return df


st.title("📊 Customer Churn Prediction & Retention Intelligence")
st.caption(
    "LightGBM-powered customer risk scoring | "
    "Explore predicted churn risk and prioritize retention outreach"
)
st.markdown(
    '<p class="section-note">Decision-support dashboard · '
    'Predictions are estimates, not guarantees of churn.</p>',
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(f"Customer risk file not found: {DATA_PATH}")
    st.info("Run src/score_customers.py first.")
    st.stop()

df = load_data(DATA_PATH)

required_columns = {
    "customerid",
    "risk_segment",
    "contract",
    "predicted_churn_probability",
    "monthly_charges",
    "tenure_months",
}
missing_columns = sorted(required_columns - set(df.columns))
if missing_columns:
    st.error(
        "The customer-risk report is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.info("Regenerate the report by running src/score_customers.py.")
    st.stop()

if st.button("Refresh customer data"):
    st.cache_data.clear()
    st.rerun()

if df.empty:
    st.warning("The customer risk file contains no usable predictions.")
    st.stop()

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filter customers")

risk_order = ["High", "Medium", "Low"]
available_risks = [
    risk for risk in risk_order if risk in df["risk_segment"].unique()
]

selected_risks = st.sidebar.multiselect(
    "Risk segment",
    options=available_risks,
    default=available_risks,
)

contracts = sorted(df["contract"].dropna().astype(str).unique())
selected_contracts = st.sidebar.multiselect(
    "Contract type",
    options=contracts,
    default=contracts,
)

min_probability = st.sidebar.slider(
    "Minimum predicted churn probability",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    format="%.2f",
)

customer_query = st.sidebar.text_input(
    "Search customer ID",
    placeholder="Enter full or partial ID",
).strip().lower()

filtered = df[
    df["risk_segment"].isin(selected_risks)
    & df["contract"].astype(str).isin(selected_contracts)
    & (df["predicted_churn_probability"] >= min_probability)
    & df["customerid"].astype(str).str.lower().str.contains(
        customer_query, na=False, regex=False
    )
].copy()

# ----------------------------
# KPI cards
# ----------------------------
total_customers = len(filtered)

avg_risk = (
    filtered["predicted_churn_probability"].mean()
    if total_customers else 0
)

high_risk_count = int((filtered["risk_segment"] == "High").sum())

monthly_revenue_at_risk = (
    filtered.loc[
        filtered["risk_segment"] == "High", "monthly_charges"
    ].sum()
    if total_customers else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers in view", f"{total_customers:,}")
col2.metric("Average predicted risk", f"{avg_risk:.1%}")
col3.metric("High-risk customers", f"{high_risk_count:,}")
col4.metric(
    "Monthly charges of high-risk customers",
    f"${monthly_revenue_at_risk:,.0f}",
)

st.caption(
    "Monthly charges of high-risk customers are a prioritization indicator, "
    "not a forecast of revenue loss. The entire amount is not assumed to be lost."
)

st.divider()

# ----------------------------
# Charts
# ----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Customer risk distribution")

    risk_counts = (
        filtered["risk_segment"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
    )
    st.bar_chart(risk_counts)

with right:
    st.subheader("Average predicted risk by contract")

    contract_risk = (
        filtered.groupby("contract", observed=False)[
            "predicted_churn_probability"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    if not contract_risk.empty:
        st.bar_chart(contract_risk)
    else:
        st.info("No customers match the current filters.")

st.divider()

# ----------------------------
# Customer table
# ----------------------------
st.subheader("Customer risk register")

if filtered.empty:
    st.warning("No customers match these filters. Adjust the sidebar.")
else:
    display_df = filtered.sort_values(
        "predicted_churn_probability", ascending=False
    ).copy()

    display_df["predicted_churn_probability"] = (
        display_df["predicted_churn_probability"].map(
            lambda value: f"{value:.1%}"
        )
    )
    display_df["monthly_charges"] = display_df["monthly_charges"].map(
        lambda value: f"${value:,.2f}" if pd.notna(value) else "N/A"
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    download_df = filtered.sort_values(
        "predicted_churn_probability", ascending=False
    )

    st.download_button(
        label="Download filtered customer list (CSV)",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customer_churn_risk.csv",
        mime="text/csv",
    )

st.divider()

st.subheader("How to interpret this dashboard")
st.markdown(
    """
    - **High risk:** predicted churn probability above 60%.
    - **Medium risk:** predicted probability above 30%, up to 60%.
    - **Low risk:** predicted probability at or below 30%.

    These are initial business-defined segments, not validated intervention
    thresholds. Evaluate thresholds against campaign cost and retention outcomes.

    - The customer risk register includes customer IDs carried through the scoring dataset.
    - Predicted churn probabilities are model estimates, not guarantees of future churn.
    - Use customer IDs only within authorized workflows; redact them from public screenshots.
    """
)

st.caption(
    "Model evaluation is reported separately using the held-out test set. "
    "This dashboard describes predictions from the current scoring file."
)

# ==========================================
# RETENTION CAMPAIGN ROI SIMULATOR
# Append this section to the bottom of app.py
# ==========================================

st.divider()
st.title("💰 Retention Campaign ROI Simulator")

st.write(
    "Model a hypothetical retention campaign for high-risk customers. "
    "Adjust the assumptions to estimate potential financial outcomes."
)

st.warning(
    "Scenario analysis only: campaign effectiveness is an assumption, "
    "not a measured causal effect. Actual results may differ."
)

# Use all customers in the risk-scoring file for this simulation.
high_risk = df[df["risk_segment"] == "High"].copy()

if high_risk.empty:
    st.info(
        "There are no high-risk customers under the current risk thresholds. "
        "Review the risk-scoring file or adjust the risk thresholds."
    )
else:
    total_high_risk = len(high_risk)

    average_risk = high_risk[
        "predicted_churn_probability"
    ].mean()

    average_monthly_charge = high_risk[
        "monthly_charges"
    ].mean()

    st.subheader("1. Configure your campaign")

    col_a, col_b = st.columns(2)

    with col_a:
        campaign_reach = st.slider(
            "Percentage of high-risk customers contacted",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
        )

        effectiveness = st.slider(
            "Assumed relative reduction in churn",
            min_value=0,
            max_value=50,
            value=20,
            step=5,
            help=(
                "For example, 20% means a relative reduction in "
                "churn probability, not a 20 percentage-point reduction."
            ),
        )

    with col_b:
        offer_cost = st.number_input(
            "Campaign cost per contacted customer ($)",
            min_value=0.0,
            max_value=1000.0,
            value=20.0,
            step=5.0,
        )

        retention_months = st.slider(
            "Assumed months of retained service",
            min_value=1,
            max_value=24,
            value=6,
        )

        gross_margin = st.slider(
            "Assumed gross margin (%)",
            min_value=1,
            max_value=100,
            value=50,
        )

    # Expected-value scenario calculations.
    contacted = total_high_risk * campaign_reach / 100

    expected_churn_without_campaign = (
        contacted * average_risk
    )

    expected_churn_reduction = (
        expected_churn_without_campaign
        * effectiveness / 100
    )

    campaign_cost = contacted * offer_cost

    expected_retained_revenue = (
        expected_churn_reduction
        * average_monthly_charge
        * retention_months
    )

    expected_retained_margin = (
        expected_retained_revenue
        * gross_margin / 100
    )

    net_benefit = expected_retained_margin - campaign_cost

    if campaign_cost > 0:
        roi = net_benefit / campaign_cost * 100
    else:
        roi = None

    st.subheader("2. Estimated campaign outcomes")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "High-risk customers",
        f"{total_high_risk:,}",
    )

    k2.metric(
        "Expected customers contacted",
        f"{contacted:,.0f}",
    )

    k3.metric(
        "Expected churn events avoided",
        f"{expected_churn_reduction:,.1f}",
        help="An expected value, so it can be fractional.",
    )

    k4.metric(
        "Estimated campaign cost",
        f"${campaign_cost:,.2f}",
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Estimated retained revenue",
        f"${expected_retained_revenue:,.2f}",
    )

    r2.metric(
        "Estimated retained gross margin",
        f"${expected_retained_margin:,.2f}",
    )

    r3.metric(
        "Estimated net benefit",
        f"${net_benefit:,.2f}",
    )

    if roi is not None:
        st.metric("Estimated ROI", f"{roi:,.1f}%")
    else:
        st.metric("Estimated ROI", "N/A")
        st.caption(
            "ROI is undefined when campaign cost is zero."
        )

    st.caption(
        "Calculation: expected churn events avoided = contacted customers "
        "× average predicted churn probability × assumed relative reduction. "
        "Retained gross margin = expected churn events avoided × average "
        "monthly charges × assumed retained months × gross margin. "
        "Net benefit = retained gross margin − campaign cost."
    )

    st.subheader("3. Assumptions and limitations")

    st.markdown(
        """
        - The model assumes campaign effectiveness applies equally to
          contacted high-risk customers.
        - It assumes the specified retention period and gross margin
          are appropriate for the scenario.
        - It excludes discount side effects, operational overhead,
          customer heterogeneity, and uncertainty in the estimates.
        - The calculation does not establish that a campaign causes
          customers to stay. A randomized experiment would help measure
          the actual incremental effect.
        - Results are expected-value estimates, not guaranteed savings.
        """
    )

# ==========================================
# MODEL EXPLAINABILITY: FEATURE IMPORTANCE
# ==========================================

st.divider()
st.title("🔎 Model Explainability")
st.write(
    "Explore which transformed features the trained LightGBM model "
    "uses most when making churn predictions."
)

importance_path = ROOT / "reports" / "feature_importance.csv"
importance_chart_path = ROOT / "reports" / "figures" / "feature_importance.png"

if importance_path.exists():
    importance_df = pd.read_csv(importance_path)
    importance_df = importance_df.sort_values(
        "importance", ascending=False
    )

    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=min(25, len(importance_df)),
        value=min(15, len(importance_df)),
    )

    top_features = importance_df.head(top_n).copy()

    st.subheader(f"Top {top_n} features")
    st.bar_chart(
        top_features.set_index("feature")[["importance"]]
    )

    st.subheader("Importance scores")
    st.dataframe(top_features, width="stretch", hide_index=True)

    st.download_button(
        "Download feature importance (CSV)",
        data=importance_df.to_csv(index=False).encode("utf-8"),
        file_name="feature_importance.csv",
        mime="text/csv",
    )

    st.caption(
        "Feature importance indicates model reliance, not causal influence "
        "or the direction of an individual customer's churn risk."
    )
else:
    st.info(
        "Feature importance has not been generated yet. "
        "Run src/feature_importance.py first."
    )

# ==========================================
# MODEL EVALUATION DASHBOARD
# ==========================================

st.divider()
st.title("📈 Model Evaluation & Threshold Analysis")

st.write(
    "Evaluate performance on the held-out test set and explore "
    "how the classification threshold changes churn predictions."
)

evaluation_path = ROOT / "reports" / "test_predictions.csv"
figures_dir = ROOT / "reports" / "figures"

if evaluation_path.exists():
    eval_df = pd.read_csv(evaluation_path)

    y_true = eval_df["actual_churn"].astype(int)
    probabilities = eval_df["predicted_probability"].astype(float)

    threshold = st.slider(
        "Classification threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="evaluation_threshold",
    )

    predicted = (probabilities >= threshold).astype(int)

    tp = int(((y_true == 1) & (predicted == 1)).sum())
    tn = int(((y_true == 0) & (predicted == 0)).sum())
    fp = int(((y_true == 0) & (predicted == 1)).sum())
    fn = int(((y_true == 1) & (predicted == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    accuracy = (tp + tn) / len(eval_df)

    st.caption(
        f"Evaluation uses {len(eval_df):,} held-out test customers. "
        "Changing this slider changes the predicted classes, not the trained model."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision:.1%}")
    c2.metric("Recall", f"{recall:.1%}")
    c3.metric("F1 score", f"{f1:.1%}")
    c4.metric("Accuracy", f"{accuracy:.1%}")

    st.subheader("Interactive confusion matrix")

    confusion_df = pd.DataFrame(
        [
            [tn, fp],
            [fn, tp],
        ],
        index=["Actual: Stayed", "Actual: Churned"],
        columns=["Predicted: Stayed", "Predicted: Churned"],
    )

    st.dataframe(confusion_df, width="stretch")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("True negatives", f"{tn:,}")
    m2.metric("False positives", f"{fp:,}")
    m3.metric("False negatives", f"{fn:,}")
    m4.metric("True positives", f"{tp:,}")

    st.subheader("Evaluation curves")

    chart_files = [
        ("ROC curve", "roc_curve.png"),
        ("Precision–recall curve", "precision_recall_curve.png"),
        ("Threshold comparison", "threshold_comparison.png"),
        ("Confusion matrix at 0.50", "evaluation_confusion_matrix.png"),
    ]

    chart_cols = st.columns(2)

    for index, (label, filename) in enumerate(chart_files):
        chart_path = figures_dir / filename
        with chart_cols[index % 2]:
            st.markdown(f"**{label}**")
            if chart_path.exists():
                st.image(str(chart_path), width="stretch")
            else:
                st.info(
                    f"{filename} not found. Run src/evaluate_churn.py first."
                )

    st.download_button(
        "Download held-out test predictions",
        data=eval_df.to_csv(index=False).encode("utf-8"),
        file_name="test_predictions.csv",
        mime="text/csv",
    )

else:
    st.warning(
        "Test predictions not found. Run src/evaluate_churn.py "
        "after confirming reports/test_predictions.csv exists."
    )
