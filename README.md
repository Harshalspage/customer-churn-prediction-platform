# Customer Churn Prediction & Retention Intelligence Platform

An end-to-end machine learning and analytics project that estimates
customer churn risk, segments customers by predicted risk, and provides
retention-oriented insights through a Streamlit dashboard.

> **Project status:** The local pipeline and Streamlit dashboard have
> been run successfully. Metrics below are from a held-out test set.
> SHAP explanations are optional and may be skipped on Windows systems
> where application-control policies block SHAP's native extension.

## Overview

Customer churn can reduce recurring revenue and increase acquisition
costs. This project uses customer attributes and service information to
estimate churn probability and help analysts prioritize retention
outreach.

The workflow includes: - Reading customer data from PostgreSQL. -
Exploring and preparing data with Python. - Engineering features for
modeling and reporting. - Training a LightGBM classification pipeline. -
Evaluating the model on a stratified hold-out test set. - Scoring
customers and assigning Low, Medium, and High risk segments. -
Displaying customer-level risk information and retention recommendations
in Streamlit.

## Key Results

Evaluation on the held-out test set:

  Metric                                Result
  ----------------------------------- --------
  ROC-AUC                               0.8573
  Average precision (PR-AUC)            0.6830
  Accuracy at threshold 0.50            76.79%
  Churn precision at threshold 0.50     54.36%
  Churn recall at threshold 0.50        78.34%
  Churn F1-score at threshold 0.50      64.18%
  Test records                           1,409

The dataset contains 7,043 customer records, with an observed churn rate
of 26.54%. The model was evaluated on 1,409 held-out records.

**Interpretation:** At the selected 0.50 threshold, the model identifies
about 78% of churners in the test set, while churn precision is about
54%. Treat the model as a prioritization aid, not a guarantee that a
customer will churn. Thresholds should reflect retention-team capacity
and the cost of false positives and missed churners.

## Risk Segments

The current report has shown these counts in a recent scoring run; they
can change if the model or scoring data changes.

  Risk segment     Customers in that run
  -------------- -----------------------
  High                             2,277
  Medium                           1,248
  Low                              3,518
  **Total**                    **7,043**

These are project-defined categories based on predicted probabilities,
not verified future outcomes.

## Technology Stack

-   **Language:** Python
-   **Data processing:** pandas, NumPy
-   **Database:** PostgreSQL
-   **Machine learning:** LightGBM, scikit-learn
-   **Visualization:** Streamlit, Matplotlib, Seaborn
-   **Model persistence:** joblib
-   **Explainability (optional):** SHAP
-   **Version control:** Git and GitHub

## Pipeline Architecture

``` text
PostgreSQL: public.telco_churn_raw
                |
                v
             src/eda.py
       validation + EDA + features
                |
        +-------+--------+
        |                |
        v                v
Scoring dataset      Modeling dataset
telco_churn_         telco_churn_
scoring_data.csv     features.csv
(customerid kept)    (ID excluded)
        |                |
        |                v
        |          src/train_churn.py
        |          preprocessing +
        |          LightGBM training
        |                |
        |                v
        |          models/
        |          lightgbm_churn_pipeline.joblib
        |                |
        +-------+--------+
                |
                v
        src/score_customers.py
        predict churn probability
        assign risk + recommendations
                |
                v
        reports/customer_churn_risk_with_ids.csv
                |
                v
               app.py
         Streamlit dashboard
```

Customer identifiers are retained in the scoring dataset so predictions
can be traced to customers, but `customerid` is excluded from the
model's input features.

## Project Structure

This shows the main pipeline files; additional files may exist in the
local repository.

``` text
customer-churn-platform/
├── app.py
├── src/
│   ├── eda.py
│   ├── train_churn.py
│   └── score_customers.py
├── data/
│   └── processed/
│       ├── telco_churn_scoring_data.csv
│       ├── telco_churn_features.csv
│       └── contract_summary.csv
├── models/
│   └── lightgbm_churn_pipeline.joblib
├── reports/
│   ├── customer_churn_risk_with_ids.csv
│   ├── model_metrics.json
│   ├── test_predictions.csv
│   └── figures/
├── .env                 # local credentials; do not commit
├── .gitignore
└── README.md
```

## Setup

### 1. Requirements

-   Python 3.10+ recommended
-   PostgreSQL
-   Git (optional)

Create and activate a virtual environment in the project root.

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, use the virtual environment's
Python executable directly, as shown below.

Install dependencies if a `requirements.txt` is present:

``` powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If no dependency file exists yet, create one from the packages imported
by the project and verify it in a clean environment before publishing.

### 2. Configure PostgreSQL

Ensure the raw source table exists and is populated:

``` text
public.telco_churn_raw
```

Create a `.env` file in the project root with your local connection
details:

``` dotenv
DB_HOST=localhost
DB_PORT=5432
DB_NAME=customer_churn_db
DB_USER=your_postgres_username
DB_PASSWORD=your_postgres_password
```

Use your own credentials. Never commit `.env` or real passwords to
GitHub.

### 3. Generate processed datasets and EDA outputs

``` powershell
.\.venv\Scripts\python.exe .\src\eda.py
```

This reads the PostgreSQL raw table and writes processed datasets and
charts.

### 4. Train and evaluate the model

``` powershell
.\.venv\Scripts\python.exe .\src\train_churn.py
```

This trains the preprocessing + LightGBM pipeline and writes the saved
model and evaluation outputs. SHAP explanations may be skipped if the
local environment blocks SHAP's native extension.

### 5. Score customers

``` powershell
.\.venv\Scripts\python.exe .\src\score_customers.py
```

This generates:

``` text
reports/customer_churn_risk_with_ids.csv
```

Check that it contains one row per scored customer and the corresponding
`customerid`.

### 6. Launch the dashboard

``` powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Open the local URL printed by Streamlit, usually
`http://localhost:8501`.

## Dashboard Features

The dashboard is intended to support: - Customer-level risk register
with customer identifiers. - Risk-segment filtering. - Summary charts
and churn-risk analysis. - Retention-oriented recommended actions. -
Downloadable customer-risk CSV. - ROI scenario simulation, as
implemented in the current dashboard.

Confirm features against the current `app.py` before describing them as
production-ready.

## Model Evaluation Notes

-   The data is split into training and test partitions using a
    stratified split with `random_state=42`.
-   Preprocessing and LightGBM are wrapped in a scikit-learn pipeline.
-   Threshold-dependent metrics use a 0.50 classification threshold.
-   The model uses class weighting to address churn-class imbalance.
-   ROC-AUC and average precision summarize ranking quality;
    threshold-dependent metrics depend on the selected threshold.
-   This is a single hold-out evaluation, not evidence of performance
    across all future customer populations.

## Explainability

The training script attempts to create SHAP feature explanations after
training. On some Windows environments, SHAP's compiled extension may be
blocked by application-control policy. The script is configured to skip
SHAP when that optional step fails, allowing training and evaluation to
complete.

Do not claim a SHAP report was generated unless
`reports/shap_feature_importance.csv` or
`reports/figures/shap_summary.png` exists and was updated by the latest
run.

## Limitations and Responsible Use

-   Predictions are estimates based on historical customer data, not
    certainties about individual behavior.
-   A high-risk label should prompt thoughtful review or a helpful
    retention offer, not automatic penalties or denial of service.
-   Some customers flagged as likely to churn will not actually churn.
-   Data quality, population changes, and changes in pricing or service
    policies can reduce performance over time.
-   Risk thresholds and recommended actions are project-defined and
    should be validated with business stakeholders.
-   A prediction model alone does not prove that a retention
    intervention will prevent churn. Use controlled experiments where
    possible.
-   Protect customer data. Do not publish real customer identifiers or
    raw customer records in public screenshots, sample outputs, or the
    repository.
-   Remove credentials and sensitive datasets before pushing the project
    to GitHub.




## Dashboard Screenshots

### 1. Customer Churn Dashboard
![Customer churn dashboard](Screen%20shots/dashboard-overview.png)

### 2. Customer Risk Register
![Customer risk register](Screen%20shots/customer-risk-register.png)

### 3. Model Evaluation
![Model evaluation](Screen%20shots/model-evaluation.png)

### 4. ROI Simulator
![ROI Simulator](Screen%20shots/Roi-stimulator.png)


Suggested screenshots: 1. Dashboard overview with KPI cards and summary
charts. 2. Risk register with customer identifiers redacted. 3.
Risk-segment filter in use. 4. Retention recommendation or ROI
simulator.

## Future Improvements

-   Add automated tests for data validation, scoring, and report
    schemas.
-   Add a reproducible dependency file or verified `requirements.txt`.
-   Compare classification thresholds using business costs and retention
    capacity.
-   Track model drift and monitor precision/recall against newly
    observed outcomes.
-   Validate recommended actions through A/B tests or controlled
    retention experiments.
-   Add a privacy-safe demo dataset and deployment instructions.
-   Add a compatible explainability workflow or alternative
    feature-importance method.

## Author

Add your name, GitHub profile, and project repository link here before
publishing.

------------------------------------------------------------------------

**Disclaimer:** This project is for portfolio and educational purposes.
It is not a production churn-management system without further
validation, monitoring, security, privacy review, and business testing.
