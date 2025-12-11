import json
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Credit Default Risk Simulator",
    layout="centered"
)

# -----------------------------
# Load model & schema
# -----------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "model_artifacts"
MODEL_PATH = ARTIFACT_DIR / "credit_risk_model_FINAL.joblib"
SCHEMA_PATH = ARTIFACT_DIR / "schema.json"


@st.cache_resource
def load_artifacts():
    """Load trained pipeline and schema.json."""
    model = load(MODEL_PATH)
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return model, schema


model, schema = load_artifacts()
threshold = float(schema.get("threshold", 0.5))

# -----------------------------
# Sidebar (project info)
# -----------------------------
with st.sidebar:
    st.header("ℹ️ About this app")
    st.write(
        "Interactive demo of a **credit default risk model** built with "
        "Python, scikit-learn, XGBoost and Streamlit."
    )
    st.caption(
        "The model estimates the probability that a loan will default "
        "(Status = 1) based on applicant and loan characteristics."
    )
    st.markdown("---")
    st.caption("For portfolio / educational use only – not financial advice.")

# -----------------------------
# Main title
# -----------------------------
st.title("💳 Credit Default Risk Simulator")

st.write(
    "Fill in the applicant and loan details below to estimate the "
    "**probability of default** according to the trained model."
)

st.markdown("---")

# -----------------------------
# Input form
# -----------------------------
st.header(" Applicant & Loan Details")

with st.form("loan_form"):
    st.subheader("Loan information")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Application Year",
            min_value=2000,
            max_value=2100,
            value=2015,
            step=1,
        )
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=1_000,
            max_value=2_000_000,
            value=150_000,
            step=1_000,
            help="Total principal amount requested by the borrower.",
        )
        term = st.number_input(
            "Loan Term (months)",
            min_value=1,
            max_value=600,
            value=360,
            step=1,
            help="Length of the loan in months (e.g., 360 = 30 years).",
        )
        property_value = st.number_input(
            "Property Value",
            min_value=1,
            max_value=5_000_000,
            value=250_000,
            step=5_000,
            help="Estimated value of the property used as collateral.",
        )

    with col2:
        income = st.number_input(
            "Applicant Income",
            min_value=0,
            max_value=1_000_000,
            value=60_000,
            step=1_000,
            help="Annual income of the applicant (or household).",
        )
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=700,
            step=1,
            help="Higher scores generally indicate lower credit risk.",
        )
        ltv = st.number_input(
            "LTV (%)",
            min_value=0.0,
            max_value=200.0,
            value=80.0,
            step=1.0,
            help="Loan-to-Value ratio = loan amount / property value.",
        )
        dtir1 = st.number_input(
            "DTI (%)",
            min_value=0.0,
            max_value=200.0,
            value=30.0,
            step=1.0,
            help="Debt-to-Income ratio including this loan.",
        )

    st.subheader("Applicant profile")

    col3, col4, col5 = st.columns(3)

    with col3:
        Gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Joint", "Sex Not Available"],
        )
        business_or_commercial = st.selectbox(
            "Business / Commercial Loan?",
            ["b/c", "nob/c"],
        )

    with col4:
        lump_sum_payment = st.selectbox(
            "Lump Sum Payment Option",
            ["lpsm", "not_lpsm"],
        )
        credit_type = st.selectbox(
            "Credit Bureau Type",
            ["CIB", "CRIF", "EXP", "EQUI"],
        )

    with col5:
        age = st.selectbox(
            "Age Group",
            ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"],
        )
        Region = st.selectbox(
            "Region",
            ["North", "South", "North-East", "Central"],
        )

    submitted = st.form_submit_button("Estimate Default Risk")

# -----------------------------
# Prediction & interpretation
# -----------------------------
if submitted:
    # Build single-row DataFrame with exact training column names
    input_row = {
        "year": year,
        "loan_amount": loan_amount,
        "term": term,
        "property_value": property_value,
        "income": income,
        "Credit_Score": credit_score,   # must match dataset column exactly
        "LTV": ltv,
        "dtir1": dtir1,
        "Gender": Gender,
        "business_or_commercial": business_or_commercial,
        "lump_sum_payment": lump_sum_payment,
        "credit_type": credit_type,
        "age": age,
        "Region": Region,
    }

    df_input = pd.DataFrame([input_row])

    try:
        proba = model.predict_proba(df_input)[0, 1]  # P(default = 1)
    except Exception as e:
        st.error(f"Prediction error: {e}")
    else:
        st.markdown("---")
        st.header(" Prediction Result")

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric(
                label="Estimated probability of default",
                value=f"{proba:.1%}",
            )
        with col_b:
            st.metric(
                label="Decision threshold used during training",
                value=f"{threshold:.1%}",
            )

        # Simple visual risk meter
        st.write("")
        st.write("**Risk level (according to the model):**")
        risk_bar = st.progress(0)
        risk_bar.progress(int(proba * 100))

        # Text interpretation
        if proba < 0.20:
            st.success("🟢 **Low-risk profile** – the model estimates a relatively low probability of default.")
        elif proba < 0.50:
            st.warning("🟡 **Medium-risk profile** – the model sees mixed signals; closer review is recommended.")
        else:
            st.error("🔴 **High-risk profile** – the model estimates a high probability of default.")

        st.caption(
            "This is a demonstration model trained on historical data. "
            "It does not replace a full credit policy or professional risk assessment."
        )
