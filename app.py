import json
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

# -----------------------------
# Basic page config
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
# Title & intro
# -----------------------------
st.title("💳 Credit Default Risk Simulator")

st.write(
    """
    This app uses a trained machine-learning model to estimate the probability that
    a loan will **default (Status = 1)** based on borrower and loan characteristics.
    Use it as an educational scoring demo, not as financial advice.
    """
)

st.markdown("---")

# -----------------------------
# Input form
# -----------------------------
st.header("🔍 Applicant & Loan Details")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Application Year", min_value=2000, max_value=2100, value=2015)
        loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=2_000_000, value=150000)
        term = st.number_input("Loan Term (months)", min_value=1, max_value=600, value=360)
        property_value = st.number_input("Property Value", min_value=1, max_value=5_000_000, value=250000)

    with col2:
        income = st.number_input("Applicant Income", min_value=0, max_value=1_000_000, value=60000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
        ltv = st.number_input("LTV (%)", min_value=0, max_value=200, value=80)
        dtir1 = st.number_input("DTI (%)", min_value=0, max_value=200, value=30)

    st.subheader("Profile & Context")

    col3, col4, col5 = st.columns(3)

    with col3:
        Gender = st.selectbox("Gender", ["Male", "Female", "Joint", "Sex Not Available"])
        business_or_commercial = st.selectbox("Business / Commercial Loan?", ["b/c", "nob/c"])

    with col4:
        lump_sum_payment = st.selectbox("Lump Sum Payment", ["lpsm", "not_lpsm"])
        credit_type = st.selectbox("Credit Bureau Type", ["CIB", "CRIF", "EXP", "EQUI"])

    with col5:
        age = st.selectbox("Age Group", ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"])
        Region = st.selectbox("Region", ["North", "South", "North-East", "Central"])

    submitted = st.form_submit_button("Estimate Default Risk")

# -----------------------------
# Prediction logic
# -----------------------------
if submitted:
    input_row = {
        "year": year,
        "loan_amount": loan_amount,
        "term": term,
        "property_value": property_value,
        "income": income,
        "Credit_Score": credit_score,   # must match training column name
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
        proba = model.predict_proba(df_input)[0, 1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
    else:
        st.markdown("---")
        st.header("📊 Prediction Result")

        st.write(f"**Estimated Probability of Default:** `{proba:.3f}`")
        st.write(f"**Decision Threshold (from training):** `{threshold:.3f}`")

        # Simple risk bands
        if proba < 0.20:
            st.success("🟢 **Low-risk profile** – according to the model, default risk is relatively low.")
        elif proba < 0.50:
            st.warning("🟡 **Medium-risk profile** – mixed signals; closer review recommended.")
        else:
            st.error("🔴 **High-risk profile** – the model estimates a high probability of default.")

        st.caption(
            "This is a demonstration model trained on historical data. "
            "It should not be used as production credit policy or financial advice."
        )
