# 💳 Credit Default Risk Prediction  
### *End-to-End Machine Learning Pipeline + Interactive Streamlit App*

Predicting whether a loan applicant will default using a complete real-world machine learning workflow.  
This project demonstrates how financial institutions use data to make safer lending decisions.

---

## 🚀 Live Demo  
🔗 **Streamlit App:** *(Link will be added after deployment)*

---

## 🧠 Why This Project Matters

Banks and lenders need to assess **which applicants are likely to default**.  
Even a small improvement in prediction accuracy can save millions.

This project shows how ML can:

- Reduce loan default losses  
- Improve credit decisioning  
- Support automated underwriting  
- Enable risk-based pricing  

The model predicts:  
**`Status = 1` → Default**  
**`Status = 0` → Non-default**

---

# 📈 Model Performance (Test Set)

| Metric | Score |
|--------|--------|
| **ROC-AUC** | **0.87** |
| **PR-AUC** | **0.81** |
| **Recall (default class)** | **0.63** |
| **Accuracy** | **0.88** |

### What this means:
- **Strong ranking ability** → The model separates risky vs. safe applicants well.  
- **Good recall on defaults** → Critical for risk teams; better to catch risky borrowers.  

Confusion Matrix:

[[16259 546]
[ 2055 3441]]

yaml
Copy code

---

# 🔍 Key Modeling Decisions

## ✔ 1. Leakage Prevention  
To avoid unrealistic accuracy, I removed features that contain *post-underwriting* information:

- `Interest_rate_spread`  
- `rate_of_interest`  
- `Upfront_charges`  

These features leak future decisions — removing them ensures **true predictive performance**.

Identifier-like leakage was also checked by detecting columns where unique values perfectly predicted the target.

---

## ✔ 2. Train/Validation/Test Strategy  
A rigorous split ensures robustness:

- **70%** training  
- **15%** validation  
- **15%** test  
- Stratified by target distribution  
- Group-aware splitting used when `ID` column existed  
- Threshold tuned to maximize **F1-score**, not accuracy  

---

## ✔ 3. Machine Learning Pipeline  

**Preprocessing (ColumnTransformer):**

- Numerical  
  - Median imputation  
- Categorical  
  - Most frequent imputation  
  - OneHotEncoding  

**Models Tested**

- Logistic Regression (baseline)  
- **XGBoost (selected model)**  

**Production Export**

Files saved in `model_artifacts/` include:

- `credit_risk_model_FINAL.joblib`  
- `schema.json` (feature names & threshold)

---

# 🖥️ Streamlit App (Interactive Scoring)

The deployed app allows anyone to:

- Enter applicant + loan details  
- Adjust income, LTV, DTI, credit score, region, age, etc.  
- See the **probability of default**  
- Get a **Low / Medium / High** risk label  
- Watch the **risk progress bar** update in real-time  

This simulates how modern lenders implement ML-powered credit scoring.

---

# 📊 Business-Friendly Summary

This project demonstrates:

- How raw loan data is cleaned and validated  
- Identification and removal of information leaks  
- Building a fair, robust, and interpretable credit scoring model  
- Deploying the model in a real app used by analysts or loan officers  

This is **industry-grade** credit risk modeling.

---

# 🧰 Tech Stack

### **Modeling**
- Python  
- Pandas, NumPy  
- scikit-learn  
- XGBoost  

### **Deployment**
- Streamlit  
- Joblib  
- JSON schema for production inference  

### **Version Control**
- Git & GitHub  

---

# 📂 Repository Structure

credit-default-risk-ml/
│
├── app.py → Streamlit app
├── Credit_Risk.ipynb → Full EDA + modeling workflow
├── Loan_Default_Data.csv → Dataset (optional for reproducibility)
├── model_artifacts/
│ ├── credit_risk_model_FINAL.joblib
│ └── schema.json
├── requirements.txt
└── README.md

yaml
Copy code

---

# 🏁 How to Run Locally

### 1. Clone repo  
```bash
git clone https://github.com/muriel1010/credit-default-risk-ml.git
cd credit-default-risk-ml
2. Create & activate virtual environment
bash
Copy code
python -m venv .venv
.venv\Scripts\activate   # Windows
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Run Streamlit app
bash
Copy code
streamlit run app.py
👩‍💼 About the Author
Muriel Tema
Data Analyst | Machine Learning | Financial Modeling
Passionate about data-driven decision-making and risk analytics.

