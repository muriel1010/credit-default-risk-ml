# 💳 Credit Default Risk Prediction  
### *End-to-End Machine Learning Pipeline + Interactive Streamlit App*

Predicting whether a loan applicant will default using a complete real-world machine learning workflow.  
This project demonstrates how financial institutions use data to make safer lending decisions.

---

## 📚 Table of Contents

- [ Live Demo](#-live-demo)
- [ Why This Project Matters](#-why-this-project-matters)
- [ Model Performance](#-model-performance-test-set)
- [ Key Modeling Decisions](#-key-modeling-decisions)
  - [ Leakage Prevention](#-1-leakage-prevention)
  - [ TrainValidationTest Strategy](#-2-trainvalidationtest-strategy)
  - [ ML Pipeline Architecture](#-3-machine-learning-pipeline)
- [ Streamlit App](#️-streamlit-app-interactive-scoring)
- [ Business Summary](#-business-friendly-summary)
- [ Tech Stack](#-tech-stack)
- [ Repository Structure](#-repository-structure)
- [ How to Run Locally](#-how-to-run-locally)
- [ About the Author](#-about-the-author)
- [ Final Notes](#️-final-notes)

---

## 🚀 Live Demo  
🔗 **Streamlit App:** *Link will be added after deployment*

---

##  Why This Project Matters

Banks and lenders must determine **which applicants are likely to default**.  
Even small improvements in risk modeling can save millions of dollars.

This project demonstrates how ML can:

- Reduce loan default losses  
- Improve lending decisions  
- Support automated underwriting  
- Enable risk-based pricing  

The model predicts:  
- **`Status = 1` → Default**  
- **`Status = 0` → Non-default**

---

##  Model Performance (Test Set)

| Metric | Score |
|--------|--------|
| **ROC-AUC** | **0.87** |
| **PR-AUC** | **0.81** |
| **Recall (default class)** | **0.63** |
| **Accuracy** | **0.88** |

Confusion Matrix:


| **16259** | **546** |

| **2055** | **3441** |



### Interpretation  
- Strong **ranking ability** between risky vs safe borrowers  
- High **recall** for defaults → essential for risk teams  
- Balanced performance for imbalanced financial data  

---

#  Key Modeling Decisions

## 1. Leakage Prevention

To avoid unrealistic accuracy, the following post-underwriting features were removed:

- `Interest_rate_spread`  
- `rate_of_interest`  
- `Upfront_charges`

These contain decisions made *after* risk evaluation , keeping them would artificially inflate performance.

Identifier-like leakage was also checked by identifying columns where unique values perfectly predicted the target.

---

##  2. Train/Validation/Test Strategy

A rigorous, industry-grade data-splitting strategy:

- **70%** training  
- **15%** validation  
- **15%** test  
- Stratified by target distribution  
- Group-aware splitting when `ID` column existed  
- Threshold tuned on validation set (maximize F1-score)  

---

##  3. Machine Learning Pipeline

### **Preprocessing (via ColumnTransformer)**  
- Numerical → Median imputation  
- Categorical → Most frequent imputation + OneHotEncoder  

### **Models tested**  
- Logistic Regression (baseline)  
- **XGBoost (selected model)**  

### **Production Export**  
Saved under `model_artifacts/`:

- `credit_risk_model_FINAL.joblib`  
- `schema.json` → feature names, threshold, metadata


---

#  Streamlit App (Interactive Scoring)

The deployed app allows users to:

- Input applicant + loan details  
- Adjust income, LTV, DTI, credit score, region, age  
- Generate **default probability** using the trained ML model  
- Display **Low / Medium / High Risk** categories  
- Simulate underwriting decisions  

This mirrors how risk officers interact with real production scoring tools.

---

# 📊 Business-Friendly Summary

This project demonstrates:

- Credit risk modeling following **best practices used in banking**  
- Clear identification + removal of information leaks  
- Building a fair, robust, transparent model  
- Deploying it in a usable application for analysts or credit officers  

It shows both **technical depth** and **business intuition**.

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
- JSON schema for inference  

### **Version Control**
- Git & GitHub  

---

# Repository Structure

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

---

#  How to Run Locally

### 1. Clone the repository  
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
 About the Author
Muriel Tema
Data Analyst | Machine Learning | Financial Modeling
Passionate about data-driven decision-making and credit risk analytics.

 Final Notes
This project covers the entire machine learning lifecycle — from raw data → modeling → evaluation → deployment.

It highlights:

End-to-end ML engineering

Responsible modeling (leakage prevention)

Real-world credit scoring logic

Strong communication and documentation

Perfect for demonstrating industry-ready skills to recruiters and hiring managers.

yaml
Copy code

---

### ✔ This is the **final clean version**, fully integrated, with table of contents and correct anchors.  
### ✔ You can paste it *exactly as-is* into your `README.md`, no edits required.

If you'd like screen
