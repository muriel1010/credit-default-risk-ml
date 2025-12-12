# 💳 Credit Default Risk Prediction  

## Table of Contents
- [Executive Summary](#executive-summary)
- [Business Context](#business-context)
- [Solution Overview](#solution-overview)
- [Model Performance](#model-performance)
- [Live Application](#live-application)
- [Technical Approach](#technical-approach)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## Executive Summary
This project delivers a complete machine learning solution for predicting the likelihood that a loan applicant will default.  
It includes data preprocessing, leakage detection, feature engineering, model development, evaluation, and deployment through a real-time scoring application.

The project demonstrates practical skills in data analytics, supervised learning, risk modeling, and ML engineering.

---

## Business Context
Financial institutions evaluate thousands of loan applications and must balance growth with risk management.  
Accurate risk prediction enables organizations to:

- Make faster and more consistent credit decisions  
- Reduce financial losses from high-risk applicants  
- Improve underwriting efficiency  
- Support risk-based pricing strategies  

This project provides an automated scoring system that addresses these needs.

---

## Solution Overview
The solution combines a production-ready machine learning pipeline with an interactive web application. It includes:

- Data cleaning and validation  
- Detection and removal of leakage-prone fields  
- Preprocessing via a robust feature pipeline  
- Training an XGBoost classifier to estimate default probability  
- Evaluation using standard risk-modeling metrics  
- Deployment through a Streamlit interface  

Users can input loan and applicant information to receive a probability of default along with a risk classification.

---

## Model Performance
Performance on the test set:

- ROC-AUC: 0.87  
- PR-AUC: 0.81  
- Recall (default class): 0.63  
- Accuracy: 0.88  

These results indicate strong ability to identify high-risk applicants and support better decision-making.

---

## Live Application
Real-time scoring interface:  
[Insert Streamlit App Link Here]

*(Optional screenshot placeholder)*  
![Application Screenshot](path/to/screenshot.png)

---

## Technical Approach

### Data Preparation
- Missing value handling  
- Leakage detection (removal of interest-based pricing columns)  
- Deduplication  
- Exploratory analysis of credit risk indicators  

### Modeling
- Stratified train/validation/test split  
- Pipeline with imputation + one-hot encoding  
- XGBoost classifier with imbalance handling  
- F1-based threshold tuning  
- Evaluation using ROC-AUC, PR curve, recall, accuracy, and confusion matrix  

### Deployment
- Trained model exported with joblib  
- Schema JSON used to validate inference inputs  
- Streamlit application for real-time scoring  

---

## System Architecture

User Input (Streamlit App)
↓
Preprocessing Pipeline (Imputation + Encoding)
↓
XGBoost Model (Serialized)
↓
Probability of Default
↓
Risk Classification

yaml
Copy code

*(Optional architecture image placeholder)*  
![Architecture Diagram](path/to/architecture.png)

---

## Repository Structure

credit-default-risk-ml/
│
├── app.py # Streamlit application
├── model_artifacts/
│ ├── credit_risk_model_FINAL.joblib
│ └── schema.json
├── Credit_Risk.ipynb # EDA & modeling notebook
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── Loan_Default_Data.csv # Optional dataset

yaml
Copy code

---

## Future Enhancements
- Add SHAP or LIME to provide model interpretability  
- Deploy an API version using FastAPI  
- Implement monitoring for model drift and data quality  
- Perform automated hyperparameter optimization  
- Add batch scoring functionality for larger portfolios  
- Build dashboards for ongoing risk analytics  

---

## Author
**Muriel Tema**  
Data Analyst



