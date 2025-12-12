#  Credit Default Risk Prediction  Machine Learning Solution

## Table of Contents
- [Project Overview](#project-overview)
- [Live Application](#live-application)
- [Solution Design](#solution-design)
- [Model Performance](#model-performance)
- [Technical Approach](#technical-approach)
- [Repository Structure](#repository-structure)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)
- [Author](#author)

---

## Project Overview
Financial institutions must evaluate thousands of loan applications each day while minimizing risk exposure.  
Default prediction is challenging due to imbalanced data, inconsistent borrower profiles, leakage-prone features, and the need for fast, explainable, and repeatable decisions.

This project provides a complete, production-ready machine learning solution that:

- Cleans and validates applicant and loan data  
- Removes variables that introduce target leakage  
- Builds a preprocessing and modeling pipeline  
- Trains an XGBoost classifier to estimate default probability  
- Evaluates performance using industry-standard metrics  
- Deploys a real-time scoring tool through a Streamlit web interface  

The result is a reliable system that assists lenders in making faster, more consistent, and data-driven credit decisions.

---

## Live Application
Access the interactive real-time scoring tool:  
[https://credit-default-risk-ml-t9u88ebhhhpugqqmakpvdc.streamlit.app/]

*Solution ScreenShot*  
![Application Screenshot](https://github.com/muriel1010/credit-default-risk-ml/blob/main/applicant-details.png)

---

## Solution Design

Users input applicant and loan characteristics (e.g., income, LTV, credit score, region), and the model returns:

- **Probability of default** : a numeric score between 0 and 1  
- **Risk classification** :(Low, Medium, High), based on an optimized decision threshold

  *Solution Prediction ScreenShot*  
![Sample Prediction Screenshot](https://github.com/muriel1010/credit-default-risk-ml/blob/main/prediction.png)
This enables lenders to understand both the *predicted likelihood* and the *interpreted risk level* instantly.
---

## Model Performance
Performance on the held-out test dataset:

- **ROC-AUC:** 0.87  
- **PR-AUC:** 0.81  
- **Recall (default class):** 0.63  
- **Accuracy:** 0.88  

These results indicate strong ability to detect high-risk applicants, supporting improved decision-making and loss mitigation.

---

## Technical Approach

### Data Preparation
- Missing value handling using imputation  
- Removal of interest-based pricing fields that leak target information  
- Deduplication and validation  
- Exploratory analysis to identify key risk drivers  

### Modeling Workflow
- Stratified train/validation/test split  
- Pipeline including imputation and one-hot encoding  
- XGBoost classifier tuned for imbalanced data  
- Threshold selection based on F1 optimization  
- Evaluation via ROC-AUC, precision-recall curves, and confusion matrix  

### Deployment
- Model exported using joblib  
- Schema JSON ensures input consistency  
- Real-time scoring app built with Streamlit  

---

## Repository Structure
```bash
credit-default-risk-ml/
│
├── app.py                        # Streamlit application
├── model_artifacts/
│     ├── credit_risk_model_FINAL.joblib   # Serialized trained model
│     └── schema.json                       # Input schema for validation
├── Credit_Risk.ipynb               # EDA and model development notebook
├── requirements.txt                # Dependencies for deployment
├── README.md                       # Project documentation
└── Loan_Default_Data.csv           # (Optional) dataset for reproducibility


```
---
## Future Enhancements

- Add SHAP or LIME for model explainability  
- Deploy an API endpoint (FastAPI) for integration with external systems
- Introduce model monitoring for drift and stability 
- Apply hyperparameter optimization using Optuna  
- Enable batch scoring for loan portfolios 
- Add dashboards to monitor risk distribution and model outputs

---
## Conclusion

This project demonstrates the full lifecycle of an applied machine learning system for credit risk assessment.
From raw data to a deployed scoring interface, it showcases practical experience in data preprocessing, leakage prevention, model development, evaluation, and real-time deployment aligning closely with the workflows used in financial institutions and modern data teams.

---
## Author
**Muriel Tema**  
Data Analyst


