# 📉 Telecom Customer Churn Prediction

A machine learning project that predicts customer churn for a telecom company using the IBM Telco Customer Churn dataset. The model is deployed as an interactive Streamlit web application.

---

## 🔍 What's Been Done

**Exploratory Data Analysis** - Analyzed all 21 features across 7,043 customer records, visualized churn patterns across categorical and numerical variables (tenure, monthly charges, total charges), and identified key insight: customers with lower monthly charges are more likely to churn.

**Data Preprocessing** - Fixed `TotalCharges` dtype, dropped nulls and the `customerID` column, and encoded all categorical features using OneHotEncoder.

**Model Training & Evaluation** - Trained and compared 5 classifiers (Random Forest, SVM, Logistic Regression, KNN, Decision Tree) using GridSearchCV for hyperparameter tuning. Applied **SMOTEENN** to handle class imbalance and re-evaluated all models on the resampled data.

**Deployment** - Best model saved as `telecom_model.pkl` and served via a Streamlit app.

🚀 **Live App:** [Click here](https://churnpredictionmodel-rrxi7d6izqpafhzhnv5hzp.streamlit.app/)

---

## 🚧 Work In Progress

- [ ] **Explainable AI (XAI)** - Add SHAP values to explain individual predictions and surface the top churn drivers per customer
- [ ] **Customer Risk Segmentation** - Cluster customers into risk tiers (high / medium / low) for prioritized action
- [ ] **AI-Powered Retention Recommendations** - Use a GenAI model to generate natural language retention strategies based on a customer's churn risk profile

---

## 🛠 Tech Stack

Python, Scikit-learn, Imbalanced-learn, Pandas, Matplotlib, Seaborn, Streamlit
