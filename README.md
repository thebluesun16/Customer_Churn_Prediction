# 📉 Telecom Customer Churn Intelligence Platform

An end-to-end machine learning system that predicts customer churn, explains *why* a customer is likely to leave, segments customers by risk level, and generates AI-powered retention strategies - built on the IBM Telco Customer Churn dataset and deployed as an interactive Streamlit web application.

---

## 🔍 What's Been Done

**Exploratory Data Analysis** - Analyzed all 21 features across 7,043 customer records, visualized churn patterns across categorical and numerical variables (tenure, monthly charges, total charges), and identified key insight: customers with lower monthly charges are more likely to churn.

**Data Preprocessing** - Fixed `TotalCharges` dtype, dropped nulls and the `customerID` column, and encoded all categorical features using OneHotEncoder.

**Model Training & Evaluation** - Trained and compared 5 classifiers (Random Forest, SVM, Logistic Regression, KNN, Decision Tree) using GridSearchCV for hyperparameter tuning. Applied **SMOTEENN** to handle class imbalance and re-evaluated all models on the resampled data.

**Explainable AI (SHAP)** - Integrated SHAP values to explain individual predictions and surface the top churn drivers per customer. Includes global feature importance, beeswarm plot, and per-customer waterfall explanation.

**Customer Risk Segmentation** - Clustered customers into High / Medium / Low risk tiers using churn probability scores, enabling prioritized retention action.

**AI-Powered Retention Recommendations** - Integrated Gemini 2.5 Flash to generate personalised, actionable retention strategies based on each customer's churn risk profile and service details.

**Deployment** - Model saved as `telecom_model.pkl` and served via a Streamlit web app with two tabs — live prediction and global insights dashboard.

🚀 **Live App:** [Click here](https://customerchurnprediction-ks8axfopv784xvtxm7wra8.streamlit.app/)

---

## 📁 Project Structure

```
├── TelecomChurnPrediction.ipynb     # EDA, preprocessing, model training, SHAP, segmentation
├── app.py                           # Streamlit web application
├── telecom_model.pkl                # Serialized trained model
├── shap_feature_importance.png      # Global SHAP bar chart
├── shap_summary.png                 # SHAP beeswarm plot
├── shap_single_customer.png         # Single customer waterfall explanation
├── risk_segmentation.png            # Customer risk tier distribution
├── customer_risk_segments.csv       # Risk-labelled customer data
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🛠 Tech Stack

Python, Scikit-learn, Imbalanced-learn, XGBoost, SHAP, Pandas, Matplotlib, Seaborn, Streamlit, Google Gemini 2.5 Flash (GenAI)
