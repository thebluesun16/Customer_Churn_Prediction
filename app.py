import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    with open("telecom_model.pkl", "rb") as f:
        model, feature_columns = pickle.load(f)
    return model, feature_columns

model, feature_columns = load_model()

def preprocess_input(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df

# UI for all inputs
st.title("📞 Telco Customer Churn Prediction App")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen?", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

if st.button("Predict"):
    inputs = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": charges,
        "InternetService": internet,
        "DeviceProtection": device_protection,
        "Contract": contract,
        "PaymentMethod": payment
    }

    try:
        processed = preprocess_input(inputs)
        pred = model.predict(processed)[0]
        st.success("✅ Customer will NOT churn." if pred == 0 else "⚠️ Customer WILL churn.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
