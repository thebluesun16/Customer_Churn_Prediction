import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📞", layout="wide")

# ── Gemini setup ──────────────────────────────────────────────
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("telecom_model.pkl", "rb") as f:
        model, feature_columns = pickle.load(f)
    return model, feature_columns

model, feature_columns = load_model()

# ── Preprocessing ─────────────────────────────────────────────
def preprocess_input(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df

# ── Risk tier helper ──────────────────────────────────────────
def get_risk_tier(prob):
    if prob < 0.35:
        return "🟢 Low Risk", "#2ecc71"
    elif prob < 0.65:
        return "🟡 Medium Risk", "#f39c12"
    else:
        return "🔴 High Risk", "#e74c3c"

# ── GenAI retention strategy ──────────────────────────────────
def generate_retention_strategy(inputs, churn_prob, risk_tier):
    prompt = f"""
You are a telecom customer retention expert.
A customer is flagged as '{risk_tier}' with {churn_prob:.0%} churn probability.

Customer Profile:
- Contract: {inputs['Contract']}
- Monthly Charges: ${inputs['MonthlyCharges']}
- Tenure: {inputs['tenure']} months
- Internet Service: {inputs['InternetService']}
- Payment Method: {inputs['PaymentMethod']}
- Senior Citizen: {inputs['SeniorCitizen']}

Give 3 specific, actionable retention strategies. Max 3 bullet points. Be concise and practical.
"""
    response = gemini_model.generate_content(prompt)
    return response.text

# ── UI ────────────────────────────────────────────────────────
st.title("📞 Telecom Customer Churn Predictor")
st.markdown("Predict churn risk, understand why, and get AI-powered retention strategies.")
st.divider()

tab1, tab2 = st.tabs(["🔍 Predict Customer", "📊 Global Insights"])

# ── Tab 2: Global Insights ────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Feature Importance")
        st.caption("Which features matter most overall for churn prediction")
        st.image("shap_feature_importance.png", width=500)

        st.markdown("### Customer Risk Segmentation")
        st.caption("Distribution of customers across risk tiers")
        st.image("risk_segmentation.png", width=500)

    with c2:
        st.markdown("### Feature Impact Direction (Beeswarm)")
        st.caption("Pink = high feature value, Blue = low. Right = increases churn risk, Left = decreases")
        st.image("shap_summary.png", width=500)

# ── Tab 1: Predict ────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Customer Info")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen?", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)

    with col2:
        st.markdown("#### Service Info")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    st.divider()

    if st.button("🔍 Predict & Analyse", use_container_width=True):

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
            prob = model.predict_proba(processed)[0][1]
            risk_label, risk_color = get_risk_tier(prob)

            # ── Result ────────────────────────────────────────
            st.markdown("## Results")
            r1, r2, r3 = st.columns(3)

            with r1:
                if pred == 1:
                    st.error("⚠️ Customer WILL Churn")
                else:
                    st.success("✅ Customer will NOT Churn")

            with r2:
                st.metric("Churn Probability", f"{prob:.1%}")

            with r3:
                st.markdown(f"**Risk Tier:** {risk_label}")

            st.divider()

            # ── SHAP Explanation ──────────────────────────────
            st.markdown("### 🧠 Why this prediction? (SHAP)")
            try:
                base_clf = model.named_steps['model']
                scaler = model.named_steps['scaler']
                processed_scaled = scaler.transform(processed)

                explainer = shap.TreeExplainer(base_clf)
                shap_values = explainer.shap_values(processed_scaled)

                sv_raw = np.array(shap_values)
                if sv_raw.ndim == 3:
                    sv = sv_raw[0, :, 1]
                elif sv_raw.ndim == 2:
                    sv = sv_raw[0]
                else:
                    sv = sv_raw

                shap_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'SHAP Value': sv
                }).sort_values('SHAP Value', key=abs, ascending=False).head(10)

                colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_df['SHAP Value']]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_xlabel("SHAP Value (Red = increases churn risk, Green = decreases)")
                ax.set_title("Top Features Driving This Prediction")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Red bars push towards churn. Green bars push away from churn.")

            except Exception as shap_err:
                st.info(f"SHAP explanation unavailable: {shap_err}")

            st.divider()

            # ── GenAI Retention Strategy ──────────────────────
            st.markdown("### 🤖 AI-Powered Retention Strategy")

            with st.spinner("Generating retention strategy..."):
                strategy = generate_retention_strategy(inputs, prob, risk_label)
            st.markdown(strategy)

        except Exception as e:
            st.error(f"Prediction error: {e}")
