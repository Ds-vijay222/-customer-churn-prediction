import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Models Load Karo

@st.cache_resource
def load_model():
    with open("models/churn_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/threshold.json", "r") as f:
        threshold = json.load(f)['threshold']
    return model, scaler, threshold

model, scaler, threshold = load_model()


# App Title

st.title("🔮 Customer Churn Prediction")
st.markdown("###  It Predict customer churn risk")
st.divider()


# User Input
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen ?", ["No", "Yes"])
    partner = st.selectbox("Partner ?", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type",
                ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"])

with col2:
    st.subheader("📱 Services")
    phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines?",
                ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service?",
                ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security?",
                ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup?",
                ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection?",
                ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support?",
                ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV?",
                ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies?",
                ["Yes", "No", "No internet service"])

st.divider()
st.subheader("💰 Charges")
col3, col4 = st.columns(2)
with col3:
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
with col4:
    total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

st.divider()


# Predict Button
if st.button("🔮 Predict Churn!", use_container_width=True):

    # Input Data Prepare
    input_data = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
        'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
        'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
        'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
        'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
    }

    # DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale 
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[scale_cols] = scaler.transform(input_df[scale_cols].values)

    # Predict 
    probability = model.predict_proba(input_df)[:, 1][0]
    prediction = 1 if probability > threshold else 0

    # Results 
    st.divider()
    st.subheader("📊 Prediction Result")

    col5, col6 = st.columns(2)
    with col5:
        if prediction == 1:
            st.error("⚠️ HIGH RISK — Customer Churn!")
        else:
            st.success("✅ LOW RISK — Customer Retain !")
    with col6:
        st.metric(
            label="Churn Probability",
            value=f"{probability*100:.1f}%"
        )

    # Progress Bar
    st.progress(float(probability))

    # Recommendations
    st.divider()
    if prediction == 1:
        st.warning("""
        ### 💡 Recommended Actions:
        - 🎁 Special discount offer 
        - 📞 Customer care call 
        - 📋 Long term contract offer
        - 💰 Better plan suggest
        """)
    else:
        st.info("""
        ### 💡 Customer is  Loyal
        - ⭐ Loyalty rewards
        - 🔄  better plan offer 
        """)