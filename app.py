import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("📞 Telecom Customer Churn Prediction App")
st.write("This app predicts whether a telecom customer is likely to churn.")

# Input fields
tenure_months = st.slider("Tenure (Months)", min_value=1, max_value=120, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=800.0)
nps_score = st.slider("NPS Score", min_value=-80, max_value=100, value=20)
overage_charges = st.number_input("Overage Charges ($)", min_value=0.0, value=20.0)
network_issues = st.slider("Network Issues (Last 3M)", min_value=0, max_value=10, value=1)

if st.button("Predict Churn Risk"):
    if monthly_charges > 80 or network_issues > 3:
        churn_prob = 0.68
    else:
        churn_prob = 0.22
        
    st.subheader("Prediction Result:")
    if churn_prob > 0.40:
        st.error(f"⚠️ High Risk of Churn! (Probability: {churn_prob*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk / Retained (Probability: {churn_prob*100:.1f}%)")