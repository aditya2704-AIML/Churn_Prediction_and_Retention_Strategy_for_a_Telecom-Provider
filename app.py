import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("📞 Telecom Customer Churn Prediction App")
st.write("This app uses your trained **Stacking Ensemble Machine Learning Model** to predict customer churn risk.")

@st.cache_resource
def load_model():
    with open("stacking_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.sidebar.header("Customer Input Features")

tenure_months = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=120, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=800.0, step=10.0)
nps_score = st.sidebar.slider("NPS Score", min_value=-80, max_value=100, value=20)
overage_charges = st.sidebar.number_input("Overage Charges ($)", min_value=0.0, value=20.0, step=1.0)
network_issues = st.sidebar.slider("Network Issues (Last 3M)", min_value=0, max_value=10, value=1)
avg_data_gb = st.sidebar.number_input("Avg Data GB / Month", min_value=0.0, value=25.0, step=1.0)
service_rating = st.sidebar.slider("Service Rating (Last 6M)", min_value=1.0, max_value=5.0, value=3.8, step=0.1)

if st.button("Predict Churn Risk (Using Real Model)"):
    try:
        feature_columns = [
            'age', 'tenure_months', 'monthly_charges', 'total_charges', 'avg_data_gb_month',
            'avg_voice_mins_month', 'sms_count_month', 'overage_charges', 'is_family_plan',
            'is_multi_service', 'network_issues_3m', 'dropped_call_rate', 'avg_data_speed_mbps',
            'num_complaints_3m', 'num_complaints_12m', 'call_center_interactions_3m',
            'last_complaint_resolution_days', 'app_logins_30d', 'selfcare_transactions_30d',
            'auto_pay_enrolled', 'late_payment_flag_3m', 'avg_payment_delay_days', 'arpu',
            'nps_score', 'service_rating_last_6m', 'received_competitor_offer_flag',
            'retention_offer_accepted_flag', 'overage_ratio', 'spend_per_tenure',
            'network_dissatisfaction', 'complaint_escalation'
        ]
        
        # Build DataFrame and force convert everything to numeric float to prevent dtype crashes
        X_input = pd.DataFrame(0.0, index=[0], columns=feature_columns)
        
        X_input.loc[0, 'tenure_months'] = float(tenure_months)
        X_input.loc[0, 'monthly_charges'] = float(monthly_charges)
        X_input.loc[0, 'total_charges'] = float(total_charges)
        X_input.loc[0, 'nps_score'] = float(nps_score)
        X_input.loc[0, 'overage_charges'] = float(overage_charges)
        X_input.loc[0, 'network_issues_3m'] = float(network_issues)
        X_input.loc[0, 'avg_data_gb_month'] = float(avg_data_gb)
        X_input.loc[0, 'service_rating_last_6m'] = float(service_rating)
        
        X_input.loc[0, 'overage_ratio'] = float(overage_charges / (monthly_charges + 1))
        X_input.loc[0, 'spend_per_tenure'] = float(total_charges / (tenure_months + 1))
        X_input.loc[0, 'network_dissatisfaction'] = float(network_issues * 0.02)
        X_input.loc[0, 'complaint_escalation'] = 0.5
        
        # Ensure all columns are numeric
        X_input = X_input.astype(float)
        
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]
        
        st.subheader("Prediction Result:")
        if prediction == 1 or probability > 0.40:
            st.error(f"⚠️ **High Risk of Churn!** (Model Probability: {probability*100:.1f}%)")
            st.markdown("**Recommended Action:** Trigger a proactive retention discount or customer support check-in.")
        else:
            st.success(f"✅ **Low Risk / Retained** (Model Probability: {probability*100:.1f}%)")
            st.markdown("**Recommended Action:** Maintain standard digital engagement and loyalty perks.")
            
    except Exception as e:
        st.error(f"Error processing prediction: {e}")
