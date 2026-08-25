app_code = """
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("📞 Telecom Customer Churn Prediction App")
st.write("This app uses your trained **Stacking Ensemble Machine Learning Model** to predict customer churn risk.")

# Load your actual trained model safely
@st.cache_resource
def load_model():
    with open("stacking_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.sidebar.header("Customer Input Features")

# Collect user inputs matching your dataset's key features
tenure_months = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=120, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=800.0, step=10.0)
nps_score = st.sidebar.slider("NPS Score", min_value=-80, max_value=100, value=20)
overage_charges = st.sidebar.number_input("Overage Charges ($)", min_value=0.0, value=20.0, step=1.0)
network_issues = st.sidebar.slider("Network Issues (Last 3M)", min_value=0, max_value=10, value=1)
avg_data_gb = st.sidebar.number_input("Avg Data GB / Month", min_value=0.0, value=25.0, step=1.0)
service_rating = st.sidebar.slider("Service Rating (Last 6M)", min_value=1.0, max_value=5.0, value=3.8, step=0.1)

contract_type = st.sidebar.selectbox("Contract Type", ["No Contract", "Month-to-Month", "1 Year", "2 Year"])
plan_type = st.sidebar.selectbox("Plan Type", ["Prepaid", "Postpaid"])

if st.button("Predict Churn Risk (Using Real Model)"):
    try:
        # Read from the Excel (.xls) file template instead of CSV
        df_template = pd.read_excel("ultimate_telecom_churn_features.xls", nrows=1)
        X_template = df_template.drop(columns=["is_churn", "customer_id"], errors="ignore")
        X_input = X_template.copy()
        X_input.loc[0, :] = 0
        
        # Populate user inputs into the input row
        if "tenure_months" in X_input.columns: X_input.loc[0, "tenure_months"] = tenure_months
        if "monthly_charges" in X_input.columns: X_input.loc[0, "monthly_charges"] = monthly_charges
        if "total_charges" in X_input.columns: X_input.loc[0, "total_charges"] = total_charges
        if "nps_score" in X_input.columns: X_input.loc[0, "nps_score"] = nps_score
        if "overage_charges" in X_input.columns: X_input.loc[0, "overage_charges"] = overage_charges
        if "network_issues_3m" in X_input.columns: X_input.loc[0, "network_issues_3m"] = network_issues
        if "avg_data_gb_month" in X_input.columns: X_input.loc[0, "avg_data_gb_month"] = avg_data_gb
        if "service_rating_last_6m" in X_input.columns: X_input.loc[0, "service_rating_last_6m"] = service_rating
        
        # Add engineered features if they exist
        if "overage_ratio" in X_input.columns:
            X_input.loc[0, "overage_ratio"] = overage_charges / (monthly_charges + 1)
        if "spend_per_tenure" in X_input.columns:
            X_input.loc[0, "spend_per_tenure"] = total_charges / (tenure_months + 1)
            
        # Make live prediction using your real Stacking Ensemble model!
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
