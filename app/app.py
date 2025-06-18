#Import Libraries and Prediction Data
import streamlit as st
import numpy as np
import joblib
import pandas as pd


xgboost=joblib.load('models/xgboost_model.pkl')
scaler=joblib.load('models/scaler.pkl')

st.title("💳 Credit Risk Prediction")
st.markdown("Predict whether a person is likely to default on a loan based on their financial profile.")


with st.sidebar:
    st.header("About this app")
    st.write("""
        This app predicts the likelihood of loan default using a machine learning model.
        Provide your financial details, and click **Predict** to get a risk estimate.
    """)
st.subheader("📊 Financial Profile Inputs")
#Inputs
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 30)
    income = st.number_input("Annual Income ($)", min_value=1000, value=50000, step=1000)
    employment_years = st.slider("Years of Employment", 0, 40, 5)
    home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    loan_purpose = st.selectbox("Loan Purpose", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOME IMPROVEMENT', 'DEBT CONSOLIDATION'])

with col2:
    loan_amount = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=10000, step=500)
    interest_rate = st.slider("Loan Interest Rate (%)", 0.0, 30.0, 10.0)
    credit_history_years = st.slider("Credit History Length (years)", 0, 30, 5)
    loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    has_previous_default = st.radio("Previous Default on File", ['Y', 'N'])

#Mapping 

home_map = {'RENT': 3, 'OWN': 2, 'OTHER': 1, 'MORTGAGE': 0}
intent_map = {'EDUCATION': 1, 'MEDICAL': 3, 'VENTURE': 5, 'PERSONAL': 4, 'HOME IMPROVEMENT': 2, 'DEBT CONSOLIDATION': 0}
grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
default_map = {'Y': 1, 'N': 0}

short_credit_history = int(credit_history_years < 2)
loan_to_income=loan_amount/income


#Prediction

features=np.array([
    age,                                    
    income,                                 
    home_map[home_ownership],               
    employment_years,                       
    intent_map[loan_purpose],               
    loan_amount,                            
    interest_rate,                          
    loan_to_income,                         
    default_map[has_previous_default],      
    credit_history_years,                   
    grade_map[loan_grade],                  
    short_credit_history
])

scaled_features = scaler.transform([features])

if st.button("Predict Credit Risk"):
    prediction = xgboost.predict(scaled_features)[0]
    probability = xgboost.predict_proba(scaled_features)[0][prediction]

    if prediction == 0:
        st.success(f"✅ Likely to **repay the loan** (Low Risk)\n\nConfidence: {probability:.2%}")
    else:
        st.error(f"⚠️ Likely to **default on the loan** (High Risk)\n\nConfidence: {probability:.2%}")