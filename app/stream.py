import pandas as pd
import joblib as jb
import streamlit as st
import numpy as np
import time
from pathlib import Path
st.set_page_config(layout="wide")


from app import load_models

st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'raw.csv'
MODELS_DIR = BASE_DIR / 'models'

df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw.dropna().reset_index(drop=True)
df_raw = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)

xgboost, scaler = load_models()

home_map = {'RENT': 3, 'OWN': 2, 'OTHER': 1, 'MORTGAGE': 0}
intent_map = {'EDUCATION': 1, 'MEDICAL': 3, 'VENTURE': 5, 'PERSONAL': 4, 'HOMEIMPROVEMENT': 2, 'DEBTCONSOLIDATION': 0}
grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
default_map = {'Y': 1, 'N': 0}

if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'sample_index' not in st.session_state:
    st.session_state.sample_index = 0
#App Title and Description
st.title("💳 Credit Risk Prediction")
st.markdown("Predict whether a person is likely to default on a loan based on their financial profile.")
col3, col4 = st.columns(2)
with col3:
    if st.button("▶ Start"):
        st.session_state.streaming = True
with col4:
    if st.button("⏹ Stop"):
        st.session_state.streaming = False

col5, col6 = st.columns(2)

with col5:
    total = st.session_state.sample_index
    total_defaults = sum(1 for r in st.session_state.history if r['Decision'] == '🔴 Default')
    default_rate = round(total_defaults / max(1, total) * 100, 2)

    st.metric("Total Processed", total)
    st.metric("Total Defaults", total_defaults)
    st.metric("Default Rate", f"{default_rate}%")

if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

if st.session_state.streaming:
    row = df_raw.iloc[st.session_state.sample_index]

    features = np.array([
        row['age'],
        row['income'],
        home_map[row['home_ownership']],
        row['employment_years'],
        intent_map[row['loan_purpose']],
        grade_map[row['loan_grade']],
        row['loan_amount'],
        row['interest_rate'],
        row['loan_to_income'],
        default_map[row['has_previous_default']],
        row['credit_history_years']
    ])

    scaled_features = scaler.transform([features])
    prediction = xgboost.predict(scaled_features)[0]
    proba = xgboost.predict_proba(scaled_features)[0]
    probability = proba[prediction]

    with col6:
        st.subheader("Current Applicant")
        st.write(f"**Age:** {row['age']} | **Income:** ${row['income']:,}")
        st.write(f"**Loan:** ${row['loan_amount']:,} | **Grade:** {row['loan_grade']} | **Intent:** {row['loan_purpose']}")
        if prediction == 0:
            st.success(f"✅ Likely to repay (Low Risk) — Confidence: {probability:.2%}")
        else:
            st.error(f"⚠️ Likely to default (High Risk) — Confidence: {probability:.2%}")

    st.session_state.history.append({
        'Age': row['age'],
        'Income': f"${row['income']:,}",
        'Loan Amount': f"${row['loan_amount']:,}",
        'Grade': row['loan_grade'],
        'Intent': row['loan_purpose'],
        'Decision': '🔴 Default' if prediction == 1 else '🟢 Approved',
        'Confidence': f'{probability:.1%}'
    })

    st.session_state.sample_index += 1
    if st.session_state.sample_index >= len(df_raw):
        st.session_state.sample_index = 0

    time.sleep(1.5)
    st.rerun()