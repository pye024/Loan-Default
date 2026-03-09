import joblib
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

@st.cache_resource
def load_models():
    xgboost = joblib.load(MODELS_DIR / 'xgboost.pkl')
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    return xgboost, scaler