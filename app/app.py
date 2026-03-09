import streamlit as st
import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

@st.cache_resource
def load_models():
    xgboost = joblib.load(MODELS_DIR / 'xgboost.pkl')
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    return xgboost, scaler

main_page = st.Page("main.py", title="Manual Prediction")
stream_page = st.Page("stream.py", title="Live Stream")
with st.sidebar:
    st.header("About this app")
    st.write("""
        This app predicts the likelihood of loan default using a machine learning model.
        Provide your financial details, and click **Predict** to get a risk estimate.
    """)
pg = st.navigation([ stream_page,main_page])
pg.run()