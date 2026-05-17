import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Volatility Prediction", layout="wide")

st.title("MS CHOOSER OPTION - Volatility Prediction Dashboard")

st.markdown("---")
st.subheader("Model Evaluation Results")
if os.path.exists("reports/regime_test_results.csv"):
    regime_df = pd.read_csv("reports/regime_test_results.csv")
    st.dataframe(regime_df, use_container_width=True)
else:
    st.error("Run machine_learning.py first")

st.markdown("---")
st.subheader("SHAP Feature Importance")
if os.path.exists("plots/shap_importance.png"):
    st.image("plots/shap_importance.png", use_container_width=True)
else:
    st.error("Run machine_learning.py first")