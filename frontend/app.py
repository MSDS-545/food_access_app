import os
import streamlit as st
import requests

# Get API URL from env var or default
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Food Access Model Prediction")
st.write("Input features to predict whether a tract is Low Income & Low Access")

# Using the correct feature names from backend
hunv_flag    = st.number_input("HUNVFlag (households with no vehicle)", min_value=0.0, value=0.0, key="hunv_flag_input")
poverty_rate = st.number_input("PovertyRate (percent below poverty level)", min_value=0.0, max_value=100.0, value=25.0, key="poverty_rate_input")
la1and10     = st.number_input("LA1and10 (low-access flag: 1 mile urban / 10 miles rural)", min_value=0.0, value=0.0, key="la1and10_input")

if st.button("Predict", key="predict_button"):
    endpoint = "/predict/simple"
    url = f"{API_URL}{endpoint}"
    payload = {
        "HUNVFlag":   hunv_flag,
        "PovertyRate": poverty_rate,
        "LA1and10":    la1and10
    }

    # Debug logs
    st.write("## Debug Info")
    st.write("Request URL:", url)
    st.write("Payload:", payload)

    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        result = resp.json()
        st.write("## Prediction Result")
        st.write(result)
        st.success(
            f"Predicted Class: {result['predicted_class']} (1 = LILA)\n"
            f"Probability: {result['probability']:.2f}"
        )
    except Exception as err:
        st.error(f"Error: {err}")
