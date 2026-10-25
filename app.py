"""Streamlit app: predict flood probability from the 20 risk factors."""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from predict import load_model, predict_one  # noqa: E402

st.set_page_config(page_title="Flood Prediction", page_icon="🌊")
st.title("🌊 Flood Probability Predictor")
st.write("Set each risk factor (0 = very low, 16 = very high) and see the predicted flood probability.")

model, features = load_model()

values = {}
cols = st.columns(2)
for i, feat in enumerate(features):
    values[feat] = cols[i % 2].slider(feat, 0, 16, 5)

if st.button("Predict"):
    prob = predict_one(values)
    st.metric("Flood probability", f"{prob:.1%}")
    st.progress(prob)
    if prob < 0.4:
        st.success("Low flood risk")
    elif prob < 0.55:
        st.warning("Moderate flood risk")
    else:
        st.error("High flood risk")
