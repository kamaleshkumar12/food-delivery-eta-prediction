
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATHS = [
    MODEL_DIR / "eta_lgbm_final.joblib",
    BASE_DIR / "eta_lgbm_final.joblib",
]

META_PATHS = [
    MODEL_DIR / "final_meta.json",
    BASE_DIR / "final_meta.json",
]


def find_file(paths):
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_model():
    path = find_file(MODEL_PATHS)
    if path is None:
        raise FileNotFoundError(
            "Model file not found. Place eta_lgbm_final.joblib "
            "inside the models/ folder."
        )
    return joblib.load(path)


def load_metadata():
    path = find_file(META_PATHS)
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


st.set_page_config(
    page_title="Food Delivery ETA Prediction",
    page_icon="🛵",
    layout="centered",
)

st.title("🛵 Food Delivery ETA Prediction")
st.write(
    "Estimate delivery duration using a trained machine learning model."
)

st.info(
    "Enter delivery details below. Predictions are estimates and "
    "may differ from actual delivery times."
)

try:
    model = load_model()
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.subheader("Delivery Details")

with st.form("eta_prediction_form"):
    distance_km = st.number_input(
        "Delivery distance (km)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.5,
    )

    age = st.number_input(
        "Delivery person age",
        min_value=18,
        max_value=70,
        value=30,
    )

    rating = st.number_input(
        "Delivery person rating",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.1,
    )

    vehicle_condition = st.selectbox(
        "Vehicle condition",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Poor",
            1: "Average",
            2: "Good",
        }[x],
    )

    weather = st.selectbox(
        "Weather",
        ["Sunny", "Cloudy", "Windy", "Fog", "Rainy", "Stormy"],
    )

    traffic = st.selectbox(
        "Traffic",
        ["Low", "Medium", "High", "Jam"],
    )

    multiple_deliveries = st.selectbox(
        "Number of simultaneous deliveries",
        [0, 1, 2, 3],
    )

    order_hour = st.slider(
        "Order hour (24-hour clock)",
        min_value=0,
        max_value=23,
        value=18,
    )

    submitted = st.form_submit_button("Predict delivery time")

if submitted:
    st.warning(
        "Demo status: prediction requires the exact feature columns "
        "and preprocessing used during training."
    )

    st.write(
        "Your saved model and training notebook are available, but "
        "the input mapping must be verified against the training "
        "pipeline before a reliable prediction can be displayed."
    )

    st.caption(
        "Do not use estimates from mismatched input features. "
        "We will connect the verified training features next."
    )
