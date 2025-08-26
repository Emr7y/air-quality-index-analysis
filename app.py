import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="AQI Category Predictor", layout="centered")

st.title("🌫️ AQI Category Predictor (Delhi – 2019)")
st.caption("Modell: GradientBoostingClassifier | Datenquelle: Kaggle (Rohan Rao)")

@st.cache_resource
def load_artifacts():
    data = joblib.load("aqi_bucket_model.pkl")
    return data["model"], data["label_encoder"], data["features"], data["classes"]

model, label_encoder, features, classes = load_artifacts()

st.sidebar.header("🔧 Eingaben (Schadstoffwerte)")
st.sidebar.caption("Einheiten wie im Datensatz (µg/m³)")

defaults = {
    "PM2.5": 80.0,
    "PM10": 150.0,
    "NO2": 40.0,
    "SO2": 12.0,
    "CO": 1.0,
    "O3": 40.0,
    "NH3": 20.0
}

inputs = []
for feat in features:
    val = st.sidebar.number_input(feat, value=float(defaults.get(feat, 0.0)), step=1.0, format="%.2f")
    inputs.append(val)

X = pd.DataFrame([inputs], columns=features)

if st.button("🔮 Vorhersagen"):
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = classes[pred_idx]

    st.success(f"**Vorhergesagte AQI-Kategorie:** {pred_label}")
    st.write("**Wahrscheinlichkeiten je Kategorie:**")
    st.bar_chart(pd.DataFrame({"Wahrscheinlichkeit": proba}, index=classes))

st.markdown("---")
st.caption("Made with ❤️ by Emr7y | Daten: Kaggle (Rohan Rao) | Inspiration: AmanXai Blog")
