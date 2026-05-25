import streamlit as st
import joblib
import pandas as pd
from tensorflow import keras

# -----------------------------
# LOAD MODELS
# -----------------------------
rf_model = joblib.load("tourism_model.pkl")

dl_preprocessor = joblib.load("dl_preprocessor.pkl")

dl_model = keras.models.load_model(
    "satisfaction_model.h5",
    compile=False
)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🌍 Smart Tourism AI System")
st.write("AI-powered travel recommendation + prediction")

# -----------------------------
# USER INPUTS
# -----------------------------
budget = st.selectbox("Budget Index", ["Low", "Medium", "High"])
weather = st.selectbox("Weather Profile", ["Cold", "Mild", "Hot"])
trip = st.selectbox("Trip Style", ["Adventure", "Family", "Cultural", "Beach", "City"])
month = st.selectbox(
    "Peak Month",
    ["January","February","March","April","May","June",
     "July","August","September","October","November","December"]
)
crowd = st.selectbox("Crowd Profile", ["Low", "Medium", "High"])

# -----------------------------
# DESTINATION PREDICTION (ML)
# -----------------------------
if st.button("🌍 Recommend Destination"):

    input_df = pd.DataFrame([[budget, weather, trip, month]],
                            columns=["Budget Index","Weather Profile","Trip Style","Peak Month"])

    pred = rf_model.predict(input_df)

    st.success(f"Recommended Destination: {pred[0]}")

# -----------------------------
# DEEP LEARNING: SATISFACTION
# -----------------------------
if st.button("⭐ Predict Satisfaction"):

    dl_input = pd.DataFrame([[budget, weather, trip, month, crowd]],
                            columns=["Budget Index","Weather Profile","Trip Style","Peak Month","Crowd Profile"])

    dl_input_processed = dl_preprocessor.transform(dl_input)

    prediction = dl_model.predict(dl_input_processed)[0][0]

    st.info(f"Predicted Satisfaction Score: {round(float(prediction),2)} %")

# -----------------------------
# FOOTER
# -----------------------------
st.write("---")
st.write("🚀 Built by Smart Tourism AI System (ML + DL Project)")
