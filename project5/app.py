import streamlit as st
import pandas as pd
import joblib

# Load trained artifacts
model = joblib.load("nb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Battery Drain Prediction", layout="centered")
st.title("Battery Drain Prediction App")
st.write("Predict whether a smartphone is likely to experience **High Battery Drain** based on usage behavior.")

#USER INPUTS (MATCH TRAINING FEATURES) 

daily_app_opens = st.number_input(
    "Daily App Opens",
    min_value=0,
    max_value=500,
    value=120
)

daily_screen_time_min = st.number_input(
    "Daily Screen Time (Minutes)",
    min_value=0,
    max_value=1440,
    value=300
)

notifications_received = st.number_input(
    "Notifications Received per Day",
    min_value=0,
    max_value=1000,
    value=80
)

primary_app_category_encoded = st.selectbox(
    "Primary App Category",
    options=[0, 1, 2, 3, 4],
    format_func=lambda x: {
        0: "Social Media",
        1: "Gaming",
        2: "Productivity",
        3: "Entertainment",
        4: "Other"
    }[x]
)

#INPUT DATAFRAME 

input_df = pd.DataFrame(
    [[
        daily_app_opens,
        daily_screen_time_min,
        notifications_received,
        primary_app_category_encoded
    ]],
    columns=[
        "Daily_App_Opens",
        "Daily_Screen_Time_Min",
        "Notifications_Received",
        "Primary_App_Category_Encoded"
    ]
)

# Scale input
# Force column order to match training
input_df = input_df[scaler.feature_names_in_]

input_scaled = scaler.transform(input_df)


# PREDICTION 

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"High Battery Drain Likely (Probability: {probability:.2%})")
    else:
        st.success(f"Normal Battery Usage (Probability of High Drain: {probability:.2%})")

st.caption("Model: Gaussian Naive Bayes | Preprocessing: StandardScaler")

