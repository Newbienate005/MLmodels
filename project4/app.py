import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Student Performance Prediction App")
st.write("Enter student details below to predict if they will PASS or FAIL.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
hours_studied = st.number_input("Hours Studied per Week", min_value=0.0, max_value=50.0, step=0.5)
attendance_percent = st.slider("Attendance Percentage", 0, 100, 75)
assignments_completed = st.number_input("Assignments Completed", min_value=0, max_value=2)

# Convert gender using encoder
gender_encoded = label_encoder.transform([gender])[0]

# Create dataframe for prediction
input_data = pd.DataFrame({
    "gender": [gender_encoded],
    "hours_studied": [hours_studied],
    "attendance_percent": [attendance_percent],
    "assignments_completed": [assignments_completed]
})

if st.button("Predict Performance"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"Prediction: PASS (Confidence: {probability:.2f})")
    else:
        st.error(f"Prediction: FAIL (Confidence: {probability:.2f})")

st.write("Model Ready.")
