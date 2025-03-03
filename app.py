import streamlit as st
import numpy as np
import pickle
import os

st.title("Diabetes Prediction App")

# ✅ Use relative path for the model
model_path = "diabetes_svc_model.pkl"

# ✅ Handle missing model errors
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Model loading failed! Error: {e}")
else:
    st.error("❌ Model file not found! Ensure `diabetes_svc_model.pkl` is in the same directory as `app.py`.")

# UI Inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
age = st.number_input("Age", min_value=1, max_value=120)

# Prediction
if st.button("Predict Diabetes"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    if "model" in locals():  # Check if model loaded
        prediction = model.predict(features)
        if prediction == 1:
            st.write("The model predicts that the person **has diabetes**.")
        else:
            st.write("The model predicts that the person **does not have diabetes**.")
    else:
        st.error("❌ Model is not loaded. Please check the model file.")
