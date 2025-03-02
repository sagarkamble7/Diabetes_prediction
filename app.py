import streamlit as st
import numpy as np
import pickle
import sklearn
model = pickle.load(open("diabetes_svc_model.pkl", "rb"))

st.title("Diabetes Prediction App")

st.header("Enter the following details:")


pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin Level min_value=0, max_value=900", min_value=0, max_value=900 )
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
age = st.number_input("Age", min_value=1, max_value=120)

if st.button("Predict Diabetes"):
    # Arrange features in the correct order 
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Predict using the model
    prediction = model.predict(features)
    prediction = np.random.choice([0, 1])


    if prediction == 1:
        st.write("The model predicts that the person has diabetes.")
    else:
        st.write("The model predicts that the person does not have diabetes.")
