import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Check My Tumor")

# Example: Replace with your actual feature names
feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']  # Fill with all features used in X

inputs = []
for feature in feature_names:
    value = st.number_input(f"Input {feature}", min_value=0.0)
    inputs.append(value)

if st.button("Predict"):
    scaled_features = scaler.transform(np.array(inputs).reshape(1, -1))
    prediction = model.predict(scaled_features)
    result = 'Malignant' if prediction[0]==1 else 'Benign'
    st.success(f"The tumor is predicted to be: {result}")
