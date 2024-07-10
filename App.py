import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model from the pickle file
with open('model_nb.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Heart Attack Prediction')

st.write('Please enter the following information:')

# User inputs
age = st.number_input('Age', min_value=1, max_value=120, value=30)
cholesterol = st.number_input('Cholesterol', min_value=100, max_value=400, value=200)
max_heart_rate = st.number_input('Max Heart Rate', min_value=60, max_value=220, value=120)
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'Abnormal'])
exercise_angina = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)
diet = st.selectbox('Diet', ['Healthy', 'Average', 'Unhealthy'])
systolic_bp = st.number_input('Systolic BP', min_value=90, max_value=200, value=120)
diastolic_bp = st.number_input('Diastolic BP', min_value=60, max_value=120, value=80)
sex = st.selectbox('Sex', ['Male', 'Female'])

# Encode the inputs as the model expects
diet_map = {'Healthy': 1, 'Average': 2, 'Unhealthy': 3}
sex_map = {'Male': [0, 1], 'Female': [1, 0]}

input_data = pd.DataFrame({
    'Age': [age],
    'Cholesterol': [cholesterol],
    'Max Heart Rate': [max_heart_rate],
    'Resting ECG': [1 if resting_ecg == 'Abnormal' else 0],
    'Exercise Induced Angina': [1 if exercise_angina == 'Yes' else 0],
    'Oldpeak': [oldpeak],
    'Diet': [diet_map[diet]],
    'Systolic BP': [systolic_bp],
    'Diastolic BP': [diastolic_bp],
    'Sex_Female': [sex_map[sex][0]],
    'Sex_Male': [sex_map[sex][1]]
})

# Reorder the columns to match the training data
input_data = input_data[['Age', 'Cholesterol', 'Max Heart Rate', 'Resting ECG', 
                         'Exercise Induced Angina', 'Oldpeak', 'Diet', 'Systolic BP', 
                         'Diastolic BP', 'Sex_Female', 'Sex_Male']]

if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    st.write(f'The predicted heart attack risk is: {prediction}')
