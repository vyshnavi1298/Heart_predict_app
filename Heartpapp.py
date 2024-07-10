import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title('Heart Attack Prediction')

# Input features for prediction
age = st.number_input('Age', min_value=0, max_value=120, value=30)
cholesterol = st.number_input('Cholesterol', min_value=0, max_value=1000, value=200)
heart_rate = st.number_input('Heart Rate', min_value=0, max_value=200, value=70)
diabetes = st.selectbox('Diabetes', [0, 1])
family_history = st.selectbox('Family History', [0, 1])
smoking = st.selectbox('Smoking', [0, 1])
obesity = st.selectbox('Obesity', [0, 1])
alcohol_consumption = st.selectbox('Alcohol Consumption', [0, 1])
exercise_hours_per_week = st.number_input('Exercise Hours Per Week', min_value=0.0, max_value=168.0, value=3.5)
diet = st.selectbox('Diet', [1, 2, 3])  # 1=Healthy, 2=Average, 3=Unhealthy
previous_heart_problems = st.selectbox('Previous Heart Problems', [0, 1])
medication_use = st.selectbox('Medication Use', [0, 1])
stress_level = st.number_input('Stress Level', min_value=0, max_value=10, value=5)
sedentary_hours_per_day = st.number_input('Sedentary Hours Per Day', min_value=0.0, max_value=24.0, value=8.0)
income = st.number_input('Income', min_value=0, max_value=1000000, value=50000)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
triglycerides = st.number_input('Triglycerides', min_value=0, max_value=1000, value=150)
physical_activity_days_per_week = st.number_input('Physical Activity Days Per Week', min_value=0, max_value=7, value=3)
sleep_hours_per_day = st.number_input('Sleep Hours Per Day', min_value=0, max_value=24, value=8)
systolic_bp = st.number_input('Systolic BP', min_value=0, max_value=300, value=120)
diastolic_bp = st.number_input('Diastolic BP', min_value=0, max_value=200, value=80)
sex_female = st.selectbox('Sex', ['Female', 'Male'])

# Convert Sex to dummy variables
if sex_female == 'Female':
    sex_female = 1
    sex_male = 0
else:
    sex_female = 0
    sex_male = 1

# Create a numpy array for the input features
input_features = np.array([[age, cholesterol, heart_rate, diabetes, family_history, smoking, obesity,
                            alcohol_consumption, exercise_hours_per_week, diet, previous_heart_problems,
                            medication_use, stress_level, sedentary_hours_per_day, income, bmi,
                            triglycerides, physical_activity_days_per_week, sleep_hours_per_day,
                            systolic_bp, diastolic_bp, sex_female, sex_male]])

# Make prediction
prediction = model.predict(input_features)

# Display prediction
if st.button('Predict'):
    if prediction[0] == 1:
        st.write('The model predicts that the person is at risk of a heart attack.')
    else:
        st.write('The model predicts that the person is not at risk of a heart attack.')
