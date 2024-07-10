import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('best_decision_tree_model.pkl')

# Page configuration
st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Add a custom background
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.pixelstalk.net/wp-content/uploads/2016/06/Free-Download-Heart-Background-Images-HD.jpg");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# Title and description
st.title('❤️ Heart Attack Risk Prediction')
st.write('Please provide the following details to predict your heart attack risk:')

# Define the input form
with st.form(key='heart_attack_form'):
    age = st.number_input('Age', min_value=1, max_value=120, value=25)
    sex = st.selectbox('Sex', options=['Male', 'Female'])
    cholesterol = st.number_input('Cholesterol', min_value=0, value=200)
    blood_pressure = st.text_input('Blood Pressure (e.g., 120/80)')
    heart_rate = st.number_input('Heart Rate', min_value=0, value=70)
    diabetes = st.selectbox('Diabetes', options=[0, 1])
    family_history = st.selectbox('Family History', options=[0, 1])
    smoking = st.selectbox('Smoking', options=[0, 1])
    obesity = st.selectbox('Obesity', options=[0, 1])
    alcohol_consumption = st.number_input('Alcohol Consumption', min_value=0, value=0)
    exercise_hours_per_week = st.number_input('Exercise Hours Per Week', min_value=0.0, value=1.0)
    diet = st.selectbox('Diet', options=['Healthy', 'Unhealthy'])
    previous_heart_problems = st.selectbox('Previous Heart Problems', options=[0, 1])
    medication_use = st.selectbox('Medication Use', options=[0, 1])
    stress_level = st.number_input('Stress Level', min_value=0, max_value=10, value=5)
    sedentary_hours_per_day = st.number_input('Sedentary Hours Per Day', min_value=0.0, value=6.0)
    income = st.number_input('Income', min_value=0, value=50000)
    bmi = st.number_input('BMI', min_value=0.0, value=25.0)
    triglycerides = st.number_input('Triglycerides', min_value=0, value=150)
    physical_activity_days_per_week = st.number_input('Physical Activity Days Per Week', min_value=0, max_value=7, value=3)
    sleep_hours_per_day = st.number_input('Sleep Hours Per Day', min_value=0.0, max_value=24.0, value=7.0)
    
    # Predict button
    submit_button = st.form_submit_button(label='Predict ❤️')

if submit_button:
    # Create a dataframe with the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Cholesterol': [cholesterol],
        'Blood Pressure': [blood_pressure],
        'Heart Rate': [heart_rate],
        'Diabetes': [diabetes],
        'Family History': [family_history],
        'Smoking': [smoking],
        'Obesity': [obesity],
        'Alcohol Consumption': [alcohol_consumption],
        'Exercise Hours Per Week': [exercise_hours_per_week],
        'Diet': [diet],
        'Previous Heart Problems': [previous_heart_problems],
        'Medication Use': [medication_use],
        'Stress Level': [stress_level],
        'Sedentary Hours Per Day': [sedentary_hours_per_day],
        'Income': [income],
        'BMI': [bmi],
        'Triglycerides': [triglycerides],
        'Physical Activity Days Per Week': [physical_activity_days_per_week],
        'Sleep Hours Per Day': [sleep_hours_per_day]
    })

    # Predict the risk
    risk_prediction = model.predict(input_data)[0]

    # Display the result
    st.markdown('### Prediction Result:')
    if risk_prediction == 1:
        st.error('High risk of heart attack.')
    else:
        st.success('Low risk of heart attack.')
