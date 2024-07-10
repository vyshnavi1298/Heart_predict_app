import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_filename = 'naive_bayes_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_data(input_data):
    # Standardize the numerical columns
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)
    return input_data

# Streamlit app
st.title("Heart Attack Risk Prediction")

st.write("""
### Enter the details to predict the risk of a heart attack
""")

# Input fields for user data
age = st.number_input('Age', min_value=1, max_value=100, value=25)
sex_female = st.selectbox('Sex (Female)', [0, 1])
sex_male = st.selectbox('Sex (Male)', [0, 1])
smoking = st.selectbox('Smoking (0 = No, 1 = Yes)', [0, 1])
diet = st.selectbox('Diet (1 = Healthy, 2 = Average, 3 = Unhealthy)', [1, 2, 3])
systolic_bp = st.number_input('Systolic Blood Pressure', min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=50, max_value=150, value=80)
exercise = st.selectbox('Exercise (0 = No, 1 = Yes)', [0, 1])
stress = st.selectbox('Stress (0 = No, 1 = Yes)', [0, 1])

# Collect user input into a dataframe
user_input = pd.DataFrame({
    'Age': [age],
    'Sex_Female': [sex_female],
    'Sex_Male': [sex_male],
    'Smoking': [smoking],
    'Diet': [diet],
    'Systolic BP': [systolic_bp],
    'Diastolic BP': [diastolic_bp],
    'Exercise': [exercise],
    'Stress': [stress]
})

# Preprocess the user input
preprocessed_data = preprocess_data(user_input)

# Make prediction
if st.button('Predict'):
    prediction = model.predict(preprocessed_data)
    st.write(f'## Predicted Heart Attack Risk: {prediction[0]}')

# Add a background image (optional)
# You can add CSS for custom styling to make the app look awesome

st.markdown(
    """
    <style>
    .main {
        background-image: url("https://example.com/background.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
