import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

# Load the trained model
model_filename = 'naive_bayes_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Preprocessing function for prediction
def preprocess_input(data):
    data[['Systolic BP', 'Diastolic BP']] = data['Blood Pressure'].str.split('/', expand=True)
    data = data.drop(columns='Blood Pressure')
    data['Diet'] = data['Diet'].map({'Healthy': 1, 'Average': 2, 'Unhealthy': 3})
    data = pd.get_dummies(data, columns=['Sex'])
    data[['Sex_Female', 'Sex_Male', 'Systolic BP', 'Diastolic BP']] = data[['Sex_Female', 'Sex_Male', 'Systolic BP', 'Diastolic BP']].astype(int)
    for col in x_train.columns:
        if col not in data.columns:
            data[col] = 0
    return data[x_train.columns]

# Streamlit App
st.set_page_config(page_title='Heart Attack Risk Predictor', page_icon='❤️', layout='centered', initial_sidebar_state='auto')

# Background Style
page_bg = '''
<style>
body {
    background-color: #f0f2f6;
    color: #333;
}
</style>
'''
st.markdown(page_bg, unsafe_allow_html=True)

# Title and Description
st.title('Heart Attack Risk Predictor')
st.markdown("""
<div style="text-align: center; font-size: 20px;">
    Predict your risk of heart attack with this simple app. Fill in the details below and click 'Predict' to see the result.
</div>
""", unsafe_allow_html=True)

# Input Form
st.subheader('Input Your Health Details')
age = st.number_input('Age', min_value=1, max_value=120, value=45, step=1)
sex = st.selectbox('Sex', ['Male', 'Female'])
blood_pressure = st.text_input('Blood Pressure (Systolic/Diastolic)', value='120/80')
cholesterol = st.number_input('Cholesterol', min_value=100, max_value=400, value=200, step=1)
smoking = st.selectbox('Smoking', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
physical_activity = st.selectbox('Physical Activity', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
diet = st.selectbox('Diet', ['Healthy', 'Average', 'Unhealthy'])
stress = st.selectbox('Stress', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Prediction
if st.button('Predict'):
    input_data = {
        'Age': age,
        'Sex': sex,
        'Blood Pressure': blood_pressure,
        'Cholesterol': cholesterol,
        'Smoking': smoking,
        'Physical Activity': physical_activity,
        'Diet': diet,
        'Stress': stress
    }
    input_df = pd.DataFrame([input_data])
    preprocessed_data = preprocess_input(input_df)
    prediction = model.predict(preprocessed_data)
    st.write(f'## Predicted Heart Attack Risk: {"High" if prediction[0] == 1 else "Low"}')

# Footer
st.markdown("""
<div style="text-align: center; font-size: 14px; margin-top: 50px;">
    Developed by [Your Name]. Powered by Streamlit.
</div>
""", unsafe_allow_html=True)
