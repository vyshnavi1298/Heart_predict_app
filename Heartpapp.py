import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
model_path = 'naive_bayes_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Sample training columns
training_columns = [
    'Age', 'BMI', 'Cholesterol', 'Systolic BP', 'Diastolic BP', 'Physical Activity',
    'Stress', 'Alcohol Consumption', 'Diabetes', 'Family History', 'Smoking', 
    'Diet', 'Sex_Female', 'Sex_Male'
]

# Preprocessing function
def preprocess_input(data):
    data = data.copy()
    data['Diet'] = data['Diet'].map({'Healthy': 1, 'Average': 2, 'Unhealthy': 3})
    data = pd.get_dummies(data, columns=['Sex'])
    data = data.reindex(columns=training_columns, fill_value=0)
    return data

# Streamlit app
def main():
    st.title('Heart Attack Risk Prediction')

    age = st.slider('Age', 18, 100, 25)
    bmi = st.slider('BMI', 10, 50, 22)
    cholesterol = st.slider('Cholesterol', 100, 300, 150)
    systolic_bp = st.slider('Systolic BP', 90, 200, 120)
    diastolic_bp = st.slider('Diastolic BP', 60, 120, 80)
    physical_activity = st.selectbox('Physical Activity', ['Low', 'Moderate', 'High'])
    stress = st.selectbox('Stress', ['Low', 'Moderate', 'High'])
    alcohol_consumption = st.selectbox('Alcohol Consumption', ['None', 'Occasional', 'Regular'])
    diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
    family_history = st.selectbox('Family History of Heart Disease', ['No', 'Yes'])
    smoking = st.selectbox('Smoking', ['No', 'Yes'])
    diet = st.selectbox('Diet', ['Healthy', 'Average', 'Unhealthy'])
    sex = st.selectbox('Sex', ['Male', 'Female'])

    input_data = {
        'Age': age,
        'BMI': bmi,
        'Cholesterol': cholesterol,
        'Systolic BP': systolic_bp,
        'Diastolic BP': diastolic_bp,
        'Physical Activity': physical_activity,
        'Stress': stress,
        'Alcohol Consumption': alcohol_consumption,
        'Diabetes': diabetes,
        'Family History': family_history,
        'Smoking': smoking,
        'Diet': diet,
        'Sex': sex
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

if __name__ == '__main__':
    main()
