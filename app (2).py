import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load the trained model and scaler
model = load_model('/content/your_model.keras')
scaler = joblib.load('scaler.pkl')

# Title of the app
st.title('Sepsis Survival Prediction App')

# Subtitle
st.markdown('This app predicts the survival probability of sepsis patients.')

# Input fields
st.sidebar.header('User Input Features')

def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 50)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    sepsis_episodes = st.sidebar.slider('Number of Sepsis Episodes', 1, 10, 1)
    
    gender = 1 if gender == 'Female' else 0  # Convert gender to binary
    
    data = {
        'Age': age,
        'Gender': gender,
        'SepsisEpisodes': sepsis_episodes
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the input features
st.subheader('User Input Features')
st.write(input_df)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Predict survival
prediction_proba = model.predict(input_scaled)[0][0]
prediction = 1 if prediction_proba > 0.7 else 0

# Display the prediction
st.subheader('Prediction')
outcome = 'Survived' if prediction == 1 else 'Not Survived'
st.write(f"The model predicts that the patient will: **{outcome}**")

# Display the prediction probability
st.subheader('Prediction Probability')
st.write(f"Survival Probability: **{prediction_proba:.2f}**")

# Additional information
st.write("""
The prediction is based on a deep learning model trained on historical sepsis survival data.
""")
