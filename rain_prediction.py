import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime

st.title("Rain Prediction by Date (Demo)")

def generate_features_from_date(date_input):
    date_ord = date_input.toordinal()
    np.random.seed(date_ord)
    temperature = np.random.normal(30, 7)
    humidity = np.random.uniform(40, 100)
    wind_speed = np.random.uniform(0, 30)
    pressure = np.random.normal(1013, 10)
    return pd.DataFrame({
        'temperature': [temperature],
        'humidity': [humidity],
        'wind_speed': [wind_speed],
        'pressure': [pressure]
    })

def generate_weather_data(n_samples=300):
    np.random.seed(42)
    temperature = np.random.normal(30, 7, n_samples)
    humidity = np.random.uniform(40, 100, n_samples)
    wind_speed = np.random.uniform(0, 30, n_samples)
    pressure = np.random.normal(1013, 10, n_samples)
    rain = (
        (humidity > 70).astype(int) +
        (pressure < 1010).astype(int) +
        (temperature < 25).astype(int)
    )
    rain = (rain >= 2).astype(int)
    return pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure,
        'rain': rain
    })
@st.cache_resource
def train_model():
    df = generate_weather_data()
    X = df[['temperature', 'humidity', 'wind_speed', 'pressure']]
    y = df['rain']
    model = LogisticRegression()
    model.fit(X, y)
    return model

model = train_model()

st.write("Select a date to predict rain (synthetic demo):")
date_input = st.date_input("Date", value=datetime.today())

if st.button("Predict Rain"):
    features = generate_features_from_date(date_input)
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    st.write("**Generated Weather Features:**")
    st.write(features)
    st.success(f"Rain prediction for {date_input}: {'Yes 🌧️' if prediction==1 else 'No ☀️'} (Probability: {prob:.2f})")