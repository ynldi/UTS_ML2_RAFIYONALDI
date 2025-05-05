import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor

# Load the trained model
ann_model = joblib.load('ann_delivery_time_model.pkl')

# Create the Streamlit app
st.title("Food Delivery Time Prediction")

# Input features
distance_km = st.number_input("Distance (km)", min_value=0.0, value=10.0)
weather = st.selectbox("Weather", ["Clear", "Windy", "Foggy", "Rainy", "Snowy"])
traffic_level = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
vehicle_type = st.selectbox("Vehicle Type", ["Scooter", "Bike", "Car"])
preparation_time_min = st.number_input("Preparation Time (min)", min_value=0, value=20)
courier_experience_yrs = st.number_input("Courier Experience (years)", min_value=0, value=3)


# Mapping categorical features to numerical values
weather_mapping = {"Clear": 0, "Windy": 1, "Foggy": 2, "Rainy": 3, "Snowy": 4}
traffic_mapping = {"Low": 0, "Medium": 1, "High": 2}
time_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
vehicle_mapping = {"Scooter": 0, "Bike": 1, "Car": 2}


# Create input DataFrame
new_data = pd.DataFrame({
    'Distance_km': [distance_km],
    'Weather': [weather_mapping[weather]],
    'Traffic_Level': [traffic_mapping[traffic_level]],
    'Time_of_Day': [time_mapping[time_of_day]],
    'Vehicle_Type': [vehicle_mapping[vehicle_type]],
    'Preparation_Time_min': [preparation_time_min],
    'Courier_Experience_yrs': [courier_experience_yrs]
})


# Prediction button
if st.button("Predict Delivery Time"):
    # Make predictions using the loaded model
    simulated_delivery_time = ann_model.predict(new_data)

    # Display the prediction
    st.success(f"Predicted Delivery Time: {simulated_delivery_time[0]:.2f} minutes")
