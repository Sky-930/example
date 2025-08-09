import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

# --- Load the model and scaler ---
with open('best_lgbm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# --- Feature names used during training ---
feature_names = [
    'pickup_longitude', 'dropoff_longitude', 'dropoff_latitude',
    'passenger_count', 'year', 'month', 'day', 'hour', 'distance',
    'pickup_season_Spring', 'pickup_season_Summer', 'pickup_season_Winter',
    'pickup_period_Evening', 'pickup_period_Morning', 'pickup_period_Night'
]

# --- Streamlit Layout ---
st.title("🚕 Fare Prediction App")
st.write("Enter trip details below to predict your fare.")

pickup_date = st.date_input("Pickup Date", datetime.today())
pickup_time = st.time_input("Pickup Time")

pickup_lat = st.number_input("Pickup Latitude", value=40.7128, format="%.6f")
pickup_lon = st.number_input("Pickup Longitude", value=-74.0060, format="%.6f")
dropoff_lat = st.number_input("Dropoff Latitude", value=40.7769, format="%.6f")
dropoff_lon = st.number_input("Dropoff Longitude", value=-73.9813, format="%.6f")
passenger_count = st.slider("Number of Passengers", min_value=1, max_value=6, value=1)

# --- Haversine Function ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# --- Helper: get season one-hot ---
def get_season(month):
    return {
        'pickup_season_Spring': int(month in [3, 4, 5]),
        'pickup_season_Summer': int(month in [6, 7, 8]),
        'pickup_season_Winter': int(month in [12, 1, 2]),
    }

# --- Helper: get pickup period one-hot ---
def get_period(hour):
    return {
        'pickup_period_Morning': int(5 <= hour < 12),
        'pickup_period_Evening': int(17 <= hour < 21),
        'pickup_period_Night': int(hour >= 21 or hour < 5),
    }

if st.button("Predict Fare"):
    try:
        # Combine date and time
        pickup_datetime = datetime.combine(pickup_date, pickup_time)

        # Base features
        year = pickup_datetime.year
        month = pickup_datetime.month
        day = pickup_datetime.day
        hour = pickup_datetime.hour
        distance = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

        # One-hot features
        season_features = get_season(month)
        period_features = get_period(hour)

        # Construct input DataFrame
        input_data = pd.DataFrame([{
            'pickup_longitude': pickup_lon,
            'dropoff_longitude': dropoff_lon,
            'dropoff_latitude': dropoff_lat,
            'passenger_count': passenger_count,
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
            'distance': distance,
            **season_features,
            **period_features
        }])

        # Ensure column order matches training
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Scale and predict
        scaled_data = scaler.transform(input_data)
        log_prediction = model.predict(scaled_data)
        fare_prediction = np.expm1(log_prediction)[0]

        st.success(f"💵 Predicted Fare: ${fare_prediction:.2f}")

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
    # pip install streamlit pandas numpy scikit-learn lightgbm

