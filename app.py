from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import logging
from waitress import serve  # For production server

# Initialize Flask App
app = Flask(__name__)
# Adjust the allowed origins as needed
CORS(app, origins=["https://capstone2025w.netlify.app"])

logging.basicConfig(level=logging.DEBUG)

# API Key (kept for compatibility)
API_KEY = "b783407e6178f465fa400808887c3e7f"
# Set the region for forecasting (changed to Toronto)
REGION = "Toronto"

def load_energy_data(file_path="PUB_DemandZonal_2024_v374.csv"):
    """
    Load and preprocess energy demand data.
    Expects a CSV with a 'Date' column and columns named 'Ontario Demand' and the region (e.g., 'Toronto').
    """
    try:
        df = pd.read_csv(file_path, skiprows=3)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing energy data: {e}")
        return None

def get_weather_data(month, start_day, api_key, mode, lat=43.7, lon=-79.42):
    """
    Simulate 7 days of hourly weather data.
    Temperature is adjusted by month and daylight hours.
    """
    current_temp = 20.0
    sunrise = 6
    sunset = 20
    temp_adjust = {1: -10, 2: -8, 3: -4, 4: 2, 5: 8, 6: 14, 7: 18, 8: 16, 9: 10, 10: 4, 11: -2, 12: -8}
    avg_temp = current_temp + temp_adjust.get(month, 0)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    hourly_data = []
    
    for day_offset in range(7):
        day = start_day + day_offset
        if day > days_in_month:
            break
        for hour in range(1, 25):
            temp = avg_temp - 5 + 10 * (sunrise <= hour <= sunset)
            hourly_data.append({
                'Month': month,
                'Day': day,
                'Hour': hour,
                'Temperature': temp,
                'Daylight': 1 if sunrise <= hour <= sunset else 0
            })
    return pd.DataFrame(hourly_data)

def predict_energy_demand():
    """
    Runs the energy demand forecast for the next 7 days using today's date.
    Trains two RandomForest models (one for overall Ontario demand and one for the specified region)
    and aggregates hourly predictions to daily averages.
    """
    try:
        # Use today's date
        today = datetime.now()
        month = today.month
        start_day = today.day

        # Adjust start_day if too close to the end of the month for a full 7-day forecast
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
        max_start_day = days_in_month - 6
        if start_day > max_start_day:
            start_day = max_start_day

        logging.info(f"Running prediction from {month}/{start_day} for 7 days in {REGION}")

        # Load historical energy data
        energy_df = load_energy_data()
        if energy_df is None:
            return {"error": "Energy data file not found or could not be processed"}

        # Generate simulated weather data for the next 7 days
        weather_df = get_weather_data(month, start_day, API_KEY, mode='daily')

        # Prepare training data from historical energy data
        X_train = energy_df[['Month', 'Day', 'Hour']].copy()
        temp_map = {1: -5, 2: -3, 3: 0, 4: 5, 5: 10, 6: 15, 7: 20, 8: 18, 9: 12, 10: 6, 11: 0, 12: -4}
        np.random.seed(42)
        X_train['Temperature'] = X_train['Month'].map(temp_map) + np.random.normal(0, 2, size=len(X_train))
        X_train['Daylight'] = X_train['Hour'].apply(lambda h: 1 if 7 <= h <= 19 else 0)

        # Ensure required columns exist
        if "Ontario Demand" not in energy_df.columns or REGION not in energy_df.columns:
            logging.error("Required energy demand columns not found in energy data.")
            return {"error": "Required energy demand columns not found"}

        y_train_ontario = energy_df['Ontario Demand']
        y_train_region = energy_df[REGION]

        # Train RandomForest models
        model_ontario = RandomForestRegressor(n_estimators=100, random_state=42)
        model_ontario.fit(X_train, y_train_ontario)

        model_region = RandomForestRegressor(n_estimators=100, random_state=42)
        model_region.fit(X_train, y_train_region)

        # Prepare features for prediction from simulated weather data
        X_pred = weather_df[['Month', 'Day', 'Hour', 'Temperature', 'Daylight']]
        y_pred_ontario = model_ontario.predict(X_pred)
        y_pred_region = model_region.predict(X_pred)

        # Build a DataFrame with hourly predictions and group by day
        pred_df = weather_df[['Day']].copy()
        pred_df['Ontario_Demand'] = y_pred_ontario
        pred_df['Toronto_Demand'] = y_pred_region

        # Aggregate hourly predictions to daily averages
        agg_df = pred_df.groupby('Day').mean().reset_index()

        # Create JSON response with keys "ontario_demand" and "toronto_demand"
        ontario_demand = {}
        toronto_demand = {}
        for i, row in enumerate(agg_df.itertuples(), 1):
            ontario_demand[f"day_{i}"] = round(row.Ontario_Demand, 2)
            toronto_demand[f"day_{i}"] = round(row.Toronto_Demand, 2)

        return {
            "ontario_demand": ontario_demand,
            "toronto_demand": toronto_demand
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": str(e)}

@app.route('/predict', methods=['GET'])
def predict():
    result = predict_energy_demand()
    return jsonify(result)

if __name__ == '__main__':
    logging.info("Starting Flask API Server...")
    serve(app, host="0.0.0.0", port=5000)
