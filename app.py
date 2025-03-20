from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import logging
import joblib  # For loading the models
from waitress import serve  # For production server

# Initialize Flask App
app = Flask(__name__)
CORS(app, origins=["https://capstone2025w.netlify.app"])

logging.basicConfig(level=logging.DEBUG)

API_KEY = "b783407e6178f465fa400808887c3e7f"
REGION = "Toronto"

# Load pre-trained models at startup
try:
    model_ontario = joblib.load("model_ontario.pkl")
    model_toronto = joblib.load("model_toronto.pkl")
    logging.info("Pre-trained models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    model_ontario, model_toronto = None, None

def load_energy_data(file_path="PUB_DemandZonal_2024_v374.csv"):
    try:
        df = pd.read_csv(file_path, skiprows=3)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        return df
    except Exception as e:
        logging.error(f"Error processing energy data: {e}")
        return None

def get_weather_data(month, start_day, api_key, mode, lat=43.7, lon=-79.42):
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
    try:
        today = datetime.now()
        month = today.month
        start_day = today.day
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
        max_start_day = days_in_month - 6
        if start_day > max_start_day:
            start_day = max_start_day

        logging.info(f"Running prediction from {month}/{start_day} for 7 days in {REGION}")

        if model_ontario is None or model_toronto is None:
            return {"error": "Models are not available."}

        weather_df = get_weather_data(month, start_day, API_KEY, mode='daily')
        X_pred = weather_df[['Month', 'Day', 'Hour', 'Temperature', 'Daylight']]
        y_pred_ontario = model_ontario.predict(X_pred)
        y_pred_toronto = model_toronto.predict(X_pred)

        pred_df = weather_df[['Day']].copy()
        pred_df['Ontario_Demand'] = y_pred_ontario
        pred_df['Toronto_Demand'] = y_pred_toronto
        agg_df = pred_df.groupby('Day').mean().reset_index()

        ontario_demand = {}
        toronto_demand = {}
        for i, row in enumerate(agg_df.itertuples(), 1):
            ontario_demand[f"day_{i}"] = round(row.Ontario_Demand, 2)
            toronto_demand[f"day_{i}"] = round(row.Toronto_Demand, 2)

        return {"ontario_demand": ontario_demand, "toronto_demand": toronto_demand}

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
