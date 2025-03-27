from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib  # For loading the pre-trained model
from waitress import serve  # For production server

app = Flask(__name__)
CORS(app, origins=["https://capstone2025w.netlify.app"])

logging.basicConfig(level=logging.DEBUG)

API_KEY = "b783407e6178f465fa400808887c3e7f"
# Use all valid zones:
TARGET_COLUMNS = [
    "Northwest", "Northeast", "Ottawa", "East", "Toronto",
    "Essa", "Bruce", "Southwest", "Niagara", "West"
]

try:
    model_multi = joblib.load("multioutput_model.pkl")
    logging.info("Multi-output model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model_multi = None

def get_weather_data(month, start_day, api_key, mode, lat=43.7, lon=-79.42):
    current_temp = 20.0
    sunrise = 6
    sunset = 20
    temp_adjust = {1: -10, 2: -8, 3: -4, 4: 2, 5: 8, 6: 14, 7: 20, 8: 16, 9: 10, 10: 4, 11: -2, 12: -8}
    avg_temp = current_temp + temp_adjust.get(month, 0)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    hourly_data = []
    if mode == 'daily':
        num_days = 7
        for day_offset in range(num_days):
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
    elif mode == 'hourly':
        # Return only one day's hourly data (24 hours)
        day = start_day
        for hour in range(1, 25):
            temp = avg_temp - 5 + 10 * (sunrise <= hour <= sunset)
            hourly_data.append({
                'Month': month,
                'Day': day,
                'Hour': hour,
                'Temperature': temp,
                'Daylight': 1 if sunrise <= hour <= sunset else 0
            })
    else:
        # Default to daily mode if mode is invalid
        num_days = 7
        for day_offset in range(num_days):
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

@app.route('/predict', methods=['GET'])
def predict():
    try:
        today = datetime.now()
        month = int(request.args.get("month", today.month))
        start_day = int(request.args.get("start_day", today.day))
        mode = request.args.get("mode", "daily").strip()
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
        if mode == 'daily':
            # Ensure a 7-day forecast is possible
            max_start_day = days_in_month - 6
            if start_day > max_start_day:
                start_day = max_start_day

        logging.info(f"Running multi-output prediction in {mode} mode from {month}/{start_day}")

        if model_multi is None:
            return jsonify({"error": "Model not available"})

        weather_df = get_weather_data(month, start_day, API_KEY, mode)
        X_pred = weather_df[['Month', 'Day', 'Hour', 'Temperature', 'Daylight']]
        predictions = model_multi.predict(X_pred)  # shape: (num_samples, num_targets)

        results = {}
        if mode == 'daily':
            # Build dataframe with predictions including "Ontario Demand" (first column)
            pred_df = weather_df[['Day']].copy()
            pred_df["Ontario Demand"] = predictions[:, 0]
            for i, col in enumerate(TARGET_COLUMNS):
                pred_df[col] = predictions[:, i+1]
            agg_df = pred_df.groupby('Day').mean().reset_index()
            agg_df.rename(columns=lambda c: c.replace(" ", "_"), inplace=True)

            for i, row in enumerate(agg_df.itertuples(), 1):
                day_forecast = {}
                day_forecast["Ontario Demand"] = round(getattr(row, "Ontario_Demand"), 2)
                for col in TARGET_COLUMNS:
                    safe_col = col.replace(" ", "_")
                    day_forecast[col] = round(getattr(row, safe_col), 2)
                results[f"day_{i}"] = day_forecast
        elif mode == 'hourly':
            # Build dataframe with predictions for hourly mode (one day: 24 hours)
            pred_df = weather_df[['Hour']].copy()
            pred_df["Ontario Demand"] = predictions[:, 0]
            for i, col in enumerate(TARGET_COLUMNS):
                pred_df[col] = predictions[:, i+1]
            for row in pred_df.itertuples():
                hour_key = str(getattr(row, "Hour"))
                results[hour_key] = {"Ontario Demand": round(getattr(row, "Ontario_Demand"), 2)}
                for col in TARGET_COLUMNS:
                    safe_col = col.replace(" ", "_")
                    results[hour_key][col] = round(getattr(row, safe_col), 2)
        else:
            return jsonify({"error": "Invalid mode"})

        # If a location is provided, filter the results to only that zone and Ontario Demand
        location = request.args.get("location", "").strip()
        if location:
            if location not in TARGET_COLUMNS:
                return jsonify({"error": f"Location '{location}' not available. Choose from {TARGET_COLUMNS}."})
            filtered = {}
            for key, forecasts in results.items():
                filtered[key] = {
                    "Ontario Demand": forecasts["Ontario Demand"],
                    location: forecasts[location]
                }
            return jsonify(filtered)
        else:
            return jsonify(results)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    logging.info("Starting multi-output Flask API Server...")
    serve(app, host="0.0.0.0", port=5000)
