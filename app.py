from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib
from waitress import serve

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # For testing, allow all origins

logging.basicConfig(level=logging.DEBUG)

API_KEY = "b783407e6178f465fa400808887c3e7f"
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

def get_weather_data(month, start_day, api_key, mode):
    current_temp = 20.0
    sunrise = 6
    sunset = 20
    temp_adjust = {1: -10, 2: -8, 3: -4, 4: 2, 5: 8, 6: 14, 7: 20, 8: 16, 9: 10, 10: 4, 11: -2, 12: -8}
    avg_temp = current_temp + temp_adjust.get(month, 0)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    hourly_data = []

    if mode == 'hourly':
        # Return 1 day of hourly data
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
        # Default daily => 7 days
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
        if model_multi is None:
            return jsonify({"error": "Model not available"})

        # Build weather data
        weather_df = get_weather_data(month, start_day, API_KEY, mode)
        X_pred = weather_df[['Month', 'Day', 'Hour', 'Temperature', 'Daylight']]

        # Predict -> first column is OntarioDemand, then zones
        predictions = model_multi.predict(X_pred)

        results = {}
        if mode == 'hourly':
            # 24 hours
            pred_df = weather_df[['Hour']].copy()
            pred_df["OntarioDemand"] = predictions[:, 0]
            for i, col in enumerate(TARGET_COLUMNS):
                pred_df[col] = predictions[:, i+1]

            # Build results keyed by hour
            for row in pred_df.itertuples():
                hour_key = str(row.Hour)
                hour_forecast = {"Ontario Demand": round(row.OntarioDemand, 2)}
                for col in TARGET_COLUMNS:
                    hour_forecast[col] = round(getattr(row, col), 2)
                results[hour_key] = hour_forecast

        else:
            # daily => group by day
            pred_df = weather_df[['Day']].copy()
            pred_df["OntarioDemand"] = predictions[:, 0]
            for i, col in enumerate(TARGET_COLUMNS):
                pred_df[col] = predictions[:, i+1]

            agg_df = pred_df.groupby('Day').mean().reset_index()

            # Build results keyed by day_i
            for idx, row in agg_df.iterrows():
                day_key = f"day_{idx+1}"
                day_forecast = {}
                # Return it as "Ontario Demand" in JSON
                day_forecast["Ontario Demand"] = round(row["OntarioDemand"], 2)
                for col in TARGET_COLUMNS:
                    day_forecast[col] = round(row[col], 2)
                results[day_key] = day_forecast

        # If user specified a location param, filter out everything else
        location = request.args.get("location", "").strip()
        if location:
            if location not in TARGET_COLUMNS:
                return jsonify({"error": f"Location '{location}' not available. Valid: {TARGET_COLUMNS}."})
            filtered = {}
            for key, forecasts in results.items():
                filtered[key] = {
                    "Ontario Demand": forecasts["Ontario Demand"],
                    location: forecasts[location]
                }
            return jsonify(filtered)

        return jsonify(results)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    logging.info("Starting multi-output Flask API Server...")
    serve(app, host="0.0.0.0", port=5000)
