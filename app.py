from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# Load the multioutput model for predictions
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

# New function for generating synthetic wind speed data
def generate_wind_data(start_date, end_date):
    time_points = pd.date_range(start=start_date, end=end_date, freq='H')
    n = len(time_points)
    wind_speed = np.clip(np.random.normal(7, 2, n), 0, 30)
    def wind_turbine_output(ws):
        if ws < 3 or ws > 25:
            return 0
        elif ws >= 12:
            return 2000
        else:
            return (2000 / (12 - 3)) * (ws - 3)
    turbine_output = [wind_turbine_output(ws) for ws in wind_speed]
    return pd.DataFrame({
        'datetime': time_points,
        'wind_speed_ms': wind_speed,
        'turbine_kwh': turbine_output
    })

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
        predictions = model_multi.predict(X_pred)

        results = {}
        if mode == 'hourly':
            pred_df = weather_df[['Hour']].copy()
            pred_df["OntarioDemand"] = predictions[:, 0]
            for i, col in enumerate(TARGET_COLUMNS):
                pred_df[col] = predictions[:, i+1]
            for row in pred_df.itertuples():
                hour_key = str(row.Hour)
                hour_forecast = {"Ontario Demand": round(row.OntarioDemand, 2)}
                for col in TARGET_COLUMNS:
                    hour_forecast[col] = round(getattr(row, col), 2)
                results[hour_key] = hour_forecast
        else:
            pred_df = weather_df[['Day']].copy()
            pred_df["OntarioDemand"] = predictions[:, 0]
            for i, col in enumerate(TARGET_COLUMNS):
                pred_df[col] = predictions[:, i+1]
            agg_df = pred_df.groupby('Day').mean().reset_index()
            for idx, row in agg_df.iterrows():
                day_key = f"day_{idx+1}"
                day_forecast = {"Ontario Demand": round(row["OntarioDemand"], 2)}
                for col in TARGET_COLUMNS:
                    day_forecast[col] = round(row[col], 2)
                results[day_key] = day_forecast

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

@app.route('/windspeed', methods=['GET'])
def windspeed():
    try:
        today = datetime.now()
        month = int(request.args.get("month", today.month))
        start_day = int(request.args.get("start_day", today.day))
        # Produce a 24-hour forecast using synthetic wind data.
        start_date = datetime(2024, month, start_day)
        end_date = start_date + timedelta(hours=23)
        wind_df = generate_wind_data(start_date, end_date)
        wind_df['datetime'] = wind_df['datetime'].astype(str)
        return jsonify(wind_df.to_dict(orient='records'))
    except Exception as e:
        logging.error(f"Wind speed error: {e}")
        return jsonify({"error": str(e)})

# New /turbine endpoint: outputs turbine predictions based on the hourly turbine model.
@app.route('/turbine', methods=['GET'])
def turbine():
    try:
        # Load the turbine model (trained separately using your wind model trainer)
        turbine_model = joblib.load("turbine_hourly_model.pkl")
        
        today = datetime.now()
        month = int(request.args.get("month", today.month))
        start_day = int(request.args.get("start_day", today.day))
        # We'll use hourly mode to generate 24 hours of weather data.
        weather_df = get_weather_data(month, start_day, API_KEY, mode="hourly")
        X_pred = weather_df[['Month', 'Day', 'Hour', 'Temperature', 'Daylight']]
        
        # Get predictions (the model is trained to predict 'awnd' values)
        predictions = turbine_model.predict(X_pred)
        # Use only the first forecast hour for each row (assumed to be the 1-hour ahead prediction)
        predicted_awnd = predictions[:, 0]
        
        # Define turbine function locally
        def wind_turbine_output(ws):
            if ws < 3 or ws > 25:
                return 0
            elif ws >= 12:
                return 2000
            else:
                return (2000 / (12 - 3)) * (ws - 3)
        
        turbine_outputs = [wind_turbine_output(ws) for ws in predicted_awnd]
        
        # Build a dictionary keyed by hour (1 to 24)
        results = {}
        for i, output in enumerate(turbine_outputs, start=1):
            results[str(i)] = round(output, 2)
        return jsonify(results)
    except Exception as e:
        logging.error(f"Turbine endpoint error: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
