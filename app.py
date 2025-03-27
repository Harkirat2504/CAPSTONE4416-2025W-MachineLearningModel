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

# Load the multioutput model used for /predict
try:
    model_multi = joblib.load("multioutput_model.pkl")
    logging.info("Multi-output model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model_multi = None

# Function to load and preprocess weather data (used in /predict)
def get_weather_data(month, start_day, api_key, mode):
    current_temp = 20.0
    sunrise = 6
    sunset = 20
    temp_adjust = {1: -10, 2: -8, 3: -4, 4: 2, 5: 8, 6: 14, 7: 20, 8: 16, 9: 10, 10: 4, 11: -2, 12: -8}
    avg_temp = current_temp + temp_adjust.get(month, 0)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    hourly_data = []

    if mode == 'hourly':
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

# New: full preprocessing function (used in turbine endpoint) that replicates training features
def load_and_preprocess_hourly_weather(file_path, track_outliers=True):
    """Load and preprocess weather data with optimized operations."""
    print("Loading CSV data...")
    weather = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')
    print("Processing missing values...")
    null_pct = weather.isna().mean()
    valid_columns = weather.columns[null_pct < 0.05]
    weather = weather[valid_columns]
    weather.columns = weather.columns.str.lower()
    string_cols = weather.select_dtypes(include=['object']).columns
    for col in string_cols:
        weather[col] = pd.to_numeric(weather[col], errors='coerce')
    weather = weather.ffill()
    return weather

# New function for generating synthetic wind speed data
def generate_wind_data(start_date, end_date):
    time_points = pd.date_range(start=start_date, end=end_date, freq='H')
    n = len(time_points)
    # Generate wind speeds using a normal distribution and clip values to a realistic range
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
# Existing /predict endpoint remains unchanged
@app.route('/predict', methods=['GET'])
def predict():
    try:
        today = datetime.now()
        month = int(request.args.get("month", today.month))
        start_day = int(request.args.get("start_day", today.day))
        mode = request.args.get("mode", "daily").strip()
        if model_multi is None:
            return jsonify({"error": "Model not available"})

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

# Existing /windspeed endpoint remains unchanged
@app.route('/windspeed', methods=['GET'])
def windspeed():
    try:
        today = datetime.now()
        month = int(request.args.get("month", today.month))
        start_day = int(request.args.get("start_day", today.day))
        # For wind speed, produce a 24-hour forecast using synthetic data
        start_date = datetime(2024, month, start_day)
        end_date = start_date + timedelta(hours=23)
        wind_df = generate_wind_data(start_date, end_date)
        # Convert datetime to string if needed for JSON compatibility
        wind_df['datetime'] = wind_df['datetime'].astype(str)
        return jsonify(wind_df.to_dict(orient='records'))
    except Exception as e:
        logging.error(f"Wind speed error: {e}")
        return jsonify({"error": str(e)})

# New /turbine endpoint
@app.route('/turbine', methods=['GET'])
def turbine():
    try:
        # Load the turbine model (trained with the full feature set)
        turbine_model = joblib.load("turbine_hourly_model.pkl")
        
        # Load full weather data using the same preprocessing as training.
        # Add fillna(0) to ensure no NaNs remain.
        weather_full = load_and_preprocess_hourly_weather("synthetic_weather_hourly_2015_2025.csv", track_outliers=True).fillna(0)
        forecast_horizon = 168  # same as used during training
        target_cols = [f'target_hour_{i}' for i in range(1, forecast_horizon + 1)]
        # Define predictors as all columns not in target columns (as during training)
        predictors = [col for col in weather_full.columns if col not in target_cols]
        
        # For a 24-hour forecast, select the last 24 rows from the full data
        test_data = weather_full.tail(24)[predictors]
        
        # Predict using the turbine model
        predictions = turbine_model.predict(test_data)
        # Assume the first column corresponds to the "awnd" (wind speed) prediction.
        predicted_awnd = predictions[:, 0]
        
        # Wind turbine output function
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
