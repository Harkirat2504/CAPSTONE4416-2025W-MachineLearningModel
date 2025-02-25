from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import os
import requests
import logging
from waitress import serve  # Use waitress for production

# Initialize Flask App
app = Flask(__name__)
CORS(app, origins=["https://capstone2025w.netlify.app"])

# Logging Setup
logging.basicConfig(level=logging.DEBUG)

# OpenWeather API Key - Ensure it's set in the environment
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON = 40.6413, -73.7781  # JFK Airport Coordinates

# Ensure API Key is Available
if not OPENWEATHER_API_KEY:
    logging.error("⚠️ Missing OpenWeather API Key! Set the environment variable.")

# Fetch Temperature Forecast from OpenWeather API
def fetch_temperature_forecast():
    """Fetch 7-day temperature forecast from OpenWeather API"""
    if not OPENWEATHER_API_KEY:
        return {"error": "Missing OpenWeather API Key"}

    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&units=metric&appid={OPENWEATHER_API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return {"error": f"API call failed with status {response.status_code}"}
        data = response.json()

        # Extract daily temperature forecast
        if "list" not in data:
            return {"error": "Invalid response from OpenWeather"}

        daily_temps = {}
        for entry in data["list"]:
            date = entry["dt_txt"].split(" ")[0]
            if date not in daily_temps:
                daily_temps[date] = {
                    "temp": entry["main"]["temp"],
                    "icon": entry["weather"][0]["icon"]
                }

        # Limit to 7-day forecast
        forecast = {f"day_{i+1}": daily_temps[key] for i, key in enumerate(daily_temps.keys()) if i < 7}
        return forecast

    except requests.RequestException as e:
        logging.error(f"API Request Failed: {e}")
        return {"error": str(e)}

# Load & Preprocess Weather Data
def load_and_preprocess_weather(file_path):
    """Loads and preprocesses weather data with missing value handling."""
    if not os.path.exists(file_path):
        logging.error(f"❌ File not found: {file_path}")
        return {"error": "weather3.csv file not found"}

    try:
        weather = pd.read_csv(file_path)
        weather.columns = weather.columns.str.lower()
        weather["date"] = pd.to_datetime(weather["date"])
        weather.set_index("date", inplace=True)

        # Remove non-numeric columns
        weather = weather.drop(columns=[col for col in ["station", "name"] if col in weather.columns], errors="ignore")

        # Fill missing values forward
        weather = weather.ffill()

        # Handle outliers using IQR
        for column in weather.select_dtypes(include=[np.number]).columns:
            Q1, Q3 = weather[column].quantile(0.25), weather[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 3 * IQR, Q3 + 3 * IQR
            weather[column] = weather[column].clip(lower=lower_bound, upper=upper_bound)

        return weather

    except Exception as e:
        logging.error(f"Error processing weather data: {e}")
        return {"error": str(e)}

# Create Targets for Multi-Day Wind Speed Prediction
def create_targets(weather, days=7):
    """Creates shifted target variables for wind speed prediction."""
    for i in range(1, days + 1):
        weather[f'target_day_{i}'] = weather['awnd'].shift(-i)
    return weather

# Run Prediction Pipeline
def run_prediction_pipeline():
    """Runs the full prediction pipeline: data preprocessing, model training, and forecasting."""
    file_path = "weather3.csv"
    weather = load_and_preprocess_weather(file_path)

    if isinstance(weather, dict):  # Handle file not found error
        return weather

    weather = create_targets(weather)
    weather = weather.iloc[14:].fillna(0)

    target_cols = [f'target_day_{i}' for i in range(1, 8)]
    predictors = weather.select_dtypes(include=[np.number]).columns
    predictors = [col for col in predictors if col not in target_cols]

    if not predictors:
        logging.error("⚠️ No valid predictors found for model training.")
        return {"error": "No valid predictors found"}

    # Replace inf/-inf and NaNs with 0
    weather[predictors] = weather[predictors].replace([np.inf, -np.inf], np.nan).fillna(0)
    weather[target_cols] = weather[target_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train Ridge Regression Model
    try:
        model = Ridge(alpha=0.1)
        model.fit(weather[predictors], weather[target_cols])
        predictions = model.predict(weather[predictors].iloc[-1:].values)

        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)

        num_days = min(7, predictions.shape[1])

        return {f'day_{i+1}': round(predictions[0][i] * 0.44704, 2) for i in range(num_days)}

    except Exception as e:
        logging.error(f"Model Training Failed: {e}")
        return {"error": "Model failed to generate predictions"}

# API Endpoint to Get Forecasts
@app.route('/predict', methods=['GET'])
def predict():
    """API endpoint to get wind speed predictions & temperature forecast."""
    wind_speed_predictions = run_prediction_pipeline()
    temperature_forecast = fetch_temperature_forecast()

    return jsonify({
        "wind_speed": wind_speed_predictions,
        "temperature": temperature_forecast
    })

# Run Flask App using Waitress (Production Mode)
if __name__ == '__main__':
    logging.info("✅ Starting Flask API Server...")
    serve(app, host="0.0.0.0", port=5000)
