from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import os
import requests  # To fetch temperature data

app = Flask(__name__)
CORS(app, origins=["https://capstone2025w.netlify.app"])

# OpenWeather API Key (Store in an Environment Variable)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # Set this in your hosting environment
LAT = 40.6413  # JFK Airport Latitude
LON = -73.7781  # JFK Airport Longitude

def fetch_temperature_forecast():
    """Fetch 7-day temperature forecast from OpenWeather API"""
    if not OPENWEATHER_API_KEY:
        return {"error": "Missing API key"}
    
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&units=metric&appid={OPENWEATHER_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()

        if "list" not in data:
            return {"error": "Invalid response from OpenWeather"}

        daily_temps = {}
        for entry in data["list"]:
            date = entry["dt_txt"].split(" ")[0]  # Extract date
            if date not in daily_temps:
                daily_temps[date] = {
                    "temp": entry["main"]["temp"],
                    "icon": entry["weather"][0]["icon"]
                }

        # Extract only 7 days of data
        forecast = {f"day_{i+1}": daily_temps[key] for i, key in enumerate(daily_temps.keys()) if i < 7}
        return forecast

    except Exception as e:
        return {"error": str(e)}

@app.route('/predict', methods=['GET'])
def predict():
    """API endpoint to get 7-day wind speed predictions and temperature forecast."""
    wind_speed_predictions = run_prediction_pipeline()
    temperature_forecast = fetch_temperature_forecast()

    return jsonify({
        "wind_speed": wind_speed_predictions,
        "temperature": temperature_forecast
    })

def load_and_preprocess_weather(file_path):
    """Loads and preprocesses weather data, handling missing values and outliers."""
    if not os.path.exists(file_path):
        return {"error": "weather3.csv file not found"}

    weather = pd.read_csv(file_path)
    weather.columns = weather.columns.str.lower()
    weather["date"] = pd.to_datetime(weather["date"])
    weather.set_index("date", inplace=True)

    non_numeric_cols = ["station", "name"]
    weather = weather.drop(columns=[col for col in non_numeric_cols if col in weather.columns], errors="ignore")

    weather = weather.ffill()

    for column in weather.select_dtypes(include=[np.number]).columns:
        Q1 = weather[column].quantile(0.25)
        Q3 = weather[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        weather[column] = weather[column].clip(lower=lower_bound, upper=upper_bound)

    return weather

def create_targets(weather, days=7):
    """Creates target variables for multi-day wind speed prediction."""
    for i in range(1, days + 1):
        weather[f'target_day_{i}'] = weather['awnd'].shift(-i)
    return weather

def run_prediction_pipeline():
    """Runs the full prediction pipeline including preprocessing, feature engineering, and model training."""
    file_path = "weather3.csv"
    weather = load_and_preprocess_weather(file_path)

    if isinstance(weather, dict):  # File not found case
        return weather

    weather = create_targets(weather)
    weather = weather.iloc[14:].fillna(0)

    target_cols = [f'target_day_{i}' for i in range(1, 8)]
    predictors = weather.select_dtypes(include=[np.number]).columns
    predictors = [col for col in predictors if col not in target_cols]

    if not predictors:
        return {"error": "No valid predictors found"}

    weather[predictors] = weather[predictors].replace([np.inf, -np.inf], np.nan).fillna(0)
    weather[target_cols] = weather[target_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    model = Ridge(alpha=0.1)
    model.fit(weather[predictors], weather[target_cols])

    predictions = model.predict(weather[predictors].iloc[-1:].values)

    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)

    num_days = min(7, predictions.shape[1])

    return {f'day_{i+1}': round(predictions[0][i] * 0.44704, 2) for i in range(num_days)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
