from flask import Flask, request, jsonify
from flask_cors import CORS  # 🔹 Enable CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import os

app = Flask(__name__)
CORS(app, origins=["https://capstone2025w.netlify.app"])  # 🔹 Allow CORS for Netlify

def load_and_preprocess_weather(file_path):
    """Loads and preprocesses weather data, handling missing values and outliers."""
    if not os.path.exists(file_path):
        return {"error": "weather3.csv file not found"}

    weather = pd.read_csv(file_path)
    weather.columns = weather.columns.str.lower()
    weather["date"] = pd.to_datetime(weather["date"])
    weather.set_index("date", inplace=True)

    # Drop non-numeric columns
    non_numeric_cols = ["station", "name"]
    weather = weather.drop(columns=[col for col in non_numeric_cols if col in weather.columns], errors="ignore")

    # Forward fill missing values
    weather = weather.ffill()

    # Remove outliers using IQR
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
    weather = weather.iloc[14:].fillna(0)  # Remove initial rolling window period

    # Define target columns
    target_cols = [f'target_day_{i}' for i in range(1, 8)]

    # Select only numeric predictors (removing target and metadata columns)
    predictors = weather.select_dtypes(include=[np.number]).columns
    predictors = [col for col in predictors if col not in target_cols]

    if not predictors:
        return {"error": "No valid predictors found"}

    # Ensure no NaN or infinite values
    weather[predictors] = weather[predictors].replace([np.inf, -np.inf], np.nan).fillna(0)
    weather[target_cols] = weather[target_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train model
    model = Ridge(alpha=0.1)
    model.fit(weather[predictors], weather[target_cols])  # ✅ Multi-output regression

    # Make predictions
    predictions = model.predict(weather[predictors].iloc[-1:].values)

    # Ensure predictions return an array and handle cases where only a single value is returned
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)

    num_days = min(7, predictions.shape[1])  # Prevent indexing errors

    return {f'day_{i+1}': round(predictions[0][i] * 0.44704, 2) for i in range(num_days)}

@app.route('/predict', methods=['GET'])
def predict():
    """API endpoint to get 7-day wind speed predictions."""
    predictions = run_prediction_pipeline()
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
