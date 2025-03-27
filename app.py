from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
from waitress import serve
import pickle
import tensorflow as tf

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.DEBUG)

API_KEY = "b783407e6178f465fa400808887c3e7f"
TARGET_COLUMNS = [
    "Northwest", "Northeast", "Ottawa", "East", "Toronto",
    "Essa", "Bruce", "Southwest", "Niagara", "West"
]

# Load Random Forest Model
try:
    rf_model = joblib.load("multioutput_model.pkl")
    logging.info("Random Forest model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading RF model: {e}")
    rf_model = None

# Load LSTM model and scaler
try:
    lstm_model = tf.keras.models.load_model("lstm_model.h5")
    with open("scaler.pkl", "rb") as f:
        lstm_scaler = pickle.load(f)
    logging.info("LSTM model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading LSTM model: {e}")
    lstm_model = None
    lstm_scaler = None

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

def predict_rf(month, start_day, mode):
    weather_df = get_weather_data(month, start_day, API_KEY, mode)
    X_pred = weather_df[['Month', 'Day', 'Hour', 'Temperature', 'Daylight']]
    predictions = rf_model.predict(X_pred)
    results = {}
    if mode == 'hourly':
        pred_df = weather_df[['Hour']].copy()
        pred_df["OntarioDemand"] = predictions[:, 0]
        for i, col in enumerate(TARGET_COLUMNS):
            pred_df[col] = predictions[:, i+1]
        for row in pred_df.itertuples():
            hour_key = str(row.Hour)
            forecast = {"Ontario Demand": round(row.OntarioDemand, 2)}
            for col in TARGET_COLUMNS:
                forecast[col] = round(getattr(row, col), 2)
            results[hour_key] = forecast
    else:
        pred_df = weather_df[['Day']].copy()
        pred_df["OntarioDemand"] = predictions[:, 0]
        for i, col in enumerate(TARGET_COLUMNS):
            pred_df[col] = predictions[:, i+1]
        agg_df = pred_df.groupby('Day').mean().reset_index()
        for idx, row in agg_df.iterrows():
            day_key = f"day_{idx+1}"
            forecast = {"Ontario Demand": round(row["OntarioDemand"], 2)}
            for col in TARGET_COLUMNS:
                forecast[col] = round(row[col], 2)
            results[day_key] = forecast
    return results

# ------------------ Sigma (LSTM) Functions for API ------------------
def predict_energy_use_lstm(model, df, scaler, lookback, pred_dates, column='energy_use_kwh'):
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    last_sequence = scaler.transform(df[[column]].tail(lookback))
    last_features = df[['hour', 'day_of_week']].tail(lookback).values
    predictions = []
    current_sequence = last_sequence.copy()
    current_features = last_features.copy()
    for dt in pred_dates:
        hour = dt.hour
        dow = dt.weekday()
        X = np.column_stack((current_sequence, current_features))
        X = X.reshape(1, lookback, 3)
        pred = model.predict(X, verbose=0)[0, 0]
        predictions.append(pred)
        current_sequence = np.vstack((current_sequence[1:], pred))
        current_features = np.vstack((current_features[1:], [hour, dow]))
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def get_lstm_predictions(location, month, start_day, lookback=6):
    # Load CSV and prepare datetime as in training
    df = pd.read_csv("PUB_DemandZonal_2024_v374.csv")
    df.rename(columns={"Ontario Demand": "OntarioDemand"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = df['Hour'].astype(int)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + (df['Hour']-1).astype(str) + ':00')
    # Use training data before the prediction start
    start_date = pd.Timestamp(year=2024, month=month, day=start_day)
    train_end = start_date - pd.Timedelta(hours=1)
    train_df = df[df['datetime'] <= train_end][['datetime', location]].rename(columns={location: 'energy_use_kwh'})
    # Set prediction period: 7 days hourly
    end_date = start_date + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
    pred_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    preds = predict_energy_use_lstm(lstm_model, train_df, lstm_scaler, lookback, pred_dates)
    # Aggregate hourly predictions to daily averages
    pred_df = pd.DataFrame({'datetime': pred_dates, 'predicted_energy_use': preds})
    pred_df['day'] = pred_df['datetime'].dt.day
    daily_preds = pred_df.groupby('day').mean().reset_index()
    sigma_daily = {}
    for idx, row in daily_preds.iterrows():
        day_key = f"day_{idx+1}"
        sigma_daily[day_key] = round(row['predicted_energy_use'], 2)
    # Also build hourly predictions (averaged over days if repeated hour)
    sigma_hourly = {}
    for dt, val in zip(pred_dates, preds):
        hour_key = str(dt.hour)
        sigma_hourly.setdefault(hour_key, []).append(val)
    for hour_key in sigma_hourly:
        sigma_hourly[hour_key] = round(np.mean(sigma_hourly[hour_key]), 2)
    return {"sigma_daily": sigma_daily, "sigma_hourly": sigma_hourly}

@app.route('/predict', methods=['GET'])
def predict_endpoint():
    try:
        today = datetime.now()
        month = int(request.args.get("month", today.month))
        start_day = int(request.args.get("start_day", today.day))
        mode = request.args.get("mode", "daily").strip()  # "daily" or "hourly"
        location = request.args.get("location", "").strip()
        if location == "":
            return jsonify({"error": "Location parameter is required."})
        if rf_model is None or lstm_model is None:
            return jsonify({"error": "Model(s) not available."})
        # Get RF predictions
        rf_results = predict_rf(month, start_day, mode)
        # Get sigma (LSTM) predictions (always computed for daily forecast here)
        sigma_results = get_lstm_predictions(location, month, start_day)
        # Combine results into a single JSON
        combined = {"rf": rf_results, "sigma": sigma_results}
        return jsonify(combined)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    logging.info("Starting combined Flask API Server...")
    serve(app, host="0.0.0.0", port=5000)
