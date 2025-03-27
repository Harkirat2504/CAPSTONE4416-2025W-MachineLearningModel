import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
import pickle

# For LSTM model training (sigma functionality)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.DEBUG)

API_KEY = "b783407e6178f465fa400808887c3e7f"
# Valid zones
TARGET_COLUMNS = [
    "Northwest", "Northeast", "Ottawa", "East", "Toronto",
    "Essa", "Bruce", "Southwest", "Niagara", "West"
]

def load_energy_data(file_path="PUB_DemandZonal_2024_v374.csv"):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Columns in CSV: {df.columns.tolist()}")
        # Rename 'Ontario Demand' -> 'OntarioDemand' to avoid spacing issues
        df.rename(columns={"Ontario Demand": "OntarioDemand"}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        return df
    except Exception as e:
        logging.error(f"Error processing energy data: {e}")
        return None

# ------------------- Random Forest Model -------------------
def train_multioutput_model():
    df = load_energy_data()
    if df is None:
        raise Exception("Energy data could not be loaded.")
    # Prepare features
    X_train = df[['Month', 'Day', 'Hour']].copy()
    temp_map = {1: -5, 2: -3, 3: 0, 4: 5, 5: 10, 6: 15, 7: 20, 8: 18, 9: 12, 10: 6, 11: 0, 12: -4}
    np.random.seed(42)
    X_train['Temperature'] = X_train['Month'].map(temp_map) + np.random.normal(0, 2, size=len(X_train))
    X_train['Daylight'] = X_train['Hour'].apply(lambda h: 1 if 7 <= h <= 19 else 0)
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            raise Exception(f"Required column '{col}' not found in CSV.")
    y_train = df[["OntarioDemand"] + TARGET_COLUMNS]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# ------------------- Sigma (LSTM) Model Functions -------------------
def prepare_lstm_data(df, lookback=6, column='energy_use_kwh'):
    # Create additional features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    energy_scaled = scaler.fit_transform(df[[column]])
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(np.column_stack((energy_scaled[i-lookback:i], 
                                  df[['hour', 'day_of_week']].iloc[i-lookback:i].values)))
        y.append(energy_scaled[i])
    return np.array(X), np.array(y), scaler

def train_lstm_model(location, month, day, lookback=6):
    # Load CSV and create a datetime column (assumes Hour is 1-indexed)
    df = load_energy_data()
    df['Hour'] = df['Hour'].astype(int)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + (df['Hour']-1).astype(str) + ':00')
    # For LSTM training, use data before prediction start date
    start_date = pd.Timestamp(year=2024, month=month, day=day)
    train_end = start_date - pd.Timedelta(hours=1)
    train_energy_df = df[df['datetime'] <= train_end][['datetime', location]].rename(columns={location: 'energy_use_kwh'})
    X_train, y_train, scaler = prepare_lstm_data(train_energy_df, lookback)
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 3), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    # Save LSTM model and scaler
    model.save("lstm_model.h5")
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return model, scaler

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

if __name__ == '__main__':
    # Train and save Random Forest model
    rf_model = train_multioutput_model()
    logging.info("Multi-output model trained successfully.")
    joblib.dump(rf_model, "multioutput_model.pkl")
    logging.info("Multi-output model saved to disk.")
    
    # Train and save LSTM model for a chosen default location (e.g., Toronto)
    default_location = "Toronto"
    default_month = 1   # Change as desired
    default_day = 1     # Change as desired
    lstm_model, lstm_scaler = train_lstm_model(default_location, default_month, default_day)
    logging.info("LSTM model trained and saved successfully.")
