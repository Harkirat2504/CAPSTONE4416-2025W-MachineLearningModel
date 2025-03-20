import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)
API_KEY = "b783407e6178f465fa400808887c3e7f"
REGION = "Toronto"

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

def train_models():
    energy_data = load_energy_data()
    if energy_data is None:
        raise Exception("Energy data could not be loaded.")
    
    X_train = energy_data[['Month', 'Day', 'Hour']].copy()
    temp_map = {1: -5, 2: -3, 3: 0, 4: 5, 5: 10, 6: 15, 7: 20, 8: 18, 9: 12, 10: 6, 11: 0, 12: -4}
    np.random.seed(42)
    X_train['Temperature'] = X_train['Month'].map(temp_map) + np.random.normal(0, 2, size=len(X_train))
    X_train['Daylight'] = X_train['Hour'].apply(lambda h: 1 if 7 <= h <= 19 else 0)

    if "Ontario Demand" not in energy_data.columns or REGION not in energy_data.columns:
        raise Exception("Required energy demand columns not found")

    y_train_ontario = energy_data['Ontario Demand']
    y_train_toronto = energy_data[REGION]

    model_ontario = RandomForestRegressor(n_estimators=100, random_state=42)
    model_ontario.fit(X_train, y_train_ontario)

    model_toronto = RandomForestRegressor(n_estimators=100, random_state=42)
    model_toronto.fit(X_train, y_train_toronto)

    return model_ontario, model_toronto

# Train the models
model_ontario, model_toronto = train_models()
logging.info("Models trained successfully.")

# Save the models to disk
joblib.dump(model_ontario, "model_ontario.pkl")
joblib.dump(model_toronto, "model_toronto.pkl")
logging.info("Models saved to disk.")
