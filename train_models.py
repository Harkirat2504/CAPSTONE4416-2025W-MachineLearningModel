import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)
API_KEY = "b783407e6178f465fa400808887c3e7f"
# Define your target columns (make sure your CSV has these columns)
TARGET_COLUMNS = ["Ontario Demand", "Toronto", "Niagara"]

def load_energy_data(file_path="PUB_DemandZonal_2024_v374.csv"):
    try:
        df = pd.read_csv(file_path, skiprows=3)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        # We assume the CSV also contains an "Hour" column.
        return df
    except Exception as e:
        logging.error(f"Error processing energy data: {e}")
        return None

def train_multioutput_model():
    df = load_energy_data()
    if df is None:
        raise Exception("Energy data could not be loaded")
    # Prepare features (assuming your CSV has an "Hour" column)
    X_train = df[['Month', 'Day', 'Hour']].copy()
    # Map temperature adjustment by month
    temp_map = {1: -5, 2: -3, 3: 0, 4: 5, 5: 10, 6: 15, 7: 20, 8: 18, 9: 12, 10: 6, 11: 0, 12: -4}
    np.random.seed(42)
    X_train['Temperature'] = X_train['Month'].map(temp_map) + np.random.normal(0, 2, size=len(X_train))
    X_train['Daylight'] = X_train['Hour'].apply(lambda h: 1 if 7 <= h <= 19 else 0)
    
    # Check that all required target columns exist
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            raise Exception(f"Required column {col} not found")
    
    # Combine targets into one DataFrame (multi-output)
    y_train = df[TARGET_COLUMNS]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train the multi-output model
model = train_multioutput_model()
logging.info("Multi-output model trained successfully.")

# Save the model to disk
joblib.dump(model, "multioutput_model.pkl")
logging.info("Multi-output model saved to disk.")
