import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)

API_KEY = "b783407e6178f465fa400808887c3e7f"
# Valid zones
TARGET_COLUMNS = [
    "Northwest", "Northeast", "Ottawa", "East", "Toronto",
    "Essa", "Bruce", "Southwest", "Niagara", "West"
]

def load_energy_data(file_path="PUB_DemandZonal_2024_v374.csv"):
    """
    Load and preprocess energy demand data.
    Make sure your CSV has a column named exactly 'Ontario Demand'.
    We rename it to 'OntarioDemand' below to avoid spacing issues.
    """
    try:
        df = pd.read_csv(file_path)
        print("Columns in CSV:", df.columns.tolist())
        # Rename 'Ontario Demand' -> 'OntarioDemand'
        df.rename(columns={"Ontario Demand": "OntarioDemand"}, inplace=True)

        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        return df
    except Exception as e:
        logging.error(f"Error processing energy data: {e}")
        return None

def train_multioutput_model():
    df = load_energy_data()
    if df is None:
        raise Exception("Energy data could not be loaded.")

    # Prepare features
    X_train = df[['Month', 'Day', 'Hour']].copy()

    # Simple temperature feature
    temp_map = {1: -5, 2: -3, 3: 0, 4: 5, 5: 10, 6: 15, 7: 20, 8: 18, 9: 12, 10: 6, 11: 0, 12: -4}
    np.random.seed(42)
    X_train['Temperature'] = X_train['Month'].map(temp_map) + np.random.normal(0, 2, size=len(X_train))

    # Daylight indicator
    X_train['Daylight'] = X_train['Hour'].apply(lambda h: 1 if 7 <= h <= 19 else 0)

    # Verify columns
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            raise Exception(f"Required column '{col}' not found in CSV.")

    # Our target includes OntarioDemand + the valid zones
    y_train = df[["OntarioDemand"] + TARGET_COLUMNS]

    # Train a multi-output RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == '__main__':
    model = train_multioutput_model()
    logging.info("Multi-output model trained successfully.")
    joblib.dump(model, "multioutput_model.pkl")
    logging.info("Multi-output model saved to disk.")
