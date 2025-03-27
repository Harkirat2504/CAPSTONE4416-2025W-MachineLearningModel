import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import joblib
from tqdm import tqdm

def load_and_preprocess_hourly_weather(file_path, track_outliers=True):
    """Load and preprocess weather data with optimized operations."""
    print("Loading CSV data...")
    # Read the CSV file (assumes the DATE column contains dates)
    weather = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')
    
    print("Processing missing values...")
    # Remove columns with more than 5% missing values
    null_pct = weather.isna().mean()
    valid_columns = weather.columns[null_pct < 0.05]
    weather = weather[valid_columns]
    
    # Convert column names to lowercase
    weather.columns = weather.columns.str.lower()
    
    # Convert string columns to numeric in one pass
    string_cols = weather.select_dtypes(include=['object']).columns
    for col in tqdm(string_cols, desc="Converting string columns to numeric"):
        weather[col] = pd.to_numeric(weather[col], errors='coerce')
    
    # Forward-fill missing values
    weather = weather.ffill()
    
    # (Optional) Process outliers here if needed.
    return weather

def create_hourly_targets(weather, hours=168):
    """Create target columns for hourly prediction using column 'awnd'."""
    targets = pd.DataFrame(index=weather.index)
    for i in range(1, hours + 1):
        targets[f'target_hour_{i}'] = weather['awnd'].shift(-i)
    return pd.concat([weather, targets], axis=1)

def create_hourly_features(weather, rolling_hours=[3, 6, 12, 24]):
    """Create additional features for hourly prediction."""
    # Example: add cyclical time features (you can extend this as needed)
    weather['hour_sin'] = np.sin(2 * np.pi * weather.index.hour / 24)
    weather['hour_cos'] = np.cos(2 * np.pi * weather.index.hour / 24)
    return weather

def train_multioutput_model(file_path, forecast_horizon=168):
    """
    Load the weather data, create features and targets, and train a multioutput model.
    Returns the trained model.
    """
    # Load and preprocess data
    weather = load_and_preprocess_hourly_weather(file_path, track_outliers=True)
    # Create additional features
    weather = create_hourly_features(weather, rolling_hours=[3, 6, 12, 24])
    # Create target columns (using 'awnd' as the variable to forecast)
    weather = create_hourly_targets(weather, hours=forecast_horizon)
    
    # Remove initial rows that contain NA due to shifting (adjust the number as needed)
    weather = weather.iloc[24:, :].fillna(0)
    
    # Define target columns and predictors
    target_cols = [f'target_hour_{i}' for i in range(1, forecast_horizon + 1)]
    # You can add any metadata columns that you want to exclude from predictors
    metadata_cols = []  
    predictors = [col for col in weather.columns if col not in target_cols + metadata_cols]
    
    # Create training matrices
    X = weather[predictors]
    y = weather[target_cols]
    
    # Train a multioutput Ridge model
    model = MultiOutputRegressor(Ridge(alpha=0.1))
    print("Training multioutput model...")
    model.fit(X, y)
    print("Training complete.")
    return model

if __name__ == '__main__':
    # Adjust the file path as needed; here we use a synthetic weather CSV file.
    file_path = "synthetic_weather_hourly_2015_2025.csv"
    forecast_horizon = 168  # 7 days ahead (hourly predictions)
    
    # Train the model
    model = train_multioutput_model(file_path, forecast_horizon)
    
    # Save the model to a pickle file
    joblib.dump(model, "weather_hourly_model.pkl")
    print("Model trained and saved to multioutput_hourly_model.pkl")
