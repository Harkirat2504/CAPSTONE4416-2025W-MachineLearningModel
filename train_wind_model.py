import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import joblib
from tqdm import tqdm

def load_and_preprocess_hourly_weather(file_path, track_outliers=True):
    """Load and preprocess weather data with optimized operations."""
    print("Loading CSV data...")
    # Read the CSV file; assumes a DATE column with dates.
    weather = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')
    
    print("Processing missing values...")
    # Remove columns with more than 5% missing values.
    null_pct = weather.isna().mean()
    valid_columns = weather.columns[null_pct < 0.05]
    weather = weather[valid_columns]
    
    # Convert column names to lowercase.
    weather.columns = weather.columns.str.lower()
    
    # Convert string columns to numeric.
    string_cols = weather.select_dtypes(include=['object']).columns
    for col in tqdm(string_cols, desc="Converting string columns to numeric"):
        weather[col] = pd.to_numeric(weather[col], errors='coerce')
    
    # Forward-fill missing values.
    weather = weather.ffill()
    
    return weather

def create_hourly_targets(weather, hours=168):
    """Create target columns for hourly prediction using 'awnd' column."""
    targets = pd.DataFrame(index=weather.index)
    for i in range(1, hours + 1):
        targets[f'target_hour_{i}'] = weather['awnd'].shift(-i)
    # Concatenate targets with original data.
    return pd.concat([weather, targets], axis=1)

def train_backtesting_model(file_path, forecast_horizon=168):
    """
    Load and preprocess weather data, create target columns,
    and train a multioutput model (Ridge via MultiOutputRegressor) for hourly prediction.
    Returns the trained model.
    """
    # Load data
    weather = load_and_preprocess_hourly_weather(file_path, track_outliers=True)
    # Create target columns (using 'awnd' as the variable to forecast)
    weather = create_hourly_targets(weather, hours=forecast_horizon)
    
    # Remove initial rows with NA due to shifting (adjust as needed)
    weather = weather.iloc[24:, :].fillna(0)
    weather = weather.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Define target columns and predictors.
    target_cols = [f'target_hour_{i}' for i in range(1, forecast_horizon + 1)]
    metadata_cols = []  # add any columns you wish to exclude from predictors
    predictors = [col for col in weather.columns if col not in target_cols + metadata_cols]
    
    X = weather[predictors]
    y = weather[target_cols]
    
    # Train a multioutput Ridge regression model.
    print("Training multioutput model...")
    model = MultiOutputRegressor(Ridge(alpha=0.1))
    model.fit(X, y)
    print("Training complete.")
    return model

if __name__ == '__main__':
    file_path = "synthetic_weather_hourly_2015_2025.csv"  # Adjust to your CSV file path
    forecast_horizon = 168  # 7 days of hourly predictions
    model = train_backtesting_model(file_path, forecast_horizon)
    
    # Save the trained model to a pickle file.
    joblib.dump(model, "turbine_hourly_model.pkl")
    print("Model trained and saved to multioutput_hourly_model.pkl")
