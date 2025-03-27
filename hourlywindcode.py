import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import concurrent.futures
import time
from tqdm import tqdm

def load_and_preprocess_hourly_weather(file_path, track_outliers=True):
    """Load and preprocess weather data with optimized operations"""
    print("Loading CSV data...")
    # Read the CSV 
    weather = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')
    
    print("Processing missing values...")
    # Handle missing values - faster column filtering
    null_pct = weather.isna().mean()
    valid_columns = weather.columns[null_pct < 0.05]
    weather = weather[valid_columns]
    
    # Convert column names to lowercase
    weather.columns = weather.columns.str.lower()
    
    # Convert string columns to numeric in one pass
    string_cols = weather.select_dtypes(include=['object']).columns
    for col in tqdm(string_cols, desc="Converting string columns to numeric"):
        weather[col] = pd.to_numeric(weather[col], errors='coerce')
    
    # Forward fill missing values
    weather = weather.ffill()
    
    # Outlier processing - optimized to use vectorized operations
    outliers_info = {}
    
    if track_outliers:
        print("Processing outliers...")
        # Process all numeric columns at once
        numeric_cols = weather.select_dtypes(include=[np.number]).columns
        
        # Calculate quantiles for all columns at once
        Q1 = weather[numeric_cols].quantile(0.25)
        Q3 = weather[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bounds = Q1 - 3 * IQR
        upper_bounds = Q3 + 3 * IQR
        
        # Process each column - still needs loop but with optimized operations
        for column in tqdm(numeric_cols, desc="Processing outliers by column"):
            lower_mask = weather[column] < lower_bounds[column]
            upper_mask = weather[column] > upper_bounds[column]
            
            lower_outlier_count = lower_mask.sum()
            upper_outlier_count = upper_mask.sum()
            
            if lower_outlier_count > 0 or upper_outlier_count > 0:
                # Only store info if there are outliers
                outliers_info[column] = {
                    'bounds': {'lower': lower_bounds[column], 'upper': upper_bounds[column]},
                    'lower_outliers': {} if lower_outlier_count == 0 else 
                        dict(zip(weather.index[lower_mask], weather.loc[lower_mask, column])),
                    'upper_outliers': {} if upper_outlier_count == 0 else 
                        dict(zip(weather.index[upper_mask], weather.loc[upper_mask, column]))
                }
            
            # Clip outliers in one operation
            weather[column] = weather[column].clip(lower=lower_bounds[column], upper=upper_bounds[column])
    
    if track_outliers:
        return weather, outliers_info
    else:
        return weather

def create_hourly_targets(weather, hours=168):
    """Create target columns for hourly prediction"""
    # Optimize by creating all targets at once with a single shift operation
    # Create a new dataframe for targets to avoid modifying original
    targets = pd.DataFrame(index=weather.index)
    for i in range(1, hours + 1):
        targets[f'target_hour_{i}'] = weather['awnd'].shift(-i)
    return pd.concat([weather, targets], axis=1)

def create_hourly_features(weather, rolling_hours=[3, 6, 12, 24]):
    """Create features for hourly prediction - optimized version"""
    print("Creating hourly features...")
    
    # Add cyclical time features in one step
    weather['hour_sin'] = np.sin(2 * np.pi * weather.index.hour / 24)
    weather['hour_cos'] = np.cos(2 * np.pi * weather.index.hour / 24)
    weather['day_of_week_sin'] = np.sin(2 * np.pi * weather.index.dayofweek / 7)
    weather['day_of_week_cos'] = np.cos(2 * np.pi * weather.index.dayofweek / 7)
    weather['month_sin'] = np.sin(2 * np.pi * weather.index.month / 12)
    weather['month_cos'] = np.cos(2 * np.pi * weather.index.month / 12)
    
    # Create pre-computed features dictionary for faster lookups
    feature_columns = {}
    
    # Create rolling window features - optimized to reduce redundant calculations
    print("  Creating rolling window features...")
    key_columns = ["tmax", "tmin", "prcp", "awnd"]
    available_cols = [col for col in key_columns if col in weather.columns]
    
    # Pre-compute rolling means for each column
    rolling_means = {}
    for col in tqdm(available_cols, desc="Computing rolling windows"):
        for horizon in rolling_hours:
            rolling_means[(col, horizon)] = weather[col].rolling(horizon).mean()
    
    # Create features using pre-computed values
    feature_combinations = [(col, horizon) for col in available_cols for horizon in rolling_hours]
    for col, horizon in tqdm(feature_combinations, desc="Creating rolling features"):
        label = f"rolling_{horizon}h_{col}"
        weather[label] = rolling_means[(col, horizon)]
        
        # Calculate percentage difference - optimized to reduce division issues
        shifted = weather[label].shift(1)
        # Replace small values in one step to avoid division issues
        shifted_safe = np.maximum(np.abs(shifted), 0.001) * np.sign(shifted)
        weather[f"{label}_pct"] = (weather[col] - shifted_safe) / shifted_safe
        # Replace infinities with 0 in one step
        weather[f"{label}_pct"] = weather[f"{label}_pct"].replace([np.inf, -np.inf], 0)
    
    # Add lag features in one batch
    print("  Adding lag features...")
    lags = [1, 3, 6, 12, 24]
    for lag in tqdm(lags, desc="Creating lag features"):
        weather[f'awnd_lag_{lag}'] = weather['awnd'].shift(lag)
    
    # Create time-based averages more efficiently
    print("  Creating time-based averages...")
    # Pre-compute all groupby transformations
    hour_avgs = {}
    weekday_avgs = {}
    month_avgs = {}
    
    for col in tqdm(available_cols, desc="Computing time-based averages"):
        hour_avgs[col] = weather[col].groupby(weather.index.hour).transform('mean')
        weekday_avgs[col] = weather[col].groupby(weather.index.dayofweek).transform('mean')
        month_avgs[col] = weather[col].groupby(weather.index.month).transform('mean')
    
    # Assign pre-computed values to dataframe
    for col in available_cols:
        weather[f"hour_avg_{col}"] = hour_avgs[col]
        weather[f"weekday_avg_{col}"] = weekday_avgs[col]
        weather[f"month_avg_{col}"] = month_avgs[col]
    
    return weather

def train_models_parallel(train, test, predictors, target_cols, parallel=True):
    """Train models in parallel for all prediction horizons"""
    predictions = {}
    
    if parallel:
        # Function to train a single model
        def train_and_predict(target):
            # Remove rows with NA targets from training
            train_subset = train.dropna(subset=[target])
            model = Ridge(alpha=0.1)
            model.fit(train_subset[predictors], train_subset[target])
            return target, model.predict(test[predictors])
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(train_and_predict, target) 
                for target in target_cols
            ]
            
            # Track progress with tqdm
            for future in tqdm(concurrent.futures.as_completed(futures), 
                               total=len(target_cols), 
                               desc="Training models in parallel"):
                target, preds = future.result()
                predictions[target] = preds
    else:
        # Sequential version with tqdm
        model = Ridge(alpha=0.1)
        for target in tqdm(target_cols, desc="Training models sequentially"):
            train_subset = train.dropna(subset=[target])
            model.fit(train_subset[predictors], train_subset[target])
            predictions[target] = model.predict(test[predictors])
    
    return predictions

def backtest_hourly_predictions(weather, predictors, hours=168, start_idx=2000, step=168, parallel=True):
    """Optimized backtesting with parallel model training"""
    all_predictions = []
    target_cols = [f'target_hour_{i}' for i in range(1, hours + 1)]
    
    # Calculate total iterations for tqdm
    total_iterations = len(range(start_idx, weather.shape[0], step))
    
    # Create progress bar for backtesting windows
    for i in tqdm(range(start_idx, weather.shape[0], step), 
                 desc="Backtesting windows", 
                 total=total_iterations):
        train = weather.iloc[:i, :]
        test = weather.iloc[i:(i+step), :]
        
        # Train models and get predictions using parallel processing
        predictions = train_models_parallel(
            train, test, predictors, target_cols, parallel=parallel
        )
        
        # Combine predictions with actuals
        preds_df = pd.DataFrame(predictions, index=test.index)
        
        # Get actuals for current test window
        actuals = test[target_cols].copy()
        
        # Rename columns
        preds_df.columns = [f'pred_hour_{i}' for i in range(1, hours + 1)]
        actuals.columns = [f'actual_hour_{i}' for i in range(1, hours + 1)]
        
        # Combine actuals and predictions
        combined = pd.concat([actuals, preds_df], axis=1)
        
        # Calculate absolute differences in one step
        for hour in range(1, hours + 1):
            combined[f'diff_hour_{hour}'] = (
                combined[f'pred_hour_{hour}'] - combined[f'actual_hour_{hour}']
            ).abs()
        
        all_predictions.append(combined)
    
    return pd.concat(all_predictions)

def evaluate_hourly_predictions(predictions, hours=168):
    """Evaluate predictions for each forecast horizon - optimized version"""
    metrics = {}
    
    # Process all hours with tqdm progress tracking
    for hour in tqdm(range(1, hours + 1), desc="Evaluating forecast horizons"):
        actuals = predictions[f'actual_hour_{hour}']
        preds = predictions[f'pred_hour_{hour}']
        
        # Remove any NA values
        mask = ~(actuals.isna() | preds.isna())
        
        if mask.sum() > 0:  # Only calculate if we have valid data
            actuals_valid = actuals[mask]
            preds_valid = preds[mask]
            
            metrics[f'hour_{hour}'] = {
                'MAE': mean_absolute_error(actuals_valid, preds_valid),
                'MSE': mean_squared_error(actuals_valid, preds_valid),
                'RMSE': np.sqrt(mean_squared_error(actuals_valid, preds_valid))
            }
        else:
            metrics[f'hour_{hour}'] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan}
    
    return pd.DataFrame(metrics).round(3)

def display_future_hourly_predictions(predictions, convert_to_ms=True):
    """Display only future predictions starting from the last available date"""
    # Get the last row of predictions
    last_predictions = predictions.iloc[-1]
    
    # Get the last date/time
    last_datetime = predictions.index[-1]
    
    # Create future datetimes for a full week (168 hours)
    future_datetimes = [last_datetime + pd.Timedelta(hours=i) for i in range(1, 169)]
    
    # Create DataFrame with future predictions
    predictions_mph = [last_predictions[f'pred_hour_{i}'] for i in range(1, 169)]
    
    # Convert to meters per second if requested
    if convert_to_ms:
        predictions_ms = [speed * 0.44704 for speed in predictions_mph]
        predictions_to_use = predictions_ms
        speed_unit = 'm/s'
    else:
        predictions_to_use = predictions_mph
        speed_unit = 'mph'
    
    future_df = pd.DataFrame({
        'Datetime': future_datetimes,
        f'Predicted_Wind_Speed_{speed_unit}': predictions_to_use
    }).set_index('Datetime')
    
    # Add day number column for easier reading
    days = [(idx - future_df.index[0]).days + 1 for idx in future_df.index]
    future_df['Day'] = days
    
    return future_df.round(2)

def summarize_outliers(outliers_info):
    """Create a summary DataFrame of the outliers"""
    all_outliers = []
    
    for column, info in outliers_info.items():
        # Lower outliers
        for date, value in info['lower_outliers'].items():
            all_outliers.append({
                'date': date,
                'column': column,
                'value': value,
                'bound': info['bounds']['lower'],
                'type': 'lower'
            })
        
        # Upper outliers
        for date, value in info['upper_outliers'].items():
            all_outliers.append({
                'date': date,
                'column': column,
                'value': value,
                'bound': info['bounds']['upper'],
                'type': 'upper'
            })
    
    if all_outliers:
        df = pd.DataFrame(all_outliers)
        # Calculate difference in one operation
        df['difference'] = np.where(
            df['type'] == 'lower',
            df['bound'] - df['value'],
            df['value'] - df['bound']
        )
        return df.sort_values(['column', 'date'])
    else:
        return pd.DataFrame(columns=['date', 'column', 'value', 'bound', 'type', 'difference'])

def run_hourly_prediction_pipeline(file_path, rolling_hours=[3, 6, 12, 24], forecast_horizon=168, 
                                  track_outliers=True, parallel=True):
    """Main function to run the hourly prediction pipeline - optimized version"""
    print("Starting optimized hourly prediction pipeline...")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    if track_outliers:
        weather, outliers_info = load_and_preprocess_hourly_weather(file_path, track_outliers=True)
    else:
        weather = load_and_preprocess_hourly_weather(file_path, track_outliers=False)
        outliers_info = {}
    
    # Create features
    weather = create_hourly_features(weather, rolling_hours)
    
    # Create target columns
    print("Creating target columns...")
    weather = create_hourly_targets(weather, hours=forecast_horizon)
    
    # Remove initial rolling window period and fill NA values
    # We only need to do this once before modeling
    weather = weather.iloc[max(rolling_hours)+24:, :].fillna(0)
    
    # Replace infinities with NaN and then fill with 0 in one operation
    weather = weather.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Get predictors (all columns except targets and metadata)
    target_cols = [f'target_hour_{i}' for i in range(1, forecast_horizon + 1)]
    metadata_cols = ["name", "station", "wdfm", "wsfm", "wt19", "wv01"]
    predictors = [col for col in weather.columns if col not in target_cols + metadata_cols]
    
    # Run backtest with parallel processing option
    print("Running backtests...")
    predictions = backtest_hourly_predictions(
        weather, predictors, hours=forecast_horizon, parallel=parallel
    )
    
    # Evaluate results
    evaluation = evaluate_hourly_predictions(predictions, hours=forecast_horizon)
    
    return predictions, evaluation, outliers_info, weather

def save_forecast_to_csv(future_forecast, file_path="wind_forecast_7days.csv"):
    """
    Save the 7-day forecast to a CSV file
    
    Parameters:
    -----------
    future_forecast : pandas.DataFrame
        DataFrame containing the forecast data
    file_path : str, optional
        Path where the CSV will be saved, defaults to 'wind_forecast_7days.csv'
    """
    try:
        # Reset index to make datetime a column
        forecast_to_save = future_forecast.reset_index()
        
        # Save to CSV
        forecast_to_save.to_csv(file_path, index=False)
        print(f"\nForecast successfully saved to {file_path}")
        
        # Display info about the saved file
        print(f"File contains {len(forecast_to_save)} hourly predictions across {forecast_to_save['Day'].nunique()} days")
        print(f"Date range: {forecast_to_save['Datetime'].min()} to {forecast_to_save['Datetime'].max()}")
        
    except Exception as e:
        print(f"Error saving forecast to CSV: {e}")
def run_and_analyze():
    print("\n=== OPTIMIZED HOURLY WEATHER PREDICTION PIPELINE ===\n")
    
    # Record start time
    start_time = time.time()
    
    # Run the pipeline with parallel processing enabled
    predictions, evaluation, outliers_info, weather = run_hourly_prediction_pipeline(
        "synthetic_weather_hourly_2015_2025.csv", 
        forecast_horizon=168,  # Predict 168 hours (7 days) ahead
        track_outliers=True,
        parallel=True  # Enable parallel processing
    )
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Create a summary DataFrame of the outliers
    outliers_df = summarize_outliers(outliers_info)
    
    # Display the outliers
    print("\nOutliers found and clipped:")
    if outliers_df.empty:
        print("No outliers were detected in the dataset.")
    else:
        # Show the total number of outliers by column
        print(f"Total outliers found: {len(outliers_df)}")
        print("\nOutlier count by column:")
        print(outliers_df['column'].value_counts())
        
        # Show the most extreme outliers (those with the largest differences)
        print("\nTop 10 most extreme outliers:")
        print(outliers_df.sort_values('difference', ascending=False).head(10))
    
    # Continue with the rest of the analysis
    print("\nPrediction Evaluation by Hour:")
    print(evaluation)
    
    # Display hourly forecast
    future_forecast = display_future_hourly_predictions(predictions, convert_to_ms=True)
    print("\nWind Speed Predictions for Next 7 Days (168 hours, in meters per second):")
    
    # Display summary by day
    day_groups = future_forecast.groupby('Day')
    print("\nDaily summary of predicted wind speeds:")
    day_summary = day_groups[future_forecast.columns[0]].agg(['mean', 'min', 'max'])
    print(day_summary.round(2))
    
    # Print first few and last few predictions
    print("\nHourly predictions (sample):")
    print("First 12 hours:")
    print(future_forecast.head(12))
    print("\nLast 12 hours:")
    print(future_forecast.tail(12))
    
    # Save forecast to CSV
    save_forecast_to_csv(future_forecast)
    
    return predictions, evaluation, outliers_info, weather, future_forecast

# Run the optimized analysis
if __name__ == "__main__":
    predictions, evaluation, outliers_info, weather, future_forecast = run_and_analyze()