from metrics_analysis.metrics import Metrics
import numpy as np
from prepare_data_inference import PrepareData
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Model configuration
dict_models_img = {
    "model_2": 10,  # minutes
    "model_7": 30,
    "model_1": 60,
    "model_8": 120
}
MODEL_NUM = 1
MODEL_PATH = f"models/model_{MODEL_NUM}"
FP_IMAGES = "/home/marta/Projects/tb/data/images/mch/1159"
num_views = 1
seq_len = 3
prediction_minutes = 10
npz_file = f"{MODEL_PATH}/test_data.npz"

# Specific test days to filter
specific_test_days = [
    "2023-03-02", "2024-12-26", "2023-02-13", "2024-10-25", "2024-11-03", 
    "2024-11-08", "2023-01-27", "2023-01-25", "2023-02-09", "2024-10-30",
    "2024-11-09", "2024-10-19", "2024-11-16"
]
specific_test_days = [pd.to_datetime(d).date() for d in specific_test_days]

def read_results_from_csv(csv_path):
    """Improved CSV reading with validation"""
    df = pd.read_csv(csv_path)
    
    # Validate required columns exist
    required_cols = ["datetime", "expected_geneva", "expected_dole", "predicted_geneva", "predicted_dole"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV missing required columns. Found: {df.columns.tolist()}")
    
    datetime_col = pd.to_datetime(df["datetime"])
    expected = df[["expected_geneva", "expected_dole"]].values
    predicted = df[["predicted_geneva", "predicted_dole"]].values
    
    return datetime_col, expected, predicted

# Initialize and prepare weather data (base timestamps)
prepare_data = PrepareData(
    fp_images=FP_IMAGES,
    fp_weather=npz_file,
    num_views=num_views,
    seq_length=seq_len,
    prediction_minutes=prediction_minutes
)
weather_data = prepare_data.data
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
weather_data = weather_data[weather_data['datetime'].dt.date.isin(specific_test_days)]
# [Previous imports and configuration remain the same until the prediction loading section]

# Load all model predictions and PROPERLY align with base timestamps
for model_name, minutes in dict_models_img.items():
    try:
        csv_path = f"models/{model_name}/comparison_dataframe.csv"
        datetime_col, expected, predicted = read_results_from_csv(csv_path)
        
        # Create prediction DataFrame with PROPER time alignment
        pred_df = pd.DataFrame({
            'datetime': datetime_col - pd.Timedelta(minutes=minutes),  # Shift predictions back
            f'predicted_geneva_{minutes}': predicted[:, 0],
            f'predicted_dole_{minutes}': predicted[:, 1],
            f'expected_geneva_{minutes}': expected[:, 0],
            f'expected_dole_{minutes}': expected[:, 1]
        })
        
        # Filter to only specific test days
        pred_df = pred_df[pred_df['datetime'].dt.date.isin(specific_test_days)]
        
        # Merge with weather data using exact datetime match
        weather_data = pd.merge(
            weather_data,
            pred_df,
            on='datetime',
            how='left',
            suffixes=('', f'_{minutes}min')
        )
        
    except Exception as e:
        print(f"Error processing {model_name}: {str(e)}")
        continue

# Post-processing
weather_data = weather_data.sort_values('datetime')

# Verify alignment by checking a specific time point
sample_time = weather_data['datetime'].iloc[0]
print(f"\nVerification for {sample_time}:")
for minutes in dict_models_img.values():
    print(f"{minutes}min prediction:")
    print(f"Pred Geneva: {weather_data.loc[weather_data['datetime'] == sample_time, f'predicted_geneva_{minutes}'].values}")
    print(f"Actual Geneva (shifted): {weather_data.loc[weather_data['datetime'] == sample_time + pd.Timedelta(minutes=minutes), 'expected_geneva_10'].values if sample_time + pd.Timedelta(minutes=minutes) in weather_data['datetime'].values else 'Not available'}")
    # Example: Filter weather_data by a specific date
    filter_date = pd.to_datetime("2024-11-16").date()
    filtered_data = weather_data[weather_data['datetime'].dt.date == filter_date]

    print(f"\nFiltered data for {filter_date}:")
    print(filtered_data[['datetime'] + [c for c in filtered_data.columns if 'geneva' in c or 'dole' in c]].head())
    # Plotting with PROPER time alignment
plot_day = "2024-11-16"
day_data = weather_data[weather_data['datetime'].dt.date == pd.to_datetime(plot_day).date()]

if len(day_data) == 0:
    print(f"No data available for {plot_day}")
else:
    plt.figure(figsize=(15, 10))
    
    # Plot actual values (using the shortest prediction time as reference)
    plt.plot(day_data['datetime'], day_data['expected_geneva_10'], 
             'ko-', label='Actual Geneva', markersize=8)
    plt.plot(day_data['datetime'], day_data['expected_dole_10'], 
             'ks--', label='Actual Dole', markersize=8)
    
    # Plot predictions with proper time alignment
    colors = ['r', 'g', 'b', 'm']
    for i, minutes in enumerate(dict_models_img.values()):
        geneva_col = f'predicted_geneva_{minutes}'
        dole_col = f'predicted_dole_{minutes}'
        
        if geneva_col in day_data.columns:
            # Plot predictions at their PREDICTION time (base time + horizon)
            pred_times = day_data['datetime'] + pd.Timedelta(minutes=minutes)
            plt.plot(pred_times, day_data[geneva_col], 
                     f'{colors[i]}o--', label=f'{minutes}min Pred Geneva', alpha=0.7)
            plt.plot(pred_times, day_data[dole_col], 
                     f'{colors[i]}s--', label=f'{minutes}min Pred Dole', alpha=0.7)
    
    plt.title(f'Actual vs Predicted Values for {plot_day}\n(Predictions shown at prediction time)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = f"{MODEL_PATH}/time_aligned_predictions_{plot_day}.png"
    plt.savefig(plot_path)
    print(f"Time-aligned plot saved to {plot_path}")
    plt.savefig("test.png")

# Debug inspection
print("\nSample of aligned data:")
print(weather_data[['datetime'] + [c for c in weather_data.columns if 'geneva' in c or 'dole' in c]].head())