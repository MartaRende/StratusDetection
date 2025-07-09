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
import ipdb
ipdb.set_trace()
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
import ipdb 
ipdb.set_trace()
# Post-processing
weather_data = weather_data.sort_values('datetime')
# Print value ranges for Geneva between 07:00 and 16:00 for each specific test day
for day in specific_test_days:
    day_data = weather_data[weather_data['datetime'].dt.date == day]
    time_filtered = day_data.set_index('datetime').between_time('07:00', '16:00')
    print(f"Value ranges for Geneva on {day}: min={time_filtered['gre000z0_gen'].min()}, max={time_filtered['gre000z0_gen'].max()}")
import ipdb
ipdb.set_trace()
# Verify alignment by checking a specific time point
sample_time = weather_data['datetime'].iloc[0]
print(f"\nVerification for {sample_time}:")
for minutes in dict_models_img.values():
    print(f"{minutes}min prediction:")
    print(f"Pred Geneva: {weather_data.loc[weather_data['datetime'] == sample_time, f'predicted_geneva_{minutes}'].values}")
    print(f"Actual Geneva (shifted): {weather_data.loc[weather_data['datetime'] == sample_time + pd.Timedelta(minutes=minutes), 'expected_geneva_10'].values if sample_time + pd.Timedelta(minutes=minutes) in weather_data['datetime'].values else 'Not available'}")
    # Example: Filter weather_data by a specific datetime range
    start_time = pd.to_datetime("2024-11-16 06:00")
    end_time = pd.to_datetime("2024-11-16 12:00")
    filtered_data = weather_data[(weather_data['datetime'] >= start_time) & (weather_data['datetime'] <= end_time)]

    print(f"\nFiltered data from {start_time} to {end_time}:")
    print(filtered_data[['datetime'] + [c for c in filtered_data.columns if 'geneva' in c or 'dole' in c]].head())
# Plotting with PROPER time alignment
# Plotting prediction lines radiating from each observation point
plot_day = "2024-11-03"
day_data = weather_data[weather_data['datetime'].dt.date == pd.to_datetime(plot_day).date()]
# Save the aligned weather_data DataFrame to CSV
csv_save_path = f"{MODEL_PATH}/aligned_weather_predictions_{plot_day}.csv"
weather_data.to_csv(csv_save_path, index=False)
print(f"Aligned data saved to {csv_save_path}")
print("Value ranges for Geneva:")
for minutes in dict_models_img.values():
    pred_col = f'predicted_geneva_{minutes}'
    actual_col = 'gre000z0_gen'
    print(f"{minutes}min -> Predicted: min={day_data[pred_col].min()}, max={day_data[pred_col].max()} | Actual: min={day_data[actual_col].min()}, max={day_data[actual_col].max()}")

print("Value ranges for Dole:")
for minutes in dict_models_img.values():
    pred_col = f'predicted_dole_{minutes}'
    actual_col = 'gre000z0_dole'
    print(f"{minutes}min -> Predicted: min={day_data[pred_col].min()}, max={day_data[pred_col].max()} | Actual: min={day_data[actual_col].min()}, max={day_data[actual_col].max()}")

import ipdb
ipdb.set_trace()
def plot_aligned_predictions(weather_data, plot_day="2024-11-03"):
    """Plot properly aligned prediction lines"""
    # Filter and prepare data
    day_data = weather_data[weather_data['datetime'].dt.date == pd.to_datetime(plot_day).date()].copy()
    if len(day_data) == 0:
        print(f"No data for {plot_day}")
        return
    
    # Convert to numeric and handle NaNs
    for m in dict_models_img.values():
        day_data[f'predicted_geneva_{m}'] = pd.to_numeric(day_data[f'predicted_geneva_{m}'], errors='coerce')
        day_data[f'predicted_dole_{m}'] = pd.to_numeric(day_data[f'predicted_dole_{m}'], errors='coerce')
    
    # Create figure
    plt.figure(figsize=(18, 10))
    
    # Plot actual values
    plt.plot(day_data['datetime'], day_data['gre000z0_gen'], 
             'ko-', label='Actual Geneva', markersize=6, linewidth=1.5)
    plt.plot(day_data['datetime'], day_data['gre000z0_dole'], 
             'ks--', label='Actual Dole', markersize=6, linewidth=1.5)
    
    # Consistent colors for each horizon
    horizon_colors = {
        10: '#FF6B6B',  # Red
        30: '#4ECDC4',   # Teal
        60: '#45B7D1',   # Blue
        120: '#A37EBD'   # Purple
    }
    
    # Plot each prediction point with proper alignment
    for idx, row in day_data.iterrows():
        base_time = row['datetime']
        actual_geneva = row['gre000z0_gen']
        actual_dole = row['gre000z0_dole']
        
        for minutes in sorted(dict_models_img.values()):
            pred_time = base_time + pd.Timedelta(minutes=minutes)
            pred_geneva = row[f'predicted_geneva_{minutes}']
            pred_dole = row[f'predicted_dole_{minutes}']
            
            if not pd.isna(pred_geneva):
                # Plot Geneva prediction line
                plt.plot([base_time, pred_time], [actual_geneva, pred_geneva],
                         color=horizon_colors[minutes], linestyle='-',
                         alpha=0.6, linewidth=1)
                
                # Plot Geneva marker
                plt.plot(pred_time, pred_geneva, 'o',
                         color=horizon_colors[minutes], markersize=6)
            
            if not pd.isna(pred_dole):
                # Plot Dole prediction line
                plt.plot([base_time, pred_time], [actual_dole, pred_dole],
                         color=horizon_colors[minutes], linestyle='--',
                         alpha=0.6, linewidth=1)
                
                # Plot Dole marker
                plt.plot(pred_time, pred_dole, 's',
                         color=horizon_colors[minutes], markersize=6)
    
    # Formatting
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', marker='o', linestyle='-', label='Actual Geneva'),
        Line2D([0], [0], color='k', marker='s', linestyle='--', label='Actual Dole')
    ]
    for minutes, color in horizon_colors.items():
        legend_elements.append(
            Line2D([0], [0], color=color, marker='o', linestyle='None',
                  label=f'{minutes}min Pred'))
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'Aligned Predictions for {plot_day}', pad=20)
    plt.xlabel('Time')
    plt.ylabel('Radiation (W/m²)')
    plt.tight_layout()
    
    plot_path = f"{MODEL_PATH}/aligned_predictions_{plot_day}.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Aligned plot saved to {plot_path}")
    plt.show()

# Usage
plot_aligned_predictions(weather_data, "2024-11-03")

# Debug inspection
print("\nSample of aligned data:")
print(weather_data[['datetime'] + [c for c in weather_data.columns if 'geneva' in c or 'dole' in c]].head())