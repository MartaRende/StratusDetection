from typing import List

from matplotlib.lines import Line2D
from metrics_analysis.config import PlotConfig
from metrics_analysis.metrics import Metrics
import numpy as np
from prepare_data_inference import PrepareData
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from PIL import Image
#import seaborn as sns
# # Model configuration
# dict_models = {
#     "model_6": 10,  # minutes
#     "model_7": 30,
#     "model_0": 60,
#     "model_9": 120
# }

# Plots for the report
dict_models= {
    "model_2": 10,  # minutes
    "model_10": 30,
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
for model_name, minutes in dict_models.items():
    try:
        csv_path = f"models/{model_name}/comparison_dataframe.csv"
        datetime_col, expected, predicted = read_results_from_csv(csv_path)
        
        # Shift prediction datetimes back by the prediction horizon
        aligned_base_time = datetime_col - pd.to_timedelta(minutes, unit='m')
        
        # Create prediction DataFrame with aligned base timestamps
        pred_df = pd.DataFrame({
            'datetime': aligned_base_time,
            f'predicted_geneva_{minutes}': predicted[:, 0],
            f'predicted_dole_{minutes}': predicted[:, 1],
            f'expected_geneva_{minutes}': expected[:, 0],
            f'expected_dole_{minutes}': expected[:, 1]
        })
        # import ipdb
        # ipdb.set_trace()
        # Filter to only specific test days
        pred_df = pred_df[pred_df['datetime'].dt.date.isin(specific_test_days)]
        
        # Ensure both datetime columns are timezone-naive
        weather_data['datetime'] = pd.to_datetime(weather_data['datetime']).dt.tz_localize(None)
        pred_df['datetime'] = pd.to_datetime(pred_df['datetime']).dt.tz_localize(None)
        
        # Merge on aligned base timestamps
        weather_data = pd.merge(
            weather_data,
            pred_df,
            on='datetime',
            how='left'
        )
    except Exception as e:
        print(f"Error processing {model_name}: {str(e)}")
        continue

# Post-processing
weather_data = weather_data.sort_values('datetime')
# Print value ranges for Geneva between 07:00 and 16:00 for each specific test day
for day in specific_test_days:
    day_data = weather_data[weather_data['datetime'].dt.date == day]
    time_filtered = day_data.set_index('datetime').between_time('07:00', '16:00')
    print(f"Value ranges for Geneva on {day}: min={time_filtered['gre000z0_gen'].min()}, max={time_filtered['gre000z0_gen'].max()}")

# Verify alignment by checking a specific time point
sample_time = weather_data['datetime'].iloc[0]
print(f"\nVerification for {sample_time}:")
for minutes in dict_models.values():
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
for minutes in dict_models.values():
    pred_col = f'predicted_geneva_{minutes}'
    actual_col = 'gre000z0_gen'
    print(f"{minutes}min -> Predicted: min={day_data[pred_col].min()}, max={day_data[pred_col].max()} | Actual: min={day_data[actual_col].min()}, max={day_data[actual_col].max()}")

print("Value ranges for Dole:")
for minutes in dict_models.values():
    pred_col = f'predicted_dole_{minutes}'
    actual_col = 'gre000z0_dole'
    print(f"{minutes}min -> Predicted: min={day_data[pred_col].min()}, max={day_data[pred_col].max()} | Actual: min={day_data[actual_col].min()}, max={day_data[actual_col].max()}")




def plot_aligned_predictions(weather_data, plot_day="2024-11-03"):
    """Plot prediction timelines with consecutive segments from t using horizon colors."""
    
    # Filtro per il giorno richiesto
    day_data = weather_data[weather_data['datetime'].dt.date == pd.to_datetime(plot_day).date()].copy()
    if len(day_data) == 0:
        print(f"No data for {plot_day}")
        return

    # Converto colonne predizione in numerico
    for m in dict_models.values():
        day_data[f'predicted_geneva_{m}'] = pd.to_numeric(day_data[f'predicted_geneva_{m}'], errors='coerce')
        day_data[f'predicted_dole_{m}'] = pd.to_numeric(day_data[f'predicted_dole_{m}'], errors='coerce')
    
    # Converto osservazioni in numerico
    day_data['gre000z0_gen'] = pd.to_numeric(day_data['gre000z0_gen'], errors='coerce')
    day_data['gre000z0_dole'] = pd.to_numeric(day_data['gre000z0_dole'], errors='coerce')
    
    # Plot setup
    plt.figure(figsize=(18, 10))

    # Plot osservazioni
    plt.plot(day_data['datetime'], day_data['gre000z0_gen'], 'ko-', label='Actual Geneva', markersize=6, linewidth=1.5)
    plt.plot(day_data['datetime'], day_data['gre000z0_dole'], 'ks--', label='Actual Dole', markersize=6, linewidth=1.5)

    # Colori per orizzonti
    horizon_colors = {
        10: '#FF6F61',   # Coral
        30: '#4ECDC4',   # Teal
        60: "#D1457B",   # Blue
        120: '#A37EBD'   # Purple
    }

    for idx, row in day_data.iloc[::3].iterrows():
        base_time = row['datetime']

        # --- Geneva ---
        geneva_series = [(base_time, row['gre000z0_gen'])]
        for m in sorted(dict_models.values()):
            pred_value = row.get(f'predicted_geneva_{m}', None)
            if not pd.isna(pred_value):
                pred_time = base_time + pd.Timedelta(minutes=m)
                geneva_series.append((pred_time, pred_value))

        for i in range(1, len(geneva_series)):
            t1, v1 = geneva_series[i - 1]
            t2, v2 = geneva_series[i]
            horizon = int((t2 - t1).total_seconds() // 60)
            color = horizon_colors.get(horizon, 'gray')
            plt.plot([t1, t2], [v1, v2], color=color, linewidth=2, linestyle='-', alpha=0.8)
            plt.plot(t2, v2, 'o', color=color, markersize=5)

        # --- Dole ---
        dole_series = [(base_time, row['gre000z0_dole'])]
        for m in sorted(dict_models.values()):
            pred_value = row.get(f'predicted_dole_{m}', None)
            if not pd.isna(pred_value):
                pred_time = base_time + pd.Timedelta(minutes=m)
                dole_series.append((pred_time, pred_value))

        for i in range(1, len(dole_series)):
            t1, v1 = dole_series[i - 1]
            t2, v2 = dole_series[i]
            horizon = int((t2 - t1).total_seconds() // 60)
            color = horizon_colors.get(horizon, 'gray')
            plt.plot([t1, t2], [v1, v2], color=color, linewidth=2, linestyle='--', alpha=0.8)
            plt.plot(t2, v2, 's', color=color, markersize=5)

    # Formatting
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Legenda
    legend_elements = [
        Line2D([0], [0], color='k', marker='o', linestyle='-', label='Actual Delta'),
    ]
    for h, c in horizon_colors.items():
        legend_elements.append(Line2D([0], [0], color=c, linestyle='-', marker='o', label=f'Delta +{h}min'))


    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'Sequenced Predictions from t → t+h ({plot_day})', pad=20)
    plt.xlabel('Time')
    plt.ylabel('Radiation (W/m²)')
    plt.tight_layout()

    # Salvataggio
    plot_path = f"{MODEL_PATH}/aligned_predictions_{plot_day}.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Plot salvato in: {plot_path}")
    plt.show()


dict_models = {10: 10, 30: 30, 60: 60, 120: 120}
MODEL_PATH = "."  # cambia se vuoi salvare altrove
def interpolate_internal_nans(series):
    """
    Interpolates NaN values that are not at the start or end of the series.
    Leaves leading and trailing NaNs untouched.
    """
    s = series.copy()
    # Only interpolate if there are at least two non-NaN values
    if s.notna().sum() >= 2:
        # Find first and last valid index
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()
        # Only interpolate between first and last valid
        s.loc[first_valid:last_valid] = s.loc[first_valid:last_valid].interpolate()
    return s
FP_IMAGES = "/home/marta/Projects/tb/data/images/mch/1159"  # Default image path

def get_image_for_datetime(dt, view=2):
    """Get image for specific datetime"""
    date_str = dt.strftime('%Y-%m-%d')
    time_str = dt.strftime('%H%M')
    img_filename = f"1159_{view}_{date_str}_{time_str}.jpeg"
    img_path = os.path.join(
        FP_IMAGES, str(view),
        dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'),
        img_filename
    )

    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        return img_array
    else:
        return []
plot_config = PlotConfig()
def plot_horizon_curves_delta(weather_data, plot_day="2024-11-03"):
    """Plot curves of actual and predicted delta values, showing 'x' markers only on original (non-interpolated) values."""
    
    day_data = weather_data[weather_data['datetime'].dt.date == pd.to_datetime(plot_day).date()].copy()
    if day_data.empty:
        return

    # Normalize and track original (non-NaN) values
    for col in ["gre000z0_gen", "gre000z0_dole"]:
        day_data[col] = pd.to_numeric(day_data[col], errors='coerce')

    original_non_nan = {}  # Track original valid values before interpolation
    for horizon in dict_models.values():
        for loc in ['geneva', 'dole']:
            col = f'predicted_{loc}_{horizon}'
            day_data[col] = pd.to_numeric(day_data[col], errors='coerce')
            original_non_nan[col] = day_data[col].notna()
            day_data[col] = interpolate_internal_nans(day_data[col])

    # Setup figure with extra space for image timeline
    fig = plt.figure(figsize=(plot_config.figsize[0], plot_config.figsize[1] * 1.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])

    # Plot actual delta
    delta_actual = day_data["gre000z0_dole"] - day_data["gre000z0_gen"]
    ax1.plot(day_data["datetime"], delta_actual, label="Actual Delta (Dole - Geneva)", color="black", linewidth=2)
    ax1.plot(day_data["datetime"], delta_actual, marker='x', linestyle='None', color="black", markersize=7, alpha=0.8)

    # Plot predicted deltas at each horizon
    horizon_colors = {10: '#FF6F61', 30: "#64E40F", 60: '#45B7D1', 120: '#A37EBD'}
    pred_handles = []

    for horizon in dict_models.values():
        x_vals = day_data["datetime"] + pd.Timedelta(minutes=horizon)
        y_vals = (
            day_data[f"predicted_dole_{horizon}"] -
            day_data[f"predicted_geneva_{horizon}"]
        )

        # Main line
        line, = ax1.plot(x_vals, y_vals, label=f"Predicted Delta +{horizon}min",
                         color=horizon_colors[horizon], linestyle="-")
        pred_handles.append(line)

        # Markers only on original (non-interpolated) points
        mask = (
            original_non_nan[f"predicted_dole_{horizon}"] &
            original_non_nan[f"predicted_geneva_{horizon}"]
        )
        ax1.plot(x_vals[mask], y_vals[mask], marker='x', linestyle='None',
                 color=horizon_colors[horizon], markersize=7, alpha=0.8)

    # Axis formatting
    ax1.set_title(f"Actual vs Predicted Delta Radiation (Dole - Geneva) – {plot_day}",
                  fontsize=plot_config.fontsize["title"])
    ax1.set_xlabel("Time", fontsize=plot_config.fontsize["labels"])
    ax1.set_ylabel("Radiation (W/m²)", fontsize=plot_config.fontsize["labels"])
    ax1.set_xlim(day_data["datetime"].min(), day_data["datetime"].max())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax1.grid(True, alpha=0.3)
    for label in ax1.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color="black", linestyle="-", marker='x',
               label="Actual Delta (Dole - Geneva)")
    ]
    for horizon in dict_models.values():
        legend_elements.append(
            Line2D([0], [0], color=horizon_colors[horizon], linestyle="-", marker='x',
                   label=f"Predicted Delta +{horizon}min")
        )
    ax1.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

    # Select timestamps for image thumbnails
    day_datetimes = day_data["datetime"].tolist()
    num_images = min(6, len(day_datetimes))
    indices = np.linspace(0, len(day_datetimes) - 1, num_images, dtype=int) if num_images > 1 else [0]

    # Load and normalize images
    valid_imgs, valid_times = [], []
    for idx in indices:
        dt = day_datetimes[idx]
        img = get_image_for_datetime(dt)
        if isinstance(img, list):
            continue
        if isinstance(img, np.ndarray) and img.size > 0:
            if img.max() - img.min() < 1e-3:
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            valid_imgs.append(img)
            valid_times.append(dt)

    # Draw image timeline
    if valid_imgs:
        num_valid = len(valid_imgs)
        img_width = 1.0 / num_valid
        for i, (img, dt) in enumerate(zip(valid_imgs, valid_times)):
            left = i * img_width
            ax_img = fig.add_axes([left / 1.1, 0.02, img_width, 0.2])
            ax_img.imshow(img)
            ax_img.set_title(dt.strftime("%H:%M"), fontsize=8)
            ax_img.axis('off')

    # Save and close
    plot_filename = f"horizon_curves_{plot_day}_delta_img.png"
    plot_path = os.path.join(MODEL_PATH, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close()

plot_horizon_curves_delta(weather_data, plot_day="2024-10-25")

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.lines import Line2D

# def plot_mae_all_horizons_heatmap(weather_data, horizons=[10,30,60,120], days=None, save_dir="."):
#     df = weather_data.copy()
#     if days:
#         df = df[df['datetime'].dt.date.isin(days)]
#     df_list = []

#     for h in horizons:
#         for loc in ['geneva', 'dole']:
#             pred_col = f'predicted_{loc}_{h}'
#             exp_col = f'expected_{loc}_{h}'
#             if pred_col in df.columns and exp_col in df.columns:
#                 tmp = df[['datetime', pred_col, exp_col]].dropna()
#                 tmp = tmp.assign(
#                     horizon=h,
#                     loc=loc,
#                     mae=(tmp[pred_col] - tmp[exp_col]).abs()
#                 )[['datetime', 'horizon', 'loc', 'mae']]
#                 df_list.append(tmp)

#     if not df_list:
#         print("No data available for the requested horizons.")
#         return

#     long_df = pd.concat(df_list, ignore_index=True)
#     long_df['date'] = long_df['datetime'].dt.date.astype(str)
#     long_df['time'] = long_df['datetime'].dt.strftime('%H:%M')

#     # Calcola media MAE aggregata su tutti gli orizzonti e location
#     heatmap_df = (long_df
#                   .groupby(['date','time'])['mae']
#                   .mean()
#                   .unstack(fill_value=np.nan))

#     heatmap_df = heatmap_df.reindex(
#         sorted(heatmap_df.columns, key=lambda t: int(t[:2])*60 + int(t[3:])),
#         axis=1
#     )

#     # Compute global mean MAE
#     global_mean = np.nanmean(heatmap_df.values)

#     plt.figure(figsize=(20, max(4, len(heatmap_df)*0.5)))
#     ax = plt.gca()
#     sns.heatmap(
#         heatmap_df,
#         cmap='YlGnBu',
#         linewidths=0.5,
#         linecolor='gray',
#         vmin=0,
#         vmax=250,
#         ax=ax
#     )

#     # Add global mean MAE to the legend as a dummy line
#     legend_elements = [
#         Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label=f'Global Mean: {global_mean:.2f}')
#     ]
#     ax.legend(handles=legend_elements, loc='lower left')

#     plt.title(f"Mean Delta MAE across all predictions model with all data")
#     plt.xlabel("Time", fontsize=12)
#     plt.ylabel("Date", fontsize=12)
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
#     plt.tight_layout()

#     out = os.path.join(save_dir, "mae_all_horizons_heatmap.png")
#     plt.savefig(out, dpi=200, bbox_inches='tight')
#     plt.close()
#     print("Heatmap salvata:", out)

# plot_mae_all_horizons_heatmap(weather_data, horizons=[10,30,60,120], days=specific_test_days, save_dir=MODEL_PATH)

# def plot_horizon_curves(weather_data, plot_day="2024-11-03"):
#     """Plot curves of actual values and predicted values at each horizon over the day, interpolating internal NaNs."""

#     day_data = weather_data[weather_data['datetime'].dt.date == pd.to_datetime(plot_day).date()].copy()
#     if day_data.empty:
#         return

#     # Normalize and interpolate data
#     for col in ["gre000z0_gen", "gre000z0_dole"]:
#         day_data[col] = pd.to_numeric(day_data[col], errors='coerce')

#     for horizon in dict_models.values():
#         for loc in ['geneva', 'dole']:
#             col = f'predicted_{loc}_{horizon}'
#             day_data[col] = pd.to_numeric(day_data[col], errors='coerce')
#             day_data[col] = interpolate_internal_nans(day_data[col])

#     # Setup figure with 2 rows (curves + image timeline)
#     fig = plt.figure(figsize=(plot_config.figsize[0], plot_config.figsize[1] * 1.5))
#     gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
#     ax1 = fig.add_subplot(gs[0])

#     # Actual values
#     ax1.plot(day_data["datetime"], day_data["gre000z0_gen"],
#              label="Actual Geneva", color="black", linewidth=2)
#     ax1.plot(day_data["datetime"], day_data["gre000z0_dole"],
#              label="Actual Dole", color="black", linestyle="--", linewidth=2)
#     # Add 'x' markers to actual curves
#     ax1.plot(day_data["datetime"], day_data["gre000z0_gen"],
#              marker='x', linestyle='None', color="black", markersize=7, alpha=0.8)
#     ax1.plot(day_data["datetime"], day_data["gre000z0_dole"],
#              marker='x', linestyle='None', color="black", markersize=7, alpha=0.8)

#     # Predicted values
#     horizon_colors = {10: '#FF6F61', 30: "#64E40F", 60: '#45B7D1', 120: '#A37EBD'}
#     for horizon in dict_models.values():
#         for loc, style in zip(["geneva", "dole"], ["-", "--"]):
#             x_vals = day_data["datetime"] + pd.Timedelta(minutes=horizon)
#             y_vals = day_data[f"predicted_{loc}_{horizon}"]
#             ax1.plot(
#                 x_vals,
#                 y_vals,
#                 label=f"{loc.capitalize()} +{horizon}min",
#                 color=horizon_colors[horizon],
#                 linestyle=style
#             )
#             # Put 'x' marker on each predicted point
#             ax1.plot(
#                 x_vals,
#                 y_vals,
#                 marker='x',
#                 linestyle='None',
#                 color=horizon_colors[horizon],
#                 markersize=7,
#                 alpha=0.8
#             )

#     # Format axis
#     ax1.set_title(f"Actual vs Predicted Radiation – {plot_day}", fontsize=plot_config.fontsize["title"])
#     ax1.set_xlabel("Time", fontsize=plot_config.fontsize["labels"])
#     ax1.set_ylabel("Radiation (W/m²)", fontsize=plot_config.fontsize["labels"])
#     for label in ax1.get_xticklabels():
#         label.set_rotation(45)
#         label.set_horizontalalignment('right')
#     first_valid = day_data["datetime"].min()
#     last_valid = day_data["datetime"].max()
#     ax1.set_xlim(first_valid, last_valid)
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#     ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
#     ax1.grid(True, alpha=0.3)
#     ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

#     # Select times for image thumbnails
#     day_datetimes = day_data["datetime"].tolist()
#     num_images = min(6, len(day_datetimes))
#     indices = np.linspace(0, len(day_datetimes) - 1, num_images, dtype=int) if num_images > 1 else [0]

#     # Load and normalize images
#     valid_imgs, valid_times = [], []
#     for idx in indices:
#         dt = day_datetimes[idx]
#         img = get_image_for_datetime(dt)

#         if isinstance(img, list):
#             continue
#         if isinstance(img, np.ndarray) and img.size > 0:
#             if img.max() - img.min() < 1e-3:
#                 img = (img - img.min()) / (img.max() - img.min() + 1e-6)
#             if img.ndim == 2:
#                 img = np.stack([img] * 3, axis=-1)
#             valid_imgs.append(img)
#             valid_times.append(dt)

#     # Draw image timeline
#     if valid_imgs:
#         num_valid = len(valid_imgs)
#         img_width = 1.0 / num_valid
#         for i, (img, dt) in enumerate(zip(valid_imgs, valid_times)):
#             left = i * img_width
#             ax_img = fig.add_axes([left/1.1, 0.02, img_width, 0.2])  # relative position
#             ax_img.imshow(img)
#             ax_img.set_title(dt.strftime("%H:%M"), fontsize=8)
#             ax_img.axis('off')

#     # Save and close
#     plot_filename = f"horizon_curves_{plot_day}.png"
#     plot_path = os.path.join(MODEL_PATH, plot_filename)
#     plt.tight_layout()
#     plt.savefig(plot_path, dpi=plot_config.dpi, bbox_inches="tight")
#     plt.close()
# plot_horizon_curves(weather_data, plot_day="2024-11-09")