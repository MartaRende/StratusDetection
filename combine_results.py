import numpy as np
import matplotlib.pyplot as plt
import os
from metrics_analysis.metrics import Metrics

# === File paths ===
img_only_fp = "models/model_1/predictions_vs_expected_ready.npz"
img_meteo_fp = "models/model_0/predictions_vs_expected_ready.npz"
npz_file = "models/model_0/test_data.npz"
FP_IMAGES = "/home/marta/Projects/tb/data/images/mch/1159"

# === Specific test days to plot ===
specific_test_days = [
    "2023-03-02", "2024-12-26", "2023-02-13", "2024-10-25", "2024-11-03",
    "2024-11-08", "2023-01-27", "2023-01-25", "2023-02-09", "2024-10-30",
    "2024-11-09", "2024-10-19", "2024-11-16"
]

# === Load predictions and expected ===
def load_predictions(file_path):
    with np.load(file_path, allow_pickle=True) as pred_data:
        predicted = pred_data["predicted"].astype(float)
        expected = pred_data["expected"].astype(float)
        return predicted, expected

all_predicted_img_only, all_expected_img_only = load_predictions(img_only_fp)
all_predicted_img_meteo, all_expected_img_meteo = load_predictions(img_meteo_fp)

# Weight for the meteo model
alpha = 0.7

# Trim predictions to the same minimum length
min_len = min(len(all_predicted_img_only), len(all_predicted_img_meteo))
all_predicted_img_only = all_predicted_img_only[:min_len]
all_predicted_img_meteo = all_predicted_img_meteo[:min_len]
all_expected_img_only = all_expected_img_only[:min_len]

print("Predizione combinata con media ponderata completata.")

# === Load original data (datetime info, etc.) ===
data = np.load(npz_file, allow_pickle=True)

# === Instantiate Metrics & extract datetime lists ===
global_metrics1 = Metrics(
    all_expected_img_meteo,
    all_predicted_img_meteo,
    data,
    save_path="test_2",
    fp_images=FP_IMAGES,
    start_date="2023-01-01",
    end_date="2024-12-31",
    prediction_minutes=60,
    stats_for_month=False,
)
datetime1 = global_metrics1.datetime_list

global_metrics2 = Metrics(
    all_expected_img_only,
    all_predicted_img_only,
    data,
    save_path="test_1",
    fp_images=FP_IMAGES,
    start_date="2023-01-01",
    end_date="2024-12-31",
    prediction_minutes=60,
    stats_for_month=False,
)
datetime2 = global_metrics2.datetime_list

# Convert datetime objects to string arrays for comparison
datetime_img_only_str = np.array([str(dt) for dt in datetime2])[:len(all_predicted_img_only)]
datetime_img_meteo_str = np.array([str(dt) for dt in datetime1])[:len(all_predicted_img_meteo)]

# Find the common datetime strings
common_dt_str = np.intersect1d(datetime_img_only_str, datetime_img_meteo_str)

# Find indices for these common datetimes in both arrays
idx_img_only = np.array([np.where(datetime_img_only_str == dt)[0][0] for dt in common_dt_str])
idx_img_meteo = np.array([np.where(datetime_img_meteo_str == dt)[0][0] for dt in common_dt_str])

# Align predictions and expected arrays by these indices
pred_img_only_aligned = all_predicted_img_only[idx_img_only]
pred_img_meteo_aligned = all_predicted_img_meteo[idx_img_meteo]
expected_aligned = all_expected_img_only[idx_img_only]

# Compute weighted average predictions (now shapes match)
combined_predicted = alpha * pred_img_meteo_aligned + (1 - alpha) * pred_img_only_aligned
combined_expected = expected_aligned

# Instantiate combined metrics
global_metrics_combined = Metrics(
    combined_expected,
    combined_predicted,
    data,
    save_path="test_combined",
    fp_images=FP_IMAGES,
    start_date="2023-01-01",
    end_date="2024-12-31",
    prediction_minutes=60,
    stats_for_month=False,
)

# Plot curves for specific test days
global_metrics_combined.plotter.plot_day_curves(specific_test_days)
