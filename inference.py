import torch
import os
import numpy as np
import sys
import importlib
from datetime import datetime, timedelta
import random
import pandas as pd

from prepare_data_inference import PrepareData
from metrics_analysis.metrics import *

# Set device for torch (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is :", device)

MODEL_NUM = 8   # Model version number

FP_IMAGES = "/home/marta/Projects/tb/data/images/mch/1159"  # Default image path

# Default parameters
num_views = 1
seq_len = 3  # Sequence length (number of time steps)
prediction_minutes = 60  # Prediction interval in minutes

# Parse command-line arguments to override defaults
if len(sys.argv) > 1:
    if sys.argv[1] == "1":
        print("Train on chacha")
        FP_IMAGES = "/home/marta.rende/local_photocast/photocastv1_5/data/images/mch/1159"
        FP_IMAGES = os.path.normpath(FP_IMAGES)
    if len(sys.argv) > 2:
        if sys.argv[2] == "1":
            num_views = 1
        elif sys.argv[2] == "2":
            num_views = 2
    if len(sys.argv) > 3:
        seq_len = int(sys.argv[3])
    if len(sys.argv) > 4:
        prediction_minutes = int(sys.argv[4])
print(f"Using {num_views} views, sequence length: {seq_len}, prediction minutes: {prediction_minutes}")

# Model and data paths
MODEL_PATH = f"models/model_{MODEL_NUM}"
module_path = f"models.model_{MODEL_NUM}.model"
module = importlib.import_module(module_path)
StratusModel = getattr(module, "StratusModel")
npz_file = f"{MODEL_PATH}/test_data.npz"
fp_stats_stratus_days = f"{MODEL_PATH}/stratus_days_stats.npz"

# Load stratus days statistics
loaded = np.load(fp_stats_stratus_days, allow_pickle=True)
stratus_days_stats_loaded = loaded["stratus_days_stats"]

# Load model and weights
model = StratusModel(15, 2, num_views, seq_len)
model.load_state_dict(torch.load(f"{MODEL_PATH}/model.pth", map_location=device))
model = model.to(device)
model.eval()

# Load test data and statistics
data = np.load(npz_file, allow_pickle=True)
stats = np.load(f"{MODEL_PATH}/stats.npz", allow_pickle=True)
stats_input = stats["stats_input"].item()
stats_label = stats["stats_label"].item()
print(f"Stats keys: {stats}")

# Prepare to loop over months (example: March 2023)
results = {}
start_year = 2023
end_year = 2024
stratus_days = []
non_stratus_days = []
all_predicted = []
all_expected = []
months = [(2023, m) for m in range(1, 4)] + [(2023, m) for m in range(9, 13)] + [(2024, m) for m in range(1, 4)] + [(2024, m) for m in range(9, 13)]
# months = [(2024, m) for m in range(11, 12)]  # Only March 2023
# # months = ... # (commented: other months)
pred_file = os.path.join(MODEL_PATH, "predictions_vs_expected_ready.npz")
df_to_save = pd.DataFrame()
# Main inference loop over months
for year, month in months:
    if os.path.exists(pred_file):
        # If predictions already exist, skip computation
        break
    start_date = f"{year}-{month:02d}-01"
    # Calculate last day of the month
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    elif month == 3:
        next_month = datetime(year, 4, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    end_date = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\nProcessing from {start_date} to {end_date}")

    with torch.no_grad():
        # Prepare and load test data for the month
        prepare_data = PrepareData(fp_images=FP_IMAGES, fp_weather=npz_file, num_views=num_views, seq_length=seq_len, prediction_minutes=prediction_minutes)
        x_meteo, x_images, y_expected = prepare_data.load_data_test(start_date=start_date, end_date=end_date)
        
        if len(x_meteo) == 0 or len(x_images) == 0 or len(y_expected) == 0:
            print(f"No data found for {start_date} to {end_date}. Skipping this month.")
            continue

        # Identify stratus and non-stratus days
        stratus_days_for_month, non_stratus_days_for_month, _ = prepare_data.find_stratus_days(median_gap=stratus_days_stats_loaded[0], mad_gap=stratus_days_stats_loaded[1])
        print(f"Stratus days: {stratus_days_for_month}, non-stratus days: {non_stratus_days_for_month}")

        # Build variable order for normalization
        var_order = []
        for i in range(seq_len):
            var_order.append("gre000z0_nyon_t" + str(i))
            var_order.append("gre000z0_dole_t" + str(i))
            var_order.append("RR_t" + str(i))
            var_order.append("TD_t" + str(i))
            var_order.append("WG_t" + str(i))
            var_order.append("TT_t" + str(i))
            var_order.append("CT_t" + str(i))
            var_order.append("FF_t" + str(i))
            var_order.append("RS_t" + str(i))
            var_order.append("TG_t" + str(i))
            var_order.append("Z0_t" + str(i))
            var_order.append("ZS_t" + str(i))
            var_order.append("SU_t" + str(i))
            var_order.append("DD_t" + str(i))
            var_order.append("pres_t" + str(i))
        # Normalize meteorological data
        x_meteo = prepare_data.normalize_data_test(
            x_meteo,
            var_order=var_order,
            stats=stats_input,
        )

        total_predictions = len(x_meteo)
        print(f"Total predictions: {total_predictions}")
        y_predicted = []
        final_expected = []

        # Convert data to torch tensors
        x_meteo_tensor = torch.tensor(x_meteo, dtype=torch.float32).to(device)
        x_images_tensor1 = None
        x_images_tensor2 = None
        if num_views == 2:
            x_images_tensor1 = torch.tensor(x_images[:, :, 0], dtype=torch.float32).permute(0, 1, 4, 2, 3).to(device)
            x_images_tensor2 = torch.tensor(x_images[:, :, 1], dtype=torch.float32).permute(0, 1, 4, 2, 3).to(device)
        else:
            x_images_tensor = torch.tensor(x_images, dtype=torch.float32).permute(0, 1, 4, 2, 3).to(device)

        # Run inference for each sample (using only a quarter of the data)
        for i in range(int(total_predictions)):
            idx_test = i
            x_meteo_sample = x_meteo_tensor[idx_test].unsqueeze(0).to(device)
            y = None
            if num_views == 2:
                img_seq1 = x_images_tensor1[i].unsqueeze(0)
                img_seq2 = x_images_tensor2[i].unsqueeze(0)
                y = model(x_meteo_sample, img_seq1, img_seq2)
            else:
                img_seq = x_images_tensor[i].unsqueeze(0)
                y = model(x_meteo_sample, img_seq)
            y = y.squeeze(0).cpu().numpy()
            expected = y_expected[idx_test]
            # Denormalize predictions
            min_nyon = stats_label["gre000z0_nyon"]["min"]
            max_nyon = stats_label["gre000z0_nyon"]["max"]
            min_dole = stats_label["gre000z0_dole"]["min"]
            max_dole = stats_label["gre000z0_dole"]["max"]
            y[0] = y[0] * (max_nyon - min_nyon) + min_nyon
            y[1] = y[1] * (max_dole - min_dole) + min_dole

            y_predicted.append(y)
            final_expected.append(expected)

        all_predicted.append(y_predicted)
        all_expected.append(final_expected)
        
        stratus_days.append(stratus_days_for_month)
        non_stratus_days.append(non_stratus_days_for_month)

        # Compute and plot metrics for the month
        metrics = Metrics(final_expected, y_predicted, data, save_path=MODEL_PATH, fp_images=FP_IMAGES, start_date=start_date, end_date=end_date, prediction_minutes=prediction_minutes)
        metrics.plotter.plot_day_curves(stratus_days_for_month)
        # Plot up to 3 random non-stratus days
        num_days_to_plot = min(3, len(non_stratus_days_for_month))
        if num_days_to_plot > 0:
            random_non_stratus_days = random.sample(non_stratus_days_for_month, num_days_to_plot)
            print(f"Random non-stratus days selected for plotting: {random_non_stratus_days}")
            metrics.plotter.plot_day_curves(random_non_stratus_days)
        else:
            print("No non-stratus days to select for plotting.")
        metrics.compute_and_save_metrics_by_month(stratus_days_for_month)
        metrics.compute_and_save_metrics_by_month(non_stratus_days_for_month, label="non_stratus_days")


# Flatten all_expected and all_predicted lists if any predictions were made
if len(all_expected) > 0 or len(all_predicted) > 0:
    all_expected = [item for sublist in all_expected for item in sublist]
    all_predicted = [item for sublist in all_predicted for item in sublist]
    # Save predictions and expected values for later use
    np.savez(
        os.path.join(MODEL_PATH, "predictions_vs_expected_ready.npz"),
        predicted=np.array(all_predicted, dtype=np.float32),
        expected=np.array(all_expected, dtype=np.float32),
    )
else:
    # If predictions already exist, load them
    with np.load(pred_file, allow_pickle=True) as pred_data:
        all_predicted = pred_data["predicted"]
        all_expected = pred_data["expected"]
        all_expected.astype(float)
        all_predicted.astype(float)

# Compute global metrics and plots
global_metrics = Metrics(
    all_expected, all_predicted, data, save_path=MODEL_PATH, start_date="2023-01-01", end_date="2024-12-31", prediction_minutes=prediction_minutes, stats_for_month=False,
)

df_to_save = global_metrics._create_comparison_dataframe()
# Save df_to_save to a CSV file if the model path exists
csv_path = os.path.join(MODEL_PATH, "comparison_dataframe.csv")
if not  os.path.exists(csv_path):
    df_to_save.to_csv(csv_path, index=False)
else:
    print(f"File alraedy saved")

specific_test_days = [
    "2023-03-02", "2024-12-26", "2023-02-13", "2024-10-25", "2024-11-03", 
    "2024-11-08", "2023-01-27", "2023-01-25", "2023-02-09", "2024-10-30",
    "2024-11-09", "2024-10-19", "2024-11-16"
]

# Various plots and analyses for specific days
global_metrics.save_metrics_report(
    stratus_days=specific_test_days, non_stratus_days=non_stratus_days
)
# global_metrics.plotter.plot_residual_errors(specific_test_days)
# global_metrics.plotter.plot_delta_scatter([], "geneva")
# global_metrics.plotter.plot_delta_scatter([], "dole")
# global_metrics.plotter.plot_delta_scatter([])
# global_metrics.plotter.plot_delta_scatter(specific_test_days, "geneva")
# global_metrics.plotter.plot_delta_scatter(specific_test_days, "dole")
# global_metrics.plotter.plot_delta_scatter(specific_test_days)
# global_metrics.plotter.plot_delta_error_heatmap(specific_test_days)
