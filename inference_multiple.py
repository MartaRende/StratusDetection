from prepareData import PrepareData
import torch
import netCDF4
import glob
import os
import h5py
import numpy as np
import sys
from metrics import Metrics
import importlib
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is :", device)
MODEL_NUM = 72 # or any number you want

FP_IMAGES = "/home/marta/Projects/tb/data/images/mch/1159"

num_views = 1
seq_len = 3  # Number of time steps in the sequence
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
MODEL_PATH = f"models/model_{MODEL_NUM}"
module_path = f"models.model_{MODEL_NUM}.model"
module = importlib.import_module(module_path)
StratusModel = getattr(module, "StratusModel")
npz_file = f"{MODEL_PATH}/test_data.npz"
fp_stats_stratus_days = f"{MODEL_PATH}/stratus_days_stats.npz"
loaded = np.load(fp_stats_stratus_days, allow_pickle=True)
stratus_days_stats_loaded = loaded["stratus_days_stats"]
model = StratusModel(15, 12, num_views, seq_len)
model.load_state_dict(torch.load(f"{MODEL_PATH}/model.pth", map_location=device))
model = model.to(device)
model.eval()

# load data test of npz file
data = np.load(npz_file, allow_pickle=True)
stats = np.load(f"{MODEL_PATH}/stats.npz", allow_pickle=True)

stats_input = stats["stats_input"].item()
stats_label = stats["stats_label"].item()

print(f"Stats keys: {stats}")

# Define prediction times from t_0 to t_5
prediction_times = [f"t_{i}" for i in range(6)]

# Loop over months from September (9) to March (3) of the next year
results = {}
start_year = 2023
end_year = 2024
stratus_days = []
non_stratus_days = []
all_predicted = {t: [] for t in prediction_times}
all_expected = {t: [] for t in prediction_times}

months = [(2024, m) for m in range(12, 13)]
#+ [(2023, m) for m in range(9, 13)] + [(2024, m) for m in range(1, 4)] + [(2024, m) for m in range(9, 13)]

for year, month in months:
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
        prepare_data = PrepareData(fp_images=FP_IMAGES, fp_weather=npz_file, num_views=num_views, seq_length=seq_len)   
        x_meteo, x_images, y_expected = prepare_data.load_data_test(start_date=start_date, end_date=end_date)
        
        if len(x_meteo) == 0 or len(x_images) == 0 or len(y_expected) == 0:
            print(f"No data found for {start_date} to {end_date}. Skipping this month.")
            continue
            
        stratus_days_for_month, non_stratus_days_for_month, _ = prepare_data.find_stratus_days(
            median_gap=stratus_days_stats_loaded[0], 
            mad_gap=stratus_days_stats_loaded[1]
        )
        print(f"Stratus days: {stratus_days_for_month}, non-stratus days: {non_stratus_days_for_month}")
        
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
        #var_order = ["gre000z0", "RR", "TD", "WG", "TT", "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD", "pres"] 
        x_meteo = prepare_data.normalize_data_test(
            x_meteo,
            var_order=var_order,
            stats=stats_input,
        )

        total_predictions = len(x_meteo)
        print(f"Total predictions: {total_predictions}")
        
        y_predicted = {t: [] for t in prediction_times}
        final_expected = {t: [] for t in prediction_times}

        x_meteo_tensor = torch.tensor(x_meteo, dtype=torch.float32).to(device)
        x_images_tensor1 = None
        x_images_tensor2 = None
        
        if num_views == 2:
            x_images_tensor1 = torch.tensor(x_images[:, :, 0], dtype=torch.float32).permute(0, 1, 4, 2, 3).to(device)
            x_images_tensor2 = torch.tensor(x_images[:, :, 1], dtype=torch.float32).permute(0, 1, 4, 2, 3).to(device)
        else:
            x_images_tensor = torch.tensor(x_images, dtype=torch.float32).permute(0, 1, 4, 2, 3).to(device)
            
        for i in range(total_predictions):
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
            
            # Process all prediction times from t_0 to t_5
            for t in range(6):  # 0-5 corresponds to t_0 to t_5
                time_key = f"t_{t}"
                
                # Get normalization parameters for this time step
                min_nyon = stats_label[f"gre000z0"]["min"]
                max_nyon = stats_label[f"gre000z0"]["max"]
                min_dole = stats_label[f"gre000z0"]["min"]
                max_dole = stats_label[f"gre000z0"]["max"]
                
                # Assuming the output is structured as [nyon_t0, dole_t0, nyon_t1, dole_t1, ...]
                nyon_idx = 2 * t
                dole_idx = 2 * t + 1
                
                # Denormalize predictions
                y[nyon_idx] = y[nyon_idx] * (max_nyon - min_nyon) + min_nyon
                y[dole_idx] = y[dole_idx] * (max_dole - min_dole) + min_dole
                
                y_predicted[time_key].append([y[nyon_idx], y[dole_idx]])
                final_expected[time_key].append([expected[t][0], expected[t][1]])
           
        stratus_days.append(stratus_days_for_month)
        non_stratus_days.append(non_stratus_days_for_month)
       
        # Plot curves for random non-stratus days
        num_days_to_plot = min(3, len(non_stratus_days_for_month))
        random_non_stratus_days = []
        if num_days_to_plot > 0:
            random_non_stratus_days = random.sample(non_stratus_days_for_month, num_days_to_plot)
            print(f"Random non-stratus days selected for plotting: {random_non_stratus_days}")
        else:
            print("No non-stratus days to select for plotting.")
            
        
        # Append to global results
        for t in prediction_times:

            all_predicted[t].extend(y_predicted[t])
            all_expected[t].extend(final_expected[t])

       
            metrics = Metrics(final_expected[t], y_predicted[t], data, save_path=MODEL_PATH,
                            fp_images=FP_IMAGES, start_date=start_date, end_date= end_date, time_key=t)
            
            # Plot curves for stratus days
            metrics.plot_day_curves(stratus_days_for_month)
            
            # Plot curves for random non-stratus days
           
            metrics.plot_day_curves(random_non_stratus_days)
        
            # Compute metrics for this month
            metrics.compute_and_save_metrics_by_month(stratus_days_for_month)
            metrics.compute_and_save_metrics_by_month(non_stratus_days_for_month, label="non_stratus_days")

for t in prediction_times:
    
# Create global metrics instance
    global_metrics = Metrics(
        all_expected[t], all_predicted[t], data, save_path=MODEL_PATH, 
        start_date="2023-01-01", end_date="2024-12-31", time_key=t,stats_for_month=False
    )

    global_metrics.save_metrics_report(
        stratus_days=stratus_days, non_stratus_days=non_stratus_days
    )
    
# Add this after your existing code, inside the same script

time_steps = prediction_times
metrics_collection = {
    'mae': [],
    'rmse': [],
    'rel_err': []
}

for t in time_steps:
    # Create metrics instance for this time step
    metrics = Metrics(
        all_expected[t], all_predicted[t], data, save_path=MODEL_PATH, 
        fp_images=FP_IMAGES, start_date="2023-01-01", end_date="2024-12-31", 
        time_key=t, stats_for_month=False
    )
    
    # Compute metrics
    mae_global, rmse_global, rel_err_global = metrics.get_mean_absolute_error(), metrics.get_root_mean_squared_error(), metrics.get_mean_relative_error()
    non_stratus_metrics = metrics.get_global_metrics_for_days(non_stratus_days)
    stratus_days_metrics = metrics.get_global_metrics_for_days(stratus_days)
    # Store metrics
    metrics_collection['mae'].append(mae_global)
    metrics_collection['rmse'].append(rmse_global)
    metrics_collection['rel_err'].append(rel_err_global)
    # Plot metrics for stratus and non-stratus days

    # Now plot the metrics across time steps on three different subplots



fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

mae_nyon = [v['nyon'] for v in metrics_collection['mae']]
mae_dole = [v['dole'] for v in metrics_collection['mae']]
rmse_nyon = [v['nyon'] for v in metrics_collection['rmse']]
rmse_dole = [v['dole'] for v in metrics_collection['rmse']]
relerr_nyon = [float(v['nyon']) for v in metrics_collection['rel_err']]
relerr_dole = [float(v['dole']) for v in metrics_collection['rel_err']]

axs[0].plot(time_steps, mae_nyon, 'o-', color='blue', label='Nyon')
axs[0].plot(time_steps, mae_dole, 's-', color='red', label='Dole')
axs[0].set_ylabel('MAE')
axs[0].set_title('Mean Absolute Error (MAE)')
axs[0].set_xlabel('Time Step')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)

axs[1].plot(time_steps, rmse_nyon, 'o--', color='blue', label='Nyon')
axs[1].plot(time_steps, rmse_dole, 's--', color='red', label='Dole')
axs[1].set_ylabel('RMSE')
axs[1].set_title('Root Mean Square Error (RMSE)')
axs[1].set_xlabel('Time Step')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.6)

axs[2].plot(time_steps, relerr_nyon, 'o-.', color='blue', label='Nyon')
axs[2].plot(time_steps, relerr_dole, 's-.', color='red', label='Dole')
axs[2].set_ylabel('Relative Error')
axs[2].set_title('Relative Error')
axs[2].set_xlabel('Time Step')
axs[2].legend()
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(pad=3.0)


combined_plot_path = os.path.join(MODEL_PATH, "metrics_across_times.png")
plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved combined metrics plot to {combined_plot_path}")

# Extract and plot metrics for stratus and non-stratus days across time steps

stratus_mae_nyon = []
stratus_mae_dole = []
stratus_rmse_nyon = []
stratus_rmse_dole = []
stratus_relerr_nyon = []
stratus_relerr_dole = []

non_stratus_mae_nyon = []
non_stratus_mae_dole = []
non_stratus_rmse_nyon = []
non_stratus_rmse_dole = []
non_stratus_relerr_nyon = []
non_stratus_relerr_dole = []

for t in time_steps:
    # Metrics for stratus days
    stratus_metrics = Metrics(
        all_expected[t], all_predicted[t], data, save_path=MODEL_PATH, 
        fp_images=FP_IMAGES, start_date="2023-01-01", end_date="2024-12-31", 
        time_key=t, stats_for_month=False
    ).get_global_metrics_for_days(stratus_days)
    # Extract metrics according to new return structure
    stratus_mae = stratus_metrics.get("mae", {"nyon": None, "dole": None})
    stratus_rmse = stratus_metrics.get("rmse", {"nyon": None, "dole": None})
    stratus_rel_err = stratus_metrics.get("relative_error", {"nyon": None, "dole": None})
    stratus_mae_nyon.append(stratus_mae["nyon"])
    stratus_mae_dole.append(stratus_mae["dole"])
    stratus_rmse_nyon.append(stratus_rmse["nyon"])
    stratus_rmse_dole.append(stratus_rmse["dole"])
    stratus_relerr_nyon.append(float(stratus_rel_err["nyon"]) if stratus_rel_err["nyon"] is not None else None)
    stratus_relerr_dole.append(float(stratus_rel_err["dole"]) if stratus_rel_err["dole"] is not None else None)

    # Metrics for non-stratus days
    non_stratus_metrics = Metrics(
        all_expected[t], all_predicted[t], data, save_path=MODEL_PATH, 
        fp_images=FP_IMAGES, start_date="2023-01-01", end_date="2024-12-31", 
        time_key=t, stats_for_month=False
    ).get_global_metrics_for_days(non_stratus_days)

    non_stratus_mae = non_stratus_metrics.get("mae", {"nyon": None, "dole": None})
    non_stratus_rmse = non_stratus_metrics.get("rmse", {"nyon": None, "dole": None})
    non_stratus_rel_err = non_stratus_metrics.get("relative_error", {"nyon": None, "dole": None})
    non_stratus_mae_nyon.append(non_stratus_mae["nyon"])
    non_stratus_mae_dole.append(non_stratus_mae["dole"])
    non_stratus_rmse_nyon.append(non_stratus_rmse["nyon"])
    non_stratus_rmse_dole.append(non_stratus_rmse["dole"])
    non_stratus_relerr_nyon.append(float(non_stratus_rel_err["nyon"]) if non_stratus_rel_err["nyon"] is not None else None)
    non_stratus_relerr_dole.append(float(non_stratus_rel_err["dole"]) if non_stratus_rel_err["dole"] is not None else None)

# Plot metrics for stratus and non-stratus days
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# MAE
axs[0].plot(time_steps, stratus_mae_nyon, 'o-', color='blue', label='Stratus Nyon')
axs[0].plot(time_steps, stratus_mae_dole, 's-', color='red', label='Stratus Dole')
axs[0].plot(time_steps, non_stratus_mae_nyon, 'o--', color='cyan', label='Non-Stratus Nyon')
axs[0].plot(time_steps, non_stratus_mae_dole, 's--', color='orange', label='Non-Stratus Dole')
axs[0].set_ylabel('MAE')
axs[0].set_title('MAE: Stratus vs Non-Stratus Days')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)

# RMSE
axs[1].plot(time_steps, stratus_rmse_nyon, 'o-', color='blue', label='Stratus Nyon')
axs[1].plot(time_steps, stratus_rmse_dole, 's-', color='red', label='Stratus Dole')
axs[1].plot(time_steps, non_stratus_rmse_nyon, 'o--', color='cyan', label='Non-Stratus Nyon')
axs[1].plot(time_steps, non_stratus_rmse_dole, 's--', color='orange', label='Non-Stratus Dole')
axs[1].set_ylabel('RMSE')
axs[1].set_title('RMSE: Stratus vs Non-Stratus Days')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.6)

# Relative Error
axs[2].plot(time_steps, stratus_relerr_nyon, 'o-', color='blue', label='Stratus Nyon')
axs[2].plot(time_steps, stratus_relerr_dole, 's-', color='red', label='Stratus Dole')
axs[2].plot(time_steps, non_stratus_relerr_nyon, 'o--', color='cyan', label='Non-Stratus Nyon')
axs[2].plot(time_steps, non_stratus_relerr_dole, 's--', color='orange', label='Non-Stratus Dole')
axs[2].set_ylabel('Relative Error')
axs[2].set_title('Relative Error: Stratus vs Non-Stratus Days')
axs[2].set_xlabel('Time Step')
axs[2].legend()
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(pad=3.0)
stratus_nonstratus_plot_path = os.path.join(MODEL_PATH, "stratus_vs_nonstratus_metrics.png")
plt.savefig(stratus_nonstratus_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved stratus vs non-stratus metrics plot to {stratus_nonstratus_plot_path}")
stratus_mae, stratus_rmse, stratus_rel_err = stratus_days_metrics
non_stratus_mae, non_stratus_rmse, non_stratus_rel_err = non_stratus_metrics

