from prepareData import PrepareData
import torch
import netCDF4
import glob
import os
import h5py
import numpy as np
import sys
from metrics import *
import importlib
from datetime import datetime, timedelta

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is :", device)
MODEL_NUM =11  # or any number you want
MODEL_PATH = f"models/model_{MODEL_NUM}"
module_path = f"models.model_{MODEL_NUM}.model"
module = importlib.import_module(module_path)
StratusModel = getattr(module, "StratusModel")
model = StratusModel()


npz_file = f"{MODEL_PATH}/test_data.npz"
FP_IMAGES = "/home/marta/Projects/tb/data/images/mch/1159/2/"
if len(sys.argv) > 1:
    if sys.argv[1] == "1":
        print("Train on chacha")
        FP_IMAGES = (
            "/home/marta.rende/local_photocast/photocastv1_5/data/images/mch/1159/2"
        )
        FP_IMAGES = os.path.normpath(FP_IMAGES)

model.load_state_dict(torch.load(f"{MODEL_PATH}/model.pth", map_location=device))
model = model.to(device)
model.eval()
# load data test of npz file

data = np.load(npz_file, allow_pickle=True)
stats = np.load(f"{MODEL_PATH}/stats.npz", allow_pickle=True)
stats_input = stats["stats_input"].item()
stats_label = stats["stats_label"].item()
print(f"Stats keys: {stats}")

# Loop over months from September (9) to March (3) of the next year

results = {}
start_year = 2023
end_year = 2024
stratus_days = []
all_predicted = []
all_expected = []
months = [(2023, m) for m in range(1, 2)]
#+  [(2023, m) for m in range(9, 13)] +  [(2024, m) for m in range(1, 4)] + [(2024, m) for m in range(9, 13)]

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
        prepare_data = PrepareData(fp_images=FP_IMAGES, fp_weather=npz_file)
        x_meteo, x_image, y_expected = prepare_data.load_data(start_date=start_date, end_date=end_date)
      
        if len(x_meteo) == 0 or len(x_image) == 0 or len(y_expected) == 0:
            print(f"No data found for {start_date} to {end_date}. Skipping this month.")
            continue
        stratus_days_for_month = prepare_data.find_stratus_days()
        stratus_days.append(stratus_days_for_month)
        print(f"Stratus days: {stratus_days_for_month}")
        x_meteo = prepare_data.normalize_data_test(
            x_meteo,
            var_order=[
                "gre000z0_nyon",
                "gre000z0_dole",
                "RR",
                "TD",
                "WG",
                "TT",
                "CT",
                "FF",
                "RS",
                "TG",
                "Z0",
                "ZS",
                "SU",
                "DD",
                "pres"
            ],
            stats=stats_input,
        )
        y_expected = prepare_data.normalize_data_test(
            y_expected,
            var_order=["gre000z0_nyon", "gre000z0_dole"],
            stats=stats_label,
        )

        total_predictions = len(x_meteo)
        print(f"Total predictions: {total_predictions}")
        y_predicted = []
        final_expected = []
        for i in range(total_predictions):
            x_meteo_tensor = torch.tensor(x_meteo, dtype=torch.float32).to(device)
            x_images_tensor = torch.tensor(x_image, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            idx_test = i
            x_meteo_sample = x_meteo_tensor[idx_test].unsqueeze(0)
            x_image_sample = x_images_tensor[idx_test].unsqueeze(0)
            y = model(x_meteo_sample, x_image_sample)
            y = y.squeeze(0).cpu().numpy()
            expected = y_expected[idx_test]
            min_nyon = stats_label["gre000z0_nyon"]["min"]
            max_nyon = stats_label["gre000z0_nyon"]["max"]
            min_dole = stats_label["gre000z0_dole"]["min"]
            max_dole = stats_label["gre000z0_dole"]["max"]
            y[0] = y[0] * (max_nyon - min_nyon) + min_nyon
            y[1] = y[1] * (max_dole - min_dole) + min_dole
            expected[0] = expected[0] * (max_nyon - min_nyon) + min_nyon
            expected[1] = expected[1] * (max_dole - min_dole) + min_dole
            y_predicted.append(y)
            final_expected.append(expected)
         
        all_predicted.append(y_predicted)
        all_expected.append(final_expected)

    
        metrics = Metrics(final_expected, y_predicted, data, save_path=MODEL_PATH, start_date=start_date, end_date=end_date)
        accuracy = metrics.get_accuracy()
        print(f"Accuracy: {accuracy * 100:.2f}%")
        mae = metrics.get_mean_absolute_error()
        print(f"Mean Absolute Error: {mae}")
        rmse = metrics.get_rmse()
        print(f"Root Mean Squared Error: {rmse}")
        mre = metrics.mean_relative_error()
        print(f"Mean Relative Error: {mre}")
        relative_error = metrics.get_relative_error()
        metrics.plot_relative_error()
        delta = metrics.get_delta_between_expected_and_predicted()
        metrics.plot_rmse(delta)
        for i in stratus_days_for_month:
            print(f"Stratus day: {i}")
            metrics.plot_day_curves(i)
        metrics.plot_random_days(exclude_days=stratus_days_for_month)
        metrics.save_metrics()
        print("strtaus days for month:", stratus_days)

# Flatten all_expected into a 1D array
all_expected =  [item for sublist in all_expected for item in sublist]
all_predicted =  [item for sublist in all_predicted for item in sublist]

global_metrics = Metrics(
    all_expected, all_predicted, data, save_path=MODEL_PATH, start_date="2023-01-01", end_date="2024-12-31", stats_for_month=False
)
global_metrics_mse  = global_metrics.get_rmse()
print(f"Global RMSE: {global_metrics_mse}")
print(f"Global Accuracy: {global_metrics.get_accuracy() * 100:.2f}%")

global_metrics_mre = global_metrics.mean_relative_error()
print(f"Global Mean Relative Error: {global_metrics_mre}")

global_metrics_mae = global_metrics.get_mean_absolute_error()
print(f"Global Mean Absolute Error: {global_metrics_mae}")


global_metrics.plot_rmse_for_specific_days(stratus_days)
non_stratus_days = global_metrics.find_unique_days_non_startus(stratus_days)

global_metrics.plot_rmse_for_specific_days(non_stratus_days, stratus_days="non_stratus_days")
