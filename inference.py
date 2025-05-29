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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is :", device)
MODEL_NUM = 5  # or any number you want
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
prepare_data = PrepareData(fp_images=FP_IMAGES, fp_weather=npz_file)
model.load_state_dict(torch.load(f"{MODEL_PATH}/model.pth", map_location=device))
model = model.to(device)
model.eval()
# load data test of npz file

data = np.load(npz_file, allow_pickle=True)
stats = np.load(f"{MODEL_PATH}/stats.npz", allow_pickle=True)
stats_input = stats["stats_input"].item()
stats_label = stats["stats_label"].item()
print(f"Stats keys: {stats}")
start_date = "2024-03-01"
end_date = "2024-03-31"
with torch.no_grad():
    x_meteo, x_image, y_expected = prepare_data.load_data( start_date=start_date, end_date=end_date)
    stratus_days = prepare_data.find_stratus_days()
    print(f"Stratus days: {stratus_days}")
    # normalize the data
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
        ],
        stats=stats_input,
    )
    y_expected = prepare_data.normalize_data_test(
        y_expected,
        var_order=["gre000z0_nyon", "gre000z0_dole"],
        stats=stats_label,
    )

    # print(stats)
    total_predictions = len(x_meteo)
    print(f"Total predictions: {total_predictions}")
    y_predicted = []
    final_expected = []
    for i in range(total_predictions):
        x_meteo = torch.tensor(x_meteo, dtype=torch.float32).to(device)  # [991, 15]
        x_images = (
            torch.tensor(x_image, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        )  # [991, 3, 512, 512]
        idx_test = i
        x_meteo_sample = x_meteo[idx_test].unsqueeze(0)  # [1, 15]
        x_image_sample = x_images[idx_test].unsqueeze(0)  # [1, 3, 512, 512]

        y = model(x_meteo_sample, x_image_sample)
        y = y.squeeze(0).cpu().numpy()
        expected = y_expected[idx_test]
        # read stats values

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
    metrics = Metrics(final_expected, y_predicted, data, save_path=MODEL_PATH, start_date=start_date, end_date=end_date)

    metrics.print_datetimes()
    
    
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
    metrics.plot_delta(delta)
    
    for i in stratus_days:
        print(f"Stratus day: {i}")
        metrics.plot_day_curves(i)
    metrics.plot_random_days(exclude_days=stratus_days)