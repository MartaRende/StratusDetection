from model import StratusModel
from prepareData import PrepareData
import torch
import netCDF4
import glob
import os
import h5py
import numpy as np
from metrics import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is :", device)
MODEL_PATH = "models/model_0"
prepare_data = PrepareData()
model = StratusModel()
# TODO : change path
model.load_state_dict(torch.load(f"{MODEL_PATH}/model.pth", map_location=device))
DATETIME = "2023-01-29T08:30:00"
WEATHER_FP = "/home/marta/Projects/tb/data/weather/inca/2023"
with torch.no_grad():
    # load data test of npz file
    npz_file = f"data/test/test_with_datetime.npz"
    data = np.load(npz_file, allow_pickle=True)
    

    x_meteo, x_image, y_expected = prepare_data.load_data(npz_file)
    # normalize the data
    x_meteo, _ = prepare_data.normalize_data(
        x_meteo,
        var_order=["gre000z0_nyon", "gre000z0_dole", "RR", "TD", "WG", "TT", "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD"])
    y_expected, stats = prepare_data.normalize_data(
        y_expected,
        var_order=["gre000z0_nyon", "gre000z0_dole"]
    )

   
    print(stats)
    total_predictions = len(x_meteo)
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
        print(idx_test)
        
        y = model(x_meteo_sample, x_image_sample)
        y = y.squeeze(0).cpu().numpy()
        expected = y_expected[idx_test]
        tol = 20  # tolerance value
        # read stats values
        
        mean_nyon = stats["gre000z0_nyon"]["mean"]
        mean_dole = stats["gre000z0_dole"]["mean"]
        std_nyon = stats["gre000z0_nyon"]["std"]
        std_dole = stats["gre000z0_dole"]["std"]

        # Denormalize predicted values
        y[0] = y[0] * std_nyon + mean_nyon
        y[1] = y[1] * std_dole + mean_dole

        # Denormalize expected values
        expected[0] = expected[0] * std_nyon + mean_nyon
        expected[1] = expected[1] * std_dole + mean_dole
        
        y_predicted.append(y)
        final_expected.append(expected)
    metrics = Metrics(final_expected, y_predicted,data, save_path=MODEL_PATH)
    metrics.find_datetimes()

    mae = metrics.get_mean_absolute_error()
    metrics.plot_mae(mae)
    
    mse = metrics.get_mse()
    metrics.plot_mse(mse)
    
    relative_error = metrics.get_relative_error()
    metrics.plot_relative_error(relative_error)
    
    delta = metrics.get_delta_between_expected_and_predicted()
    metrics.plot_delta(delta)