from model import StratusModel
from prepareData import PrepareData
import torch
import netCDF4
import glob
import os
import h5py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is :", device)

prepare_data = PrepareData()
model = StratusModel()
# TODO : change path
model.load_state_dict(torch.load("models/model_1/model.pth", map_location=device))
DATETIME = "2023-01-29T08:30:00"
WEATHER_FP = "/home/marta/Projects/tb/data/weather/inca/2023"
with torch.no_grad():
    wanted_date = "20230129"
    # load data test of npz file
    npz_file = f"data/test/dole_test_data.npz"
    
  

    x_meteo, x_image, y_expected = prepare_data.load_data(npz_file)
    # normalize the data
    x_meteo = prepare_data.normalize_data(x_meteo, var_order = [
        "nyon_gre000z0", "dole_gre000z0","RR", "TD", "WG", "TT", "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD"
        ])
    y_expected = prepare_data.normalize_data(y_expected)
    x_meteo = torch.tensor(x_meteo, dtype=torch.float32).to(device)  # [991, 15]
    x_images = torch.tensor(x_image, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # [991, 3, 512, 512]

    x_meteo_sample = x_meteo[3].unsqueeze(0)      # [1, 15]
    x_image_sample = x_images[3].unsqueeze(0)     # [1, 3, 512, 512]

    print("x_meteo_sample", x_meteo_sample.shape)
    y = model(x_meteo_sample, x_image_sample)
    print("Predicted values:", y, "Expected values:", y_expected[3])
