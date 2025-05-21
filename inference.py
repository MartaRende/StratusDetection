from model import StratusModel
from prepareData import PrepareData
import torch
import netCDF4
import glob
import os
import h5py
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is :", device)

prepare_data = PrepareData()
model = StratusModel()
# TODO : change path
model.load_state_dict(torch.load("models/model_10/model.pth", map_location=device))
DATETIME = "2023-01-29T08:30:00"
WEATHER_FP = "/home/marta/Projects/tb/data/weather/inca/2023"
with torch.no_grad():
    wanted_date = "20230129"
    # load data test of npz file
    npz_file = f"data/test/dole_test_data.npz"
    data = np.load(npz_file, allow_pickle=True)
    good_prediction = 0
    x_meteo, x_image, y_expected, stats = prepare_data.load_data(npz_file)
    # TODO : save right path

    print(stats)
    total_predictions = len(x_meteo)
    for i in range(total_predictions):
        x_meteo = torch.tensor(x_meteo, dtype=torch.float32).to(device)  # [991, 15]
        x_images = (
            torch.tensor(x_image, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        )  # [991, 3, 512, 512]
        idx_test = i
        x_meteo_sample = x_meteo[idx_test].unsqueeze(0)  # [1, 15]
        x_image_sample = x_images[idx_test].unsqueeze(0)  # [1, 3, 512, 512]

        print("x_meteo_sample", x_meteo_sample.shape)
        y = model(x_meteo_sample, x_image_sample)
        y = y.squeeze(0).cpu().numpy()
        expected = y_expected[idx_test]
        tol = 20  # tolerance value
        # read stats values
        stats = np.load(f"stats.npy", allow_pickle=True).item()
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
        good_prediction += int(np.all(np.abs(y - expected) <= tol))
        print("Predicted values:", y, "Expected values:", expected)

        matching_items = [
            (i, item)
            for (i, item) in enumerate(data["dole"])
            if abs(item["gre000z0_dole"] - expected[1]) <= 1
            and abs(item["gre000z0_nyon"] - expected[0]) <= 1
        ]

        index = matching_items[0][0]
        print("Matching items:", matching_items[0][0])
        datetime = data["dole"][index]["datetime"]
        print("Datetime:", datetime)
    


print("Good predictions:", good_prediction, "Total predictions:", total_predictions)
print("Accuracy:", good_prediction / total_predictions)
