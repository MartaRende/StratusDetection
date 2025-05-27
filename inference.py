from model import StratusModel
from prepareData import PrepareData
import torch
import numpy as np
import sys
from metrics import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is:", device)

MODEL_PATH = "models/model_3"
npz_file = f"{MODEL_PATH}/test_data.npz"
FP_IMAGES = "/home/marta/Projects/tb/data/images/mch/1159/2/"

if len(sys.argv) > 1 and sys.argv[1] == "1":
    print("Train on chacha")
    FP_IMAGES = "/home/marta.rende/local_photocast/photocastv1_5/data/images/mch/1159/2"
    FP_IMAGES = os.path.normpath(FP_IMAGES)

# Initialize and load model
model = StratusModel().to(device)
model.load_state_dict(torch.load(f"{MODEL_PATH}/model.pth", map_location=device))
model.eval() 

# Load and prepare data
prepare_data = PrepareData(fp_images=FP_IMAGES, fp_weather=npz_file)
data = np.load(npz_file, allow_pickle=True)

with torch.no_grad():
    x_meteo, x_image, y_expected = prepare_data.load_data(npz_file)
    
    # Normalize data
    x_meteo, _ = prepare_data.normalize_data(
        x_meteo,
        var_order=["gre000z0_nyon", "gre000z0_dole", "RR", "TD", "WG", "TT", "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD"])
    
    y_expected, stats = prepare_data.normalize_data(
        y_expected,
        var_order=["gre000z0_nyon", "gre000z0_dole"]
    )

    # Convert to tensors once (outside loop)
    x_meteo_tensor = torch.tensor(x_meteo, dtype=torch.float32).to(device)
    x_images_tensor = torch.tensor(x_image, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    
    # Batch processing instead of individual samples
    y_predicted = model(x_meteo_tensor, x_images_tensor).cpu().numpy()
    
    # Denormalize
    mean_nyon = stats["gre000z0_nyon"]["mean"]
    mean_dole = stats["gre000z0_dole"]["mean"]
    std_nyon = stats["gre000z0_nyon"]["std"]
    std_dole = stats["gre000z0_dole"]["std"]

    y_predicted[:, 0] = y_predicted[:, 0] * std_nyon + mean_nyon
    y_predicted[:, 1] = y_predicted[:, 1] * std_dole + mean_dole
    
    y_expected[:, 0] = y_expected[:, 0] * std_nyon + mean_nyon
    y_expected[:, 1] = y_expected[:, 1] * std_dole + mean_dole

# Calculate metrics
metrics = Metrics(y_expected, y_predicted, data, save_path=MODEL_PATH)
metrics.print_datetimes()

print(f"Accuracy: {metrics.get_accuracy() * 100:.2f}%")
print(f"Mean Absolute Error: {metrics.get_mean_absolute_error()}")
print(f"Root Mean Squared Error: {metrics.get_rmse()}") 
print(f"Mean Relative Error: {metrics.mean_relative_error()}")

metrics.plot_relative_error(metrics.get_relative_error())
metrics.plot_delta(metrics.get_delta_between_expected_and_predicted())