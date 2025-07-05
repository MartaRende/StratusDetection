import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

from prepareData import PrepareData
from data_loader import PrepareDataset
from model import StratusModel


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is : {device}")
print(os.system("whoami"))
print(f"Script UID/GID: {os.getuid()}/{os.getgid()}")

# Set image folder path and views based on script args
FP_IMAGES = "/home/marta/Projects/tb/data/images/mch/1159"
num_views = 1
seq_len = 3  # Number of timesteps
prediction_minutes = 60  # Prediction time in minutes
if len(sys.argv) > 1:
    if sys.argv[1] == "1":
        print("Train on chacha")
        FP_IMAGES = "/home/marta.rende/local_photocast/photocastv1_5/data/images/mch/1159"
        FP_IMAGES = os.path.normpath(FP_IMAGES)
    if len(sys.argv) > 2:
        num_views = int(sys.argv[2])
    if len(sys.argv) > 3:
        seq_len = int(sys.argv[3])

if not os.path.exists(FP_IMAGES):
    print(f"Path {FP_IMAGES} does not exist. Please check the path.")
else:
    print(f"Path {FP_IMAGES} exists.")

print("FP_IMAGES:", FP_IMAGES)
FP_WEATHER_DATA = "data/complete_data_gen.npz"

# Initialize data loader
prepare_data = PrepareData(FP_IMAGES, FP_WEATHER_DATA, num_views=num_views,seq_length=seq_len, prediction_minutes=prediction_minutes)

# Load filtered data
x_meteo, x_images, y = prepare_data.load_data()
print("Data after filter:", x_meteo.shape, y.shape)

# Concatenate all data if multiple sources (your code suggests potential multiple)
all_weatherX = x_meteo
all_imagesX = x_images
allY = y

# Initial split into train/test sets
weather_train, images_train, y_train, weather_test, images_test, y_test, train_datetimes, test_datetimes = prepare_data.split_data(
    all_weatherX, all_imagesX, allY
)



# Further split train into train/validation sets
weather_train, images_train, y_train, weather_validation, images_validation, y_validation, train_datetimes, val_datetimes = prepare_data.split_train_validation(
    weather_train, images_train, y_train
)

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

# Normalize input data
weather_train, weather_validation, weather_test, stats_input = prepare_data.normalize_data(
    weather_train, weather_validation, weather_test,
    var_order=var_order)

# Normalize labels
y_train, y_validation, y_test, stats_label = prepare_data.normalize_data(
    y_train, y_validation, y_test,
    var_order=["gre000z0_nyon", "gre000z0_dole"]
)

train_dataset = PrepareDataset(weather_train, FP_IMAGES, train_datetimes, y_train, num_views=num_views, seq_len=seq_len, data_augmentation=False)
validation_dataset = PrepareDataset(weather_validation, FP_IMAGES, val_datetimes, y_validation, num_views=num_views, seq_len=seq_len)
test_dataset = PrepareDataset(weather_test, FP_IMAGES, test_datetimes, y_test, num_views=num_views, seq_len=seq_len)
print("train_dataset size:", len(train_dataset))
print("validation_dataset size:", len(validation_dataset))
print("test_dataset size:", len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=8)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=8)
# Instantiate model, loss, optimizer, scheduler
model = StratusModel(input_feature_size=15, output_size=2, num_views=num_views, seq_len=seq_len).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

losses = {"train": [], "eval": [], "test": []}
accuracies = {"train": [], "eval": [], "test": []}  # Placeholder if accuracy metrics added

# Training loop
num_epochs = 100  # Increase as needed

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    for step in ["train", "eval", "test"]:
        if step == "train":
            model.train()
            loader = train_loader
        elif step == "eval":
            model.eval()
            loader = validation_loader
        else:
            model.eval()
            loader = test_loader

        total_loss = 0.0
        count = 0

        for batch in loader:
        
            optimizer.zero_grad()
            if num_views == 2:
                weather_x, img1, img2, labels = batch
                weather_x, img1, img2, labels = weather_x.to(device), img1.to(device), img2.to(device), labels.to(device)
                y_pred = model(weather_x, img1, img2)
            else:
             
                weather_x, images_x, labels = batch
                weather_x, images_x, labels = weather_x.to(device), images_x.to(device), labels.to(device)
                y_pred = model(weather_x, images_x)

            # Check for NaNs or Infs in inputs or labels
            if torch.isnan(weather_x).any() or torch.isnan(labels).any() or \
               (num_views == 2 and (torch.isnan(img1).any() or torch.isnan(img2).any())) or \
               torch.isinf(weather_x).any() or torch.isinf(labels).any() or \
               (num_views == 2 and (torch.isinf(img1).any() or torch.isinf(img2).any())):
                print("Warning: NaN or Inf values detected in input data!")

            batch_loss = loss_fn(y_pred, labels)

            if step == "train":
                batch_loss.backward()
                optimizer.step()

            total_loss += batch_loss.item()
            count += 1

        avg_loss = total_loss / count
        losses[step].append(avg_loss)

        if step == "eval":
            scheduler.step(avg_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {losses['train'][-1]:.4f}, "
          f"Validation Loss: {losses['eval'][-1]:.4f}, Test Loss: {losses['test'][-1]:.4f}")


#
MODEL_BASE_PATH = "./models/"


def saveResults():
    currIndex = 0
    currPath: str
    if not os.path.exists(MODEL_BASE_PATH):
        os.mkdir(MODEL_BASE_PATH)
    while True:
        currPath = MODEL_BASE_PATH + f"model_{currIndex}"
        if not os.path.exists(currPath):
            break
        currIndex += 1

    os.mkdir(currPath)
    torch.save(model.state_dict(), currPath + "/model.pth")
    print("Saved model to ", currPath)

    # Plot global loss (all epochs, log scale)
    plt.figure()
    for key in losses:
        if key == "test":
            continue
        plt.plot(losses[key], label=f"{key.capitalize()} loss")
    plt.yscale("log")
    plt.title("Loss (all epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(currPath + "/loss_log_all.png")
    plt.clf()

    # Plot first 15 epochs (log scale)
    plt.figure()
    for key in losses:
        if key == "test":
            continue
        plt.plot(range(min(15, len(losses[key]))), losses[key][:15], label=f"{key.capitalize()} loss")
    plt.yscale("log")
    plt.title("Loss (first 15 epochs)")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(currPath + "/loss_log_first15.png")
    plt.clf()

    # Plot from epoch 15 to the end (log scale)
    plt.figure()
    for key in losses:
        if key == "test":
            continue
        if len(losses[key]) > 15:
            plt.plot(range(15, len(losses[key])), losses[key][15:], label=f"{key.capitalize()} loss")

    plt.title("Loss (epoch 15 to end)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(currPath + "/loss_log_after15.png")
    plt.clf()

    for key in accuracies:
        plt.plot(accuracies[key], label=f"{key.capitalize()} accuracy")
    plt.ylim((0, 1))
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(currPath + "/accuracy.png")

    # save existing file model.py in the same folder
    with open(currPath + "/model.py", "w") as f:
        f.write("# Path: model.py\n")
        with open("model.py", "r") as original_file:
            for line in original_file:
                f.write(line)
        
    # save test data taken from test_dataset
    test_save_path = os.path.join(currPath, "test_data.npz")
    np.savez(test_save_path, dole=prepare_data.test_data)
    # save the stats
    stats_save_path = os.path.join(currPath, "stats.npz")
    np.savez(stats_save_path, stats_input=stats_input, stats_label=stats_label)
    # Save a tuple using numpy
    stratus_days_stats = prepare_data.stats_stratus_days
    print("Stratus days stats:", stratus_days_stats)
    np.savez(os.path.join(currPath, "stratus_days_stats.npz"), stratus_days_stats=stratus_days_stats)
    print("All data saved to", currPath)

    


saveResults()