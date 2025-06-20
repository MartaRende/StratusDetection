import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import StratusModel
from prepareData import PrepareData
from metrics import Metrics
from PIL import Image
from prepare_data import random_flip, random_rotate, random_brightness, random_contrast, random_color_jitter, random_blur
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is : {device}")
print(os.system("whoami"))
print(f"Script UID/GID: {os.getuid()}/{os.getgid()}")

# Set image folder path and views based on script args
FP_IMAGES = "/home/marta/Projects/tb/data/images/mch/1159"
num_views = 1
seq_len = 3  # Number of timesteps
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
FP_WEATHER_DATA = "data/complete_data.npz"

# Initialize data loader
prepare_data = PrepareData(FP_IMAGES, FP_WEATHER_DATA, num_views=num_views,seq_length=seq_len)

# Load filtered data
x_meteo, x_images, y = prepare_data.load_data()
print("Data after filter:", x_meteo.shape, x_images.shape, y.shape)

# Concatenate all data if multiple sources (your code suggests potential multiple)
all_weatherX = x_meteo
all_imagesX = x_images
allY = y

# Initial split into train/test sets
weather_train, images_train, y_train, weather_test, images_test, y_test  = prepare_data.split_data(
    all_weatherX, all_imagesX, allY
)

# Further split train into train/validation sets
weather_train, images_train, y_train, weather_validation, images_validation, y_validation = prepare_data.split_train_validation(
    weather_train, images_train, y_train
)
var_order = []
for i in range(seq_len):
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

# Dataset class
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, weather, images, y, data_augmentation=False):
        self.weather = weather
        self.images = images
        self.y = y
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        weather_x = torch.tensor(self.weather[idx], dtype=torch.float32).view(3, -1)

        y_val = torch.tensor(self.y[idx], dtype=torch.float32)
        
        # Handle images based on num_views
        img_data = self.images[idx]  # Shape: (3, 2, 512, 512, 3)
        if self.data_augmentation:
                # Apply data augmentation to each image in the sequence
                img_data = np.array([random_flip(Image.fromarray(img)) for img in img_data])
                img_data = np.array([random_rotate(Image.fromarray(img)) for img in img_data])
                img_data = np.array([random_brightness(Image.fromarray(img)) for img in img_data])
                img_data = np.array([random_contrast(Image.fromarray(img)) for img in img_data])
                img_data = np.array([random_color_jitter(Image.fromarray(img)) for img in img_data])
                img_data = np.array([random_blur(Image.fromarray(img)) for img in img_data])
        if num_views == 2:
           
            # For 2 views, select both cameras for all timesteps
            img1 = torch.tensor(img_data[:, 0], dtype=torch.float32).permute(0, 3, 1, 2)  # (3, 512, 512, 3) -> (3, 3, 512, 512)
            img2 = torch.tensor(img_data[:, 1], dtype=torch.float32).permute(0, 3, 1, 2)
            
            return weather_x, img1, img2, y_val
        
        else:
        
            # For single view, just take first camera
            img = torch.tensor(img_data, dtype=torch.float32).permute(0, 3, 1, 2)
            return weather_x, img, y_val

# Create datasets and loaders
train_dataset = SimpleDataset(weather_train, images_train, y_train, data_augmentation=True)
validation_dataset = SimpleDataset(weather_validation, images_validation, y_validation)
test_dataset = SimpleDataset(weather_test, images_test, y_test)
print("train_dataset size:", len(train_dataset))
print("validation_dataset size:", len(validation_dataset))
print("test_dataset size:", len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
# Instantiate model, loss, optimizer, scheduler
model = StratusModel(input_feature_size=13, output_size=2, num_views=num_views, seq_len=seq_len).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

losses = {"train": [], "val": [], "test": []}
accuracies = {"train": [], "val": [], "test": []}  # Placeholder if accuracy metrics added

# Training loop
num_epochs = 70  # Increase as needed

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    for step in ["train", "val", "test"]:
        if step == "train":
            model.train()
            loader = train_loader
        elif step == "val":
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

        if step == "val":
            scheduler.step(avg_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {losses['train'][-1]:.4f}, "
          f"Validation Loss: {losses['val'][-1]:.4f}, Test Loss: {losses['test'][-1]:.4f}")


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
    plt.figure()
    for key in losses:
        if key == "test":
            continue
        plt.plot(losses[key], label=f"{key.capitalize()} loss")
    # Plot log scale for first 15 epochs, then linear for the rest
    if len(losses["train"]) > 15:
        plt.yscale("log")
        plt.xlim(0, 15)
        plt.legend()
        plt.title("Loss (log scale, first 15 epochs)")
        plt.savefig(currPath + "/loss_log_first15.png")
        plt.clf()
        for key in losses:
            if key == "test":
                continue
            plt.plot(range(15, len(losses[key])), losses[key][15:], label=f"{key.capitalize()} loss")
        plt.yscale("linear")
        plt.title("Loss (linear scale, after epoch 15)")
        plt.legend()
        plt.savefig(currPath + "/loss_linear_after15.png")

    plt.yscale("log")
    plt.legend()
    plt.title("Loss")
    plt.savefig(currPath + "/loss.png")
    plt.figure()
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
    
    # model.eval()
    # with torch.no_grad():
    #     def predict(weather_np, images_np):
    #         x_weather = torch.tensor(weather_np, dtype=torch.float32, device=device)
    #         if num_views == 2:
    #             img1 = torch.tensor(images_np[0], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    #             img2 = torch.tensor(images_np[1], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    #             return model(x_weather, img1, img2).cpu().numpy()
    #         else:
    #             x_images = torch.tensor(images_np, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    #             return model(x_weather, x_images).cpu().numpy()

    #     # Train metrics
    #     preds_train = predict(weather_train, images_train)
    #     train_metrics = Metrics(
    #         y_train.tolist(),
    #         preds_train,
    #         data=None,
    #         save_path=f"{MODEL_BASE_PATH}/train_stats",
    #         start_date="2023-01-01",
    #         end_date="2024-12-31",
    #         stats_for_month=False
    #     )
    #     train_metrics.save_metrics_report(stratus_days=train_stratus_days, non_stratus_days=train_non_stratus_days)

    #     # Validation metrics
    #     preds_val = predict(weather_validation, images_validation)
    #     val_metrics = Metrics(
    #         y_validation.tolist(),
    #         preds_val,
    #         data=None,
    #         save_path=f"{MODEL_BASE_PATH}/validation_stats",
    #         start_date="2023-01-01",
    #         end_date="2024-12-31",
    #         stats_for_month=False
    #     )
    #     val_metrics.save_metrics_report(stratus_days=validation_stratus_days, non_stratus_days=validation_non_stratus_days)

    #     # Test metrics
    #     preds_test = predict(weather_test, images_test)
    #     test_metrics = Metrics(
    #         y_test.tolist(),
    #         preds_test,
    #         data=None,
    #         save_path=f"{MODEL_BASE_PATH}/test_stats",
    #         start_date="2023-01-01",
    #         end_date="2024-12-31",
    #         stats_for_month=False
    #     )
    #     test_metrics.save_metrics_report(stratus_days=test_stratus_days, non_stratus_days=test_non_stratus_days)

    


saveResults()