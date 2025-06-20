import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import StratusModel
from prepareData import PrepareData
from metrics import Metrics
from PIL import Image
from prepare_data.data_augmentation import random_flip, random_rotate, random_brightness, random_contrast, random_color_jitter, random_blur
import pandas as pd
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is : {device}")

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

# =================== NEW: Create or load preprocessed npz file ====================
PREPARED_DATA_PATH = "data/prepared_dataset.npz"
if not os.path.exists(PREPARED_DATA_PATH):
    print("Prepared dataset not found. Creating...")
    prepare_data = PrepareData(FP_IMAGES, FP_WEATHER_DATA, num_views=num_views, seq_length=seq_len)
    x_weather, x_images, y = prepare_data.load_data()
    column_names = prepare_data.data.columns.tolist()
    np.savez_compressed(PREPARED_DATA_PATH, x_weather_df=prepare_data.data, x_weather=x_weather,x_images=x_images, y=y, column_names=column_names)
    print("Saved prepared dataset to:", PREPARED_DATA_PATH)
else:
    print("Found prepared dataset:", PREPARED_DATA_PATH)

# ============= Load the npz with lazy loading =================
data_npz = np.load(PREPARED_DATA_PATH, mmap_mode='r', allow_pickle=True)
x_weather = data_npz['x_weather']
x_images = data_npz['x_images']
y = data_npz['y']
data= data_npz['x_weather_df']
column_names =data_npz['column_names'].tolist()

# Initial split into train/test sets
prepare_data = PrepareData(FP_IMAGES, PREPARED_DATA_PATH, num_views=num_views, seq_length=seq_len)

prepare_data.data = pd.DataFrame(data, columns=column_names)

weather_train, images_train, y_train, weather_test, images_test, y_test  = prepare_data.split_data(
    x_weather, x_images, y
)

# Further split train into train/validation sets
weather_train, images_train, y_train, weather_validation, images_validation, y_validation = prepare_data.split_train_validation(
    weather_train, images_train, y_train
)

var_order = []
for i in range(seq_len):
    var_order += [f"{var}_t{i}" for var in ["RR", "TD", "WG", "TT", "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD", "pres"]]

# Normalize input data
weather_train, weather_validation, weather_test, stats_input = prepare_data.normalize_data(
    weather_train, weather_validation, weather_test, var_order=var_order)

# Normalize labels
y_train, y_validation, y_test, stats_label = prepare_data.normalize_data(
    y_train, y_validation, y_test, var_order=["gre000z0_nyon", "gre000z0_dole"])

# ================= Dataset class =====================
class LazyDataset(torch.utils.data.Dataset):
    def __init__(self, weather, images, y, data_augmentation=False):
        self.weather = weather
        self.images = images
        self.y = y
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        weather_x = torch.tensor(self.weather[idx], dtype=torch.float32).view(seq_len, -1)
        y_val = torch.tensor(self.y[idx], dtype=torch.float32)
        
        img_data = self.images[idx]
        
        # Only load image slice when accessed
        img_data = np.array(img_data)
        print(f"Image data shape: {img_data.shape}")
        import ipdb 
        ipdb.set_trace()
        if self.data_augmentation:
            img_data_view1 = img_data[:, 0]  # (seq_len, H, W, C)
            img_data_view2 = img_data[:, 1]  # (seq_len, H, W, C)
            img_data_view1 = [Image.fromarray(img) for img in img_data_view1]
            img_data_view2 = [Image.fromarray(img) for img in img_data_view2]
            img_data_view1 = [random_flip(img) for img in img_data_view1]
            img_data_view1 = [random_rotate(img) for img in img_data_view1]
            img_data_view1 = [random_brightness(img) for img in img_data_view1]
            img_data_view1 = [random_contrast(img) for img in img_data_view1]
            img_data_view1 = [random_color_jitter(img) for img in img_data_view1]
            img_data_view1 = [random_blur(img) for img in img_data_view1]
            img_data_view2 = [random_flip(img) for img in img_data_view2]
            img_data_view2 = [random_rotate(img) for img in img_data_view2]
            img_data_view2 = [random_brightness(img) for img in img_data_view2]
            img_data_view2 = [random_contrast(img) for img in img_data_view2]
            img_data_view2 = [random_color_jitter(img) for img in img_data_view2]
            img_data_view2 = [random_blur(img) for img in img_data_view2]
     
        
        if num_views == 2:
            img1 = torch.tensor(img_data[:, 0], dtype=torch.float32).permute(0, 3, 1, 2)
            img2 = torch.tensor(img_data[:, 1], dtype=torch.float32).permute(0, 3, 1, 2)
            return weather_x, img1, img2, y_val
        else:
            img = torch.tensor(img_data, dtype=torch.float32).permute(0, 3, 1, 2)
            return weather_x, img, y_val

# Create datasets and loaders
train_dataset = LazyDataset(weather_train, images_train, y_train, data_augmentation=False)
validation_dataset = LazyDataset(weather_validation, images_validation, y_validation)
test_dataset = LazyDataset(weather_test, images_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

# Model, loss, optimizer, scheduler
model = StratusModel(input_feature_size=13, output_size=2, num_views=num_views, seq_len=seq_len).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

losses = {"train": [], "val": [], "test": []}

# Training loop
num_epochs = 70
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    for step in ["train", "val", "test"]:
        model.train() if step == "train" else model.eval()
        loader = train_loader if step == "train" else validation_loader if step == "val" else test_loader

        total_loss = 0.0
        count = 0

        for batch in loader:
            optimizer.zero_grad()

            if num_views == 2:
                weather_x, img1, img2, labels = [b.to(device) for b in batch]
                y_pred = model(weather_x, img1, img2)
            else:
                weather_x, images_x, labels = [b.to(device) for b in batch]
                y_pred = model(weather_x, images_x)

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

    print(f"Train Loss: {losses['train'][-1]:.4f}, Val Loss: {losses['val'][-1]:.4f}, Test Loss: {losses['test'][-1]:.4f}")

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