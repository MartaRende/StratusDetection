from multiprocessing.pool import ThreadPool
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from model import StratusModel
from prepareData import PrepareData
from metrics import Metrics
from torch.utils.data import Dataset
from PIL import Image

from prepare_data.data_augmentation import random_flip, random_rotate, random_brightness, random_contrast, random_color_jitter, random_blur
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
import os
from datetime import datetime
# Modify your SimpleDataset class to use more efficient loading
class SimpleDataset(Dataset):
    def __init__(self, weather, image_base_folder, seq_infos, labels, num_views=1, seq_len=3, data_augmentation=False):
        self.weather = torch.tensor(weather, dtype=torch.float32)  
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.image_base_folder = image_base_folder
        self.seq_infos = seq_infos
        self.num_views = num_views
        self.seq_len = seq_len
        self.data_augmentation = data_augmentation
        
        self.image_paths = self._precompute_image_paths()
        
        self.cache = {}
        self.cache_size_limit = 1000  
        
    def _load_single_image(self, path):
        """Versione ottimizzata con cache intelligente"""
        if path in self.cache:
            return self.cache[path]
        
        try:
            with Image.open(path) as img:
                if self.data_augmentation:
                    img = random_brightness(img)
                    img = random_contrast(img)
                    img = random_color_jitter(img)
                    img = random_blur(img)
                img = img.crop((0, 0, 512, 200))  # Crop to 512x200
                img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) 
                
                if len(self.cache) < self.cache_size_limit:
                    self.cache[path] = img_tensor
                
                return img_tensor
        except:
            return torch.zeros((3, 512, 512), dtype=torch.float32)
        
    def _precompute_image_paths(self):
        """Precompute all image paths to avoid repeated disk access during training"""
        paths = []
        for seq_info in self.seq_infos:
            view_paths = []
            for view in range(1, self.num_views + 1):
                seq_paths = [self.get_image_path(dt, view) for dt in seq_info]
                view_paths.append(seq_paths)
            paths.append(view_paths if self.num_views > 1 else view_paths[0])
        return paths
    
    def get_image_path(self, dt, view=1):
        """Your existing method for getting image paths"""
        if isinstance(dt, np.datetime64):
            dt = pd.Timestamp(dt)
            
        date_str = dt.strftime('%Y-%m-%d')
        time_str = dt.strftime('%H%M')
        img_filename = f"1159_{view}_{date_str}_{time_str}.jpeg"
        
        return os.path.join(
            self.image_base_folder,
            str(view),
            dt.strftime('%Y'),
            dt.strftime('%m'),
            dt.strftime('%d'),
            img_filename
        )
    
    def __len__(self):
        return len(self.weather)
    
    
    def __getitem__(self, idx):
        """Versione ottimizzata per grandi dataset"""
        weather_data = self.weather[idx]
        labels = self.labels[idx]
        
        if self.num_views == 2:
            view1_paths, view2_paths = self.image_paths[idx]
            
            view1_images = []
            view2_images = []
            for p1, p2 in zip(view1_paths, view2_paths):
                view1_images.append(self._load_single_image(p1))
                view2_images.append(self._load_single_image(p2))
            
            view1_tensor = torch.stack(view1_images)
            view2_tensor = torch.stack(view2_images)
            
            return weather_data, view1_tensor, view2_tensor, labels
        else:
            img_paths = self.image_paths[idx]
            
            images = []
            for p in img_paths:
                img_tensor = self._load_single_image(p)
                images.append(img_tensor)
            
            
            images_tensor = torch.stack(images)
            
            return weather_data, images_tensor, labels

# Create datasets and loaders



train_dataset = SimpleDataset(weather_train, FP_IMAGES, train_datetimes, y_train, num_views=num_views, seq_len=seq_len, data_augmentation=False)
validation_dataset = SimpleDataset(weather_validation, FP_IMAGES, val_datetimes, y_validation, num_views=num_views, seq_len=seq_len)
test_dataset = SimpleDataset(weather_test, FP_IMAGES, test_datetimes, y_test, num_views=num_views, seq_len=seq_len)
print("train_dataset size:", len(train_dataset))
print("validation_dataset size:", len(validation_dataset))
print("test_dataset size:", len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=8)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=8)
# Instantiate model, loss, optimizer, scheduler
model = StratusModel(input_feature_size=15, output_size=2, num_views=num_views, seq_len=seq_len).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

losses = {"train": [], "eval": [], "test": []}
accuracies = {"train": [], "eval": [], "test": []}  # Placeholder if accuracy metrics added

# Training loop
num_epochs = 50  # Increase as needed

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
    plt.title("Loss (all epochs, log scale)")
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
    plt.title("Loss (first 15 epochs, log scale)")
    plt.legend()
    plt.savefig(currPath + "/loss_log_first15.png")
    plt.clf()

    # Plot from epoch 15 to the end (log scale)
    plt.figure()
    for key in losses:
        if key == "test":
            continue
        if len(losses[key]) > 15:
            plt.plot(range(15, len(losses[key])), losses[key][15:], label=f"{key.capitalize()} loss")

    plt.title("Loss (epoch 15 to end, log scale)")
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