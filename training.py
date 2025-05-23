from prepareData import PrepareData
import torch    
import os
import numpy as np
import glob
from PIL import Image
from multiprocessing import Pool

from model import StratusModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is : {device}")

files = glob.glob("data/complete_data.npz")


all_weatherX = []
all_imagesX = []
allY = []
prepare_data = PrepareData()
for file in files:
    x_meteo, x_images, y = prepare_data.load_data(file)
  
    all_weatherX.append(x_meteo)
    all_imagesX.append(x_images)
    allY.append(y)

all_weatherX = np.concatenate(all_weatherX)
all_imagesX = np.concatenate(all_imagesX)
allY = np.concatenate(allY)

print("Data after filter:", all_weatherX.shape, all_imagesX.shape, allY.shape)

weather_train, weather_test, images_train, images_test, y_train, y_test = prepare_data.split_data(
    all_weatherX, all_imagesX, allY
)
import ipdb
ipdb.set_trace()
# normalize the data
weather_train, _ = prepare_data.normalize_data(weather_train, var_order=["gre000z0_nyon", "gre000z0_dole","RR", "TD", "WG", "TT", "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD"])
weather_test, _ = prepare_data.normalize_data(weather_test, var_order=[ "gre000z0_nyon", "gre000z0_dole","RR", "TD", "WG", "TT", "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD"])
# normalize labels
y_train,_ = prepare_data.normalize_data(y_train, var_order=["gre000z0_nyon", "gre000z0_dole"])  
y_test, _= prepare_data.normalize_data(y_test, var_order=["gre000z0_nyon", "gre000z0_dole"])

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, weather, images, y):
        self.weather = weather
        self.images = images
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        if img.ndim == 3:
            img = img.permute(2, 0, 1)
        return (
            torch.tensor(self.weather[idx], dtype=torch.float32),
            img,
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

train_dataset = SimpleDataset(weather_train, images_train, y_train)
test_dataset = SimpleDataset(weather_test, images_test, y_test)
print("train_dataset", len(train_dataset))
print("test_dataset", len(test_dataset))

# Split train/validation
train_size = int(0.8 * len(train_dataset))
validation_size = len(train_dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# Model creation

model = StratusModel(input_data_size=15, output_size=2).to(device)
# Loss function and optimizer
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3
)
loss.to(device)
model.to(device)

losses = {
    "train": [],
    "eval": [],
    "test": []
}
accuracies = {
    "train": [],
    "eval": [],
    "test": []
}

patience = 3  
best_val_loss = float('inf')
epochs_no_improve = 0


num_epochs = 20
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for step in ["train", "eval", "test"]:
        if step == "train": 
            currLoader = train_loader 
        elif step == "eval":
            currLoader = validation_loader
        else: 
            currLoader = test_loader
        if step == "train":
            model.train()
        else:
            model.eval()
        curr_loss = 0
        nbr_items = 0
        for weather_x, images_x, labels in currLoader:
            weather_x, images_x, labels = weather_x.to(device), images_x.to(device), labels.to(device)
            if torch.isnan(weather_x).any() or torch.isnan(images_x).any() or torch.isnan(labels).any():
                print("NaN in input data!")
            if torch.isinf(weather_x).any() or torch.isinf(images_x).any() or torch.isinf(labels).any():
                print("Inf in input data")

            optimizer.zero_grad()
            y_pred = model(weather_x, images_x)
            class_loss = loss(y_pred, labels)
            if step == "train":
                class_loss.backward()
                optimizer.step()
            curr_loss += class_loss.item()
            nbr_items += 1

        avg_loss = curr_loss / nbr_items
        losses[step].append(avg_loss)

        if step == "eval":
            scheduler.step(avg_loss)
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()  
            else:
                epochs_no_improve += 1

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses['train'][-1]:.4f}, Val loss : {losses['eval'][-1]:.4f}, Test loss : {losses['test'][-1]:.4f}"
    )
    
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

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
        plt.plot(losses[key], label=f"{key.capitalize()} loss")
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


saveResults()