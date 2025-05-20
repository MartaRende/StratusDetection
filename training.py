from prepareData import PrepareData
import torch    
import os
import numpy as np
import glob
from PIL import Image
from multiprocessing import Pool

from model import StratusModel
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is : {device}")

train_files = glob.glob("data/train/*.npz")
test_files = glob.glob("data/test/*.npz")

from torch.utils.data import Dataset
import numpy as np

class MeteoImageDataset(Dataset):
    def __init__(self, file_list, prepare_data):
        self.file_list = file_list
        self.prepare_data = prepare_data
        self.index_map = []
        self.file_data = []
        # Precompute index mapping: (file_idx, sample_idx)
        for file_idx, file in enumerate(file_list):
            x_meteo, x_images, y = prepare_data.load_data(file)
    
            self.file_data.append((x_meteo, x_images, y))
            for sample_idx in range(len(y)):
                self.index_map.append((file_idx, sample_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        x_meteo, x_images, y = self.file_data[file_idx]
        # print("x_meteo", x_meteo.shape)
        # # print("x_images", x_images.shape)
        # print("y", y.shape)
        
        img = torch.tensor(x_images[sample_idx], dtype=torch.float32)
        if img.ndim == 3:
            img = img.permute(2, 0, 1)  # HWC -> CHW
        
        # Ensure y[sample_idx] is shape (2,)
        label = y[sample_idx]
        label = torch.tensor(label, dtype=torch.float32)
        if label.ndim == 0:
            raise ValueError(f"Label for sample {sample_idx} is scalar, expected 2-element vector")

        return (
            torch.tensor(x_meteo[sample_idx], dtype=torch.float32),
            img,
            label
    )

prepare_data = PrepareData()
# for file_idx, file in enumerate(train_files):
#     x_meteo, x_images, y = prepare_data.load_data(file)
#     print(f"{file}: {len(y)} samples")
train_weatherX = []
train_imagesX = []
trainY = []
test_source_weather_x = []
test_imagesX = []
testY = []
# give train file randomly from


# # Load training data sequentially
# for file in train_files:
#     x_weather, x_img, y = prepare_data.load_data(file)
#     train_weatherX.append(x_weather)
#     train_imagesX.append(x_img)
#     trainY.append(y)
#     print("here")

# train_weatherX = np.concatenate(train_weatherX)
# print("Training data loaded")

# train_imagesX = np.concatenate(train_imagesX)

# trainY = np.concatenate(trainY)
train_dataset = MeteoImageDataset(train_files, prepare_data)
test_dataset = MeteoImageDataset(test_files, prepare_data)
print("train_dataset", len(train_dataset))
print("test_dataset", len(test_dataset))
if len(train_dataset) == 0:
    raise ValueError("Training dataset is empty")
if len(test_dataset) == 0:
    raise ValueError("Testing dataset is empty")
print("Training data loaded")
for i in range(5):
    _, _, label = train_dataset[i]
    print("Sample label shape:", label.shape)

# # Process testing data in parallel
# with Pool() as pool:
#     results = pool.map(prepare_data.load_data, test_files)
#     test_source_weather_x = np.concatenate([x for x,_, _ in results])
#     test_imagesX = np.concatenate([x for _, x, _ in results])
#     testY = np.concatenate([y for _, _, y in results])

# Tensor creation
# print("Tensor creation started")
# train_source_weather_tensor = torch.tensor(train_weatherX, dtype=torch.float32)
# train_source_images_tensor = torch.tensor(train_imagesX, dtype=torch.float32)
# train_target_tesor = torch.tensor(trainY, dtype=torch.float32)
# test_source_weather_x = torch.tensor(test_source_weather_x, dtype=torch.float32)
# test_source_images_tensor = torch.tensor(test_imagesX, dtype=torch.float32)
# test_target_sensor = torch.tensor(testY, dtype=torch.float32)
# # Dataset creation
# train_dataset = torch.utils.data.TensorDataset(train_source_weather_tensor, train_source_images_tensor, train_target_tesor)
# test_dataset = torch.utils.data.TensorDataset(test_source_weather_x, test_source_images_tensor, test_target_sensor)

print("Tensor creation done")


# decide size of training and testing data
train_size = int(0.9 * len(train_dataset))
validation_size = len(train_dataset) - train_size

train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

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

# Training loop
num_epochs = 5
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
        if step == "eval":
            scheduler.step(curr_loss)
        losses[step].append(curr_loss / nbr_items)
        # Puoi aggiungere qui una metrica di accuratezza se il tuo problema è classificazione

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {class_loss.item():.4f}, Val loss : {losses['eval'][-1]:.4f}, Test loss : {losses['test'][-1]:.4f}"
    )

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


saveResults()