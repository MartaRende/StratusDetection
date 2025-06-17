import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from model import StratusModel
from prepareData import PrepareData
from metrics import Metrics

# Configurazioni iniziali (device, paths, etc.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is : {device}")

# Configura i percorsi e parametri
FP_IMAGES = "/home/marta/Projects/tb/data/images/mch/1159"
FP_WEATHER_DATA = "data/complete_data.npz"
num_views = 1
seq_len = 3

if len(sys.argv) > 1:
    if sys.argv[1] == "1":
        FP_IMAGES = "/home/marta.rende/local_photocast/photocastv1_5/data/images/mch/1159"
        FP_IMAGES = os.path.normpath(FP_IMAGES)
    if len(sys.argv) > 2:
        num_views = int(sys.argv[2])
    if len(sys.argv) > 3:
        seq_len = int(sys.argv[3])

# Inizializza il modello, loss, optimizer
model = StratusModel(input_feature_size=15, output_size=2, num_views=num_views, seq_len=seq_len).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

# Definisci il periodo di tempo da processare
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)

# Inizializza le liste per i risultati
losses = {"train": [], "val": [], "test": []}
accuracies = {"train": [], "val": [], "test": []}

# Dataset class (come nel tuo codice originale)
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, weather, images, y):
        self.weather = weather
        self.images = images
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        weather_x = torch.tensor(self.weather[idx], dtype=torch.float32).view(3, -1)
        y_val = torch.tensor(self.y[idx], dtype=torch.float32)
        img_data = self.images[idx]
        
        if num_views == 2:
            img1 = torch.tensor(img_data[:, 0], dtype=torch.float32).permute(0, 3, 1, 2)
            img2 = torch.tensor(img_data[:, 1], dtype=torch.float32).permute(0, 3, 1, 2)
            return weather_x, img1, img2, y_val
        else:
            img = torch.tensor(img_data, dtype=torch.float32).permute(0, 3, 1, 2)
            return weather_x, img, y_val

# Funzione per processare un mese specifico
def process_month(year, month, prepare_data):
    # Calcola l'intervallo di date per il mese
    month_start = datetime(year, month, 1)
    if month == 12:
        month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = datetime(year, month + 1, 1) - timedelta(days=1)
    
    # Carica i dati per questo mese
    x_meteo, x_images, y = prepare_data.load_data(start_date=month_start, end_date=month_end)
    print(f"Data for {month_start.strftime('%Y-%m')}: {x_meteo.shape}, {x_images.shape}, {y.shape}")
    
   
    
    return x_meteo, x_images, y

# Loop principale per il training mese per mese
current_date = start_date
while current_date <= end_date:
    year = current_date.year
    month = current_date.month
    
    print(f"\nProcessing {year}-{month:02d}")
    
    # Inizializza PrepareData per questo mese
    prepare_data = PrepareData(FP_IMAGES, FP_WEATHER_DATA, num_views=num_views, seq_length=seq_len)
    
    # Carica i dati per il mese corrente
    x_meteo, x_images, y = process_month(year, month, prepare_data)
    
    # Split in train/val/test (potresti voler modificare questa logica)
    weather_train, images_train, y_train, weather_test, images_test, y_test = prepare_data.split_data(
        x_meteo, x_images, y
    )
    
    weather_train, images_train, y_train, weather_validation, images_validation, y_validation = prepare_data.split_train_validation(
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
    # Crea i dataset e dataloader
    train_dataset = SimpleDataset(weather_train, images_train, y_train)
    validation_dataset = SimpleDataset(weather_validation, images_validation, y_validation)
    test_dataset = SimpleDataset(weather_test, images_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    # Training loop per questo mese
    num_epochs = 10  # Puoi regolare questo valore
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} - {year}-{month:02d}")
        
        for step, loader in [("train", train_loader), ("val", validation_loader), ("test", test_loader)]:
            if step == "train":
                model.train()
            else:
                model.eval()
            
            total_loss = 0.0
            count = 0
            
            for batch in loader:
                optimizer.zero_grad()
                
                if num_views == 2:
                    weather_x, img1, img2, labels = [x.to(device) for x in batch]
                    y_pred = model(weather_x, img1, img2)
                else:
                    weather_x, images_x, labels = [x.to(device) for x in batch]
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
        
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {losses['train'][-1]:.4f}, "
              f"Validation Loss: {losses['val'][-1]:.4f}, Test Loss: {losses['test'][-1]:.4f}")
    
    # Passa al mese successivo
    if month == 12:
        current_date = datetime(year + 1, 1, 1)
    else:
        current_date = datetime(year, month + 1, 1)


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