# import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class StratusModel(nn.Module):
    def __init__(self, input_data_size=15, output_size = 2):
        super(StratusModel, self).__init__()
        output_mlp_meteo_size = 64
        output_ccn_size = 128 * 64 * 64
        self.cnn = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), 
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # (128,64,64)
   
    )

        self.mlp_meteo = nn.Sequential(
            nn.Linear(input_data_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_mlp_meteo_size),
            nn.ReLU(),
        )
        self.mlp_head = nn.Sequential(
        nn.Linear(output_ccn_size+output_mlp_meteo_size, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 64),  # Predicting rayonnement for Nyon and La Dôle
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, output_size) # Output size is 2 for Nyon and La Dôle
        
    )

    def forward(self, meteo_data, image):
        z_img = self.cnn(image)       
        z_meteo = self.mlp_meteo(meteo_data)  
        z_img = z_img.view(z_img.size(0), -1)
        z = torch.cat([z_img, z_meteo], dim=1) 
        out = self.mlp_head(z)  # [B, 2]
        return out
