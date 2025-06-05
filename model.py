# Path: model.py
# Path: model.py
# Path: model.py
import torch
import torch.nn as nn

class StratusModel(nn.Module):
    def __init__(self, input_data_size=15, output_size=2):
        super(StratusModel, self).__init__()
        
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 512 x512 --> 256 x256
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 256 x 256 --> 128 x 128
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 128 x 128 --> 64 x 64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 64 x 64 --> 32 x 32
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32 x 32 --> 16 x 16
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 16 x 16 --> 8 x 8  
        )
        self.cnn_output_size = 64 * 8 *8 
        
        # MLP for meteorological data
        self.mlp_meteo = nn.Sequential(
            nn.Linear(input_data_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

        )
        
        # Final classification head
        self.mlp_head = nn.Sequential(
            nn.Linear(self.cnn_output_size + 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_size) 
        )

    def forward(self, meteo_data, image):
        # Process image through CNN
        z_img = self.cnn(image)
        z_img = z_img.view(z_img.size(0), -1)  # Flatten to [batch_size, 2048]
        
        # Process meteo data
        z_meteo = self.mlp_meteo(meteo_data)
        
        # Combine features
        z = torch.cat([z_img, z_meteo], dim=1)
        
        # Final prediction
        return self.mlp_head(z)