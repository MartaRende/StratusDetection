
import torch
import torch.nn as nn

class StratusModel(nn.Module):
    def __init__(self, input_data_size=16, output_size=2):
        super(StratusModel, self).__init__()
        
        # CNN for image processing
     # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 512 x512 --> 256 x256
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 256 x 256 --> 128 x 128
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 128 x 128 --> 64 x 64
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 64 x 64 --> 32 x 32
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32 x 32 --> 16 x 16
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 16 x 16 --> 8 x 8  
        )
        self.cnn_output_size = 32 * 8 * 8  # 32 channels and 16x16 feature map size
        
        # MLP for meteorological data
        self.mlp_meteo = nn.Sequential(
            nn.Linear(input_data_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

        )
        
        # Final classification head
        self.mlp_head = nn.Sequential(
            nn.Linear(self.cnn_output_size * 2 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )


    def forward(self, weather_x, image1_x, image2_x):
        x1 = self.cnn(image1_x)
        x2 = self.cnn(image2_x)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        cnn_out = torch.cat((x1, x2), dim=1)
        meteo_out = self.mlp_meteo(weather_x)
        combined = torch.cat((cnn_out, meteo_out), dim=1)
        return self.mlp_head(combined)
