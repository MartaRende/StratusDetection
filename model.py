# Path: model.py
# Path: model.py
# Path: model.py
# Path: model.py
import torch
import torch.nn as nn

class StratusModel(nn.Module):
    def __init__(self, input_data_size=16, output_size=2, num_views=1):
        super(StratusModel, self).__init__()
        
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 512 x512 --> 256 x256
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 256 x 256 --> 128 x 128
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 128 x 128 --> 64 x 64
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 64 x 64 --> 32 x 32
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32 x 32 --> 16 x 16
            # nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),   # 16 x 16 --> 8 x 8  
        )
        self.cnn_output_size = 32 * 16 * 16  # 32 channels * 8x8 spatial = 2048
        self.num_views = num_views
        # # MLP for meteorological data
        self.mlp_meteo = nn.Sequential(
            nn.Linear(input_data_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

        )
        factor = 1 if num_views == 1 else 2
        
        # Final classification head
        self.mlp_head = nn.Sequential(
            nn.Linear(self.cnn_output_size * factor, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size) 
        )

    def forward(self, meteo_data, image_1=None, image_2=None):
        if self.num_views == 2:
            # If two views are provided, process both images
            image_1 = self.cnn(image_1)
            image_1 = image_1.view(image_1.size(0), -1)
            image_2 = self.cnn(image_2)
            image_2 = image_2.view(image_2.size(0), -1)
            # Combine features from both images
            z_img = torch.cat([image_1, image_2], dim=1)
        else:
        # Process image through CNN
            z_img = self.cnn(image_1)
            z_img = z_img.view(z_img.size(0), -1)  # Flatten to [batch_size, 2048]
        
        # Process meteo data
        z_meteo = self.mlp_meteo(meteo_data)
        
        z = torch.cat([z_img, z_meteo], dim=1)
        
        # Final prediction
        return self.mlp_head(z_meteo)