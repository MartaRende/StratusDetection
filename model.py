# Path: model.py
# Path: model.py
# Path: model.py
# Path: model.py
# Path: model.py
import torch
import torch.nn as nn

class StratusModel(nn.Module):
    def __init__(self, input_feature_size=15, output_size=2, num_views=1, seq_len=3):
        super(StratusModel, self).__init__()
        self.seq_len = seq_len
        self.num_views = num_views
        self.input_feature_size = input_feature_size

        # CNN 
        self.cnn_view1 = nn.Sequential(
            nn.Conv2d(3 * seq_len, 32, kernel_size=3, stride=1, padding=1),  # Input channels = 3*seq_len
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 512x512 -> 256x256
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256x256 -> 128x128
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
        )


        self.cnn_output_size = 32 * 16 * 16  # 8192

        # # MLP for weather data
        self.mlp_meteo = nn.Sequential(
            nn.Linear(input_feature_size * seq_len, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        img_total_dim = self.cnn_output_size * (2 if num_views == 2 else 1)
        meteo_total_dim = 128
        mlp_input_size = img_total_dim + meteo_total_dim

        # MLP final
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
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
            nn.Linear(64, output_size)
        )


    def forward(self, meteo_seq, image_seq_1, image_seq_2=None):
        batch_size = image_seq_1.size(0)
        seq_len = image_seq_1.size(1)  

        view1_input = image_seq_1.reshape(batch_size, seq_len * 3, image_seq_1.size(3), image_seq_1.size(4))

        view1_features = self.cnn_view1(view1_input).reshape(batch_size, -1)
        
        if self.num_views == 2 and image_seq_2 is not None:
            view2_input = image_seq_2.reshape(batch_size, -1, image_seq_2.size(3), image_seq_2.size(4))
            view2_features = self.cnn_view1(view2_input).reshape(batch_size, -1)
            img_features = torch.cat([view1_features, view2_features], dim=1)
        else:
            img_features = view1_features
        
        meteo_flat = meteo_seq.reshape(batch_size, -1)
        z_meteo = self.mlp_meteo(meteo_flat)
        
        z = torch.cat([img_features, z_meteo], dim=1)
        output = self.mlp_head(img_features)

        return output