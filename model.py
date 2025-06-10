import torch
import torch.nn as nn

class StratusModel(nn.Module):
    def __init__(self, input_data_size=16, output_size=2, num_views=1, seq_len=3):
        super(StratusModel, self).__init__()

        self.seq_len = seq_len
        self.num_views = num_views

        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 512 → 256
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 → 128
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 → 64
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 → 32
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 → 16
        )
        self.cnn_output_size = 32 * 16 * 16  # = 8192 per image

        # MLP for meteorological data
        self.mlp_meteo = nn.Sequential(
            nn.Linear(input_data_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Compute total feature size for mlp_head input
        img_total_dim = self.cnn_output_size * seq_len * (2 if num_views == 2 else 1)
        meteo_total_dim = 128 * seq_len
        mlp_input_size = img_total_dim + meteo_total_dim

        # Final MLP head for classification
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_size, 2048),
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
    def forward(self, meteo_seq, image_seq_1=None, image_seq_2=None):
        """
        Inputs:
            - meteo_seq: Tensor [B, seq_len, input_data_size]
            - image_seq_1: Tensor [B, seq_len, 3, H, W]
            - image_seq_2 (optional): Tensor [B, seq_len, 3, H, W] if num_views == 2
        """

        batch_size = meteo_seq.size(0)
        img_features = []
        meteo_features = []

        # Ensure input_data_size matches the last dimension of meteo_seq
        if meteo_seq.size(-1) != self.mlp_meteo[0].in_features:
            raise ValueError(f"Expected meteo_seq last dimension to be {self.mlp_meteo[0].in_features}, but got {meteo_seq.size(-1)}")

        for t in range(self.seq_len):
            # CNN features
            img1_t = image_seq_1[:, t]  # Shape: [B, 3, H, W]
            img1_feat = self.cnn(img1_t).reshape(batch_size, -1)

            if self.num_views == 2:
                img2_t = image_seq_2[:, t]
                img2_feat = self.cnn(img2_t).reshape(batch_size, -1)
                img_feat = torch.cat([img1_feat, img2_feat], dim=1)
            else:
                img_feat = img1_feat

            img_features.append(img_feat)

            # MLP features
            meteo_t = meteo_seq[:, t]
            meteo_feat = self.mlp_meteo(meteo_t)
            meteo_features.append(meteo_feat)

        # Concatenate all time steps
        z_img = torch.cat(img_features, dim=1)        # [B, seq_len * cnn_feat_size]
        z_meteo = torch.cat(meteo_features, dim=1)    # [B, seq_len * 128]
        z = torch.cat([z_img, z_meteo], dim=1)

        return self.mlp_head(z)
