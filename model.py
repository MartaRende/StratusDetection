import torch
import torch.nn as nn

class StratusModel(nn.Module):
    def __init__(self, input_feature_size=15, output_size=2, num_views=1, seq_len=3):
        super(StratusModel, self).__init__()
        self.seq_len = seq_len
        self.num_views = num_views
        self.input_feature_size = input_feature_size  # Should be 15 (features per timestep)

        # CNN for image processing (unchanged)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # ... [rest of your CNN layers] ...
        )
        self.cnn_output_size = 32 * 16 * 16  # = 8192 per image

        # MLP for meteorological data (modified input size)
        self.mlp_meteo = nn.Sequential(
            nn.Linear(input_feature_size, 64),  # Now takes 15 features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Compute total feature size
        img_total_dim = self.cnn_output_size * seq_len * (2 if num_views == 2 else 1)
        meteo_total_dim = 128 * seq_len
        mlp_input_size = img_total_dim + meteo_total_dim

        # Final MLP head (unchanged)
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_size, 2048),
            # ... [rest of your head layers] ...
            nn.Linear(256, output_size)
        )
    def forward(self, meteo_seq, image_seq_1=None, image_seq_2=None):
        batch_size = meteo_seq.size(0)
        img_features = []
        meteo_features = []

        for t in range(self.seq_len):
            img1_t = image_seq_1[:, t]
            img1_feat = self.cnn(img1_t).reshape(batch_size, -1)

            if self.num_views == 2:
                img2_t   = image_seq_2[:, t]
                img2_feat= self.cnn(img2_t).reshape(batch_size, -1)
                img_features.append(torch.cat([img1_feat, img2_feat], dim=1))
            else:
                img_features.append(img1_feat)

            meteo_t = meteo_seq[:, t]
            meteo_features.append(self.mlp_meteo(meteo_t))

        z_img   = torch.cat(img_features,   dim=1)
        z_meteo = torch.cat(meteo_features, dim=1)
        z       = torch.cat([z_img, z_meteo], dim=1)

        # DEBUG: print what size you’re actually feeding into the head
        print(f"→ z_img: {z_img.shape}, z_meteo: {z_meteo.shape}, concatenated z: {z.shape}")

        return self.mlp_head(z)
