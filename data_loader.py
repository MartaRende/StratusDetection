
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os
from data_tools.data_augmentation import random_brightness, random_color_jitter, random_blur
# This dataset class is designed to prepare the dataset for training by loading images and weather data.
# It supports both single-view and dual-view configurations, applies data augmentation if specified,
# and precomputes image paths to optimize loading during training.
# The dataset returns weather data, image tensors, and labels for each sample.
# It handles missing images by returning a blank tensor, ensuring robustness during training.
class PrepareDataset(Dataset):
    def __init__(self, weather, image_base_folder, seq_infos, labels, num_views=1, seq_len=3, data_augmentation=False,prepare_data=None):
        self.weather = torch.tensor(weather, dtype=torch.float32)  
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.image_base_folder = image_base_folder
        self.seq_infos = seq_infos
        self.num_views = num_views
        self.seq_len = seq_len
        self.data_augmentation = data_augmentation
        self.prepare_data = prepare_data
        
        # Precompute image paths
        self.image_paths = self._precompute_image_paths()

    def _precompute_image_paths(self):
        """Precompute all image paths to avoid repeated disk access during training."""
        paths = []
        for seq_info in self.seq_infos:
            view_paths = []
            for view in range(1, self.num_views + 1):
                seq_paths = [self.get_image_path(dt, 2) for dt in seq_info]
                view_paths.append(seq_paths)
            paths.append(view_paths if self.num_views > 1 else view_paths[0])
        return paths

    def get_image_path(self, dt, view=2):
        """Generate the image path based on the datetime and view."""
        if isinstance(dt, np.datetime64):
            dt = pd.Timestamp(dt)
            
        date_str = dt.strftime('%Y-%m-%d')
        time_str = dt.strftime('%H%M')
        img_filename = f"1159_{view}_{date_str}_{time_str}.jpeg"
        
        return os.path.join(
            self.image_base_folder,
            str(view),
            dt.strftime('%Y'),
            dt.strftime('%m'),
            dt.strftime('%d'),
            img_filename
        )
    
    def __len__(self):
        return len(self.weather)
    def _load_single_image(self, path):
        try:
            with Image.open(path) as img:
                # img = img.crop((0, 0, 512, 200))  
                img_array = np.array(img) 
                img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
                return img_tensor
        except:
            print(f"Warning: Could not load image at {path}. Returning blank tensor.")
            return torch.zeros((3, 512, 512), dtype=torch.float32)  # Return a blank tensor for missing images
        
    def __getitem__(self, idx):
        weather_data = self.weather[idx]
        labels = self.labels[idx]
        
        if self.num_views == 2:
            view1_paths, view2_paths = self.image_paths[idx]
            
            view1_images = []
            view2_images = []
            for p1, p2 in zip(view1_paths, view2_paths):
                view1_images.append(self._load_single_image(p1))
                view2_images.append(self._load_single_image(p2))
            
            view1_tensor = torch.stack(view1_images)  # Shape: (seq_len, C, H, W)
            view2_tensor = torch.stack(view2_images)
            
            return weather_data, view1_tensor, view2_tensor, labels
        else:
            img_paths = self.image_paths[idx]
            
            images = []
            for p in img_paths:
                img_tensor = self._load_single_image(p)
                images.append(img_tensor)
           
            images_tensor = torch.stack(images)  # Shape: (seq_len, C, H, W)
            
            return weather_data, images_tensor, labels
