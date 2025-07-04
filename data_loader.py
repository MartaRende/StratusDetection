
import torch
from torch.utils.data import Dataset
from data_tools.data_augmentation import random_flip, random_rotate, random_brightness, random_contrast, random_color_jitter, random_blur
from PIL import Image
import numpy as np
class PrepareDataset(Dataset):
    def __init__(self, weather, image_base_folder, seq_infos, labels, num_views=1, seq_len=3, data_augmentation=False, prepare_data=None):
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
            if self.num_views == 2:
                # For two views, get both view 1 and view 2 paths
                view1_paths = [self.prepare_data.get_image_path(dt, view=1) for dt in seq_info]
                view2_paths = [self.prepare_data.get_image_path(dt, view=2) for dt in seq_info]
                paths.append((view1_paths, view2_paths))
            else:
                # For one view, only get view 2 paths
                view2_paths = [self.prepare_data.get_image_path(dt, view=2) for dt in seq_info]
                paths.append(view2_paths)
        return paths
    
    def __len__(self):
        return len(self.weather)
    def _load_single_image(self, path):
        try:
            with Image.open(path) as img:
                # img = img.crop((0, 0, 512, 200))  
                img = img.convert("RGB")  # Ensure image is in RGB format
                if self.data_augmentation:
                    img = random_brightness(img)
                    img = random_color_jitter(img)
                    img = random_blur(img)
                img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
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

