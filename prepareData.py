from collections import defaultdict
import torch    
import os
import numpy as np
import glob
from PIL import Image
import h5py

class PrepareData:
    def __init__(self, fp_images="/home/marta/Projects/tb/data/images/2023"):
        self.image_base_folder = fp_images


    def get_image_for_datetime(self,dt):
        date_part, time_part = dt.split('T')
        year, month, day = date_part.split('-')
        hourmin = time_part.replace(':', '')[:4]  # e.g. '0310' from '03:10:00'

        img_filename = f"1159_2_{year}-{month}-{day}_{hourmin}.jpeg"
        # If your images are in subfolders by month/day:
        img_path = os.path.join(self.image_base_folder, month, day, img_filename)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_arr = np.array(img)
        else:
            img_arr = np.zeros((512, 512, 3), dtype=np.uint8)  # Placeholder if missing
        return img_arr
    def normalize_data(self, data, var_order=None):
   
        log_vars = ["RR", "RS"]
        angle_var = "DD"

        if var_order is None:
            stats = {"mean": np.mean(data, axis=0), "std": np.std(data, axis=0)}
            x_norm = (data - stats["mean"]) / stats["std"]
            x_norm = np.nan_to_num(x_norm)
            return x_norm

        stats = {}
        for idx, var in enumerate(var_order):
            if var == angle_var:
                continue
            values = data[:, idx]
            stats[var] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

        x_norm = []
        for row in data:
            norm_row = []
            for idx, var in enumerate(var_order):
                val = row[idx]
                # print dole gre000z0 values
                if var == angle_var:
                    angle_rad = np.deg2rad(val)
                    norm_row.append(np.cos(angle_rad))
                    norm_row.append(np.sin(angle_rad))
                elif var in log_vars:
                    val = np.log1p(val)
                    mean, std = stats[var]["mean"], stats[var]["std"]
                    norm_row.append((val - mean) / std)
                else:
                    mean, std = stats[var]["mean"], stats[var]["std"]
                    norm_row.append((val - mean) / std)
                if np.isnan(norm_row[-1]):
                    norm_row[-1] = 0
            x_norm.append(norm_row)
        
        return np.array(x_norm), stats
            
    def prepare_data(self,loaded, filtered_datatimes=None):
        x_images = []
        y = []
        x_meteo = []
        dt_to_idx = {dt: i for i, dt in enumerate(loaded["datetime"])}
        for i, dt in enumerate(loaded["datetime"]):
            
            if filtered_datatimes is not None and dt not in filtered_datatimes:
                continue
            meteo_row = [
                loaded["gre000z0_nyon"][i], # rayonnement global
                loaded["gre000z0_dole"][i], # rayonnement global
                loaded["RR"][i], # precipitation
                loaded["TD"][i], # dew point temperature
                loaded["WG"][i], # wind gust
                loaded["TT"][i], # temperature
                loaded["CT"][i], #  couverture du ciel 
                loaded["FF"][i], #  wind speed
                loaded["RS"][i], #  snowfall
                loaded["TG"][i], #  ground temperature
                loaded["Z0"][i], #  ground height
                loaded["ZS"][i], #  snow height
                loaded["SU"][i], #  sunshine duration
                loaded["DD"][i], #  wind direction
            ]
            
            # Remove datetimes where any corresponding input data row has a NaN

            if not any(np.isnan(meteo_row)):
                x_meteo.append(meteo_row)
                x_images.append(self.get_image_for_datetime(dt))
                from datetime import datetime, timedelta
                dt_next = (datetime.fromisoformat(dt) + timedelta(minutes=10)).isoformat()

                if dt_next in dt_to_idx:
                    idx_next = dt_to_idx[dt_next]
                    temp = [loaded["gre000z0_nyon"][idx_next], loaded["gre000z0_dole"][idx_next]]
                    y.append(temp)
                else:
                    x_meteo.pop()
                    x_images.pop()

           
        x_meteo = np.array(x_meteo)
        x_images = np.array(x_images)
        y = np.array(y)
        return x_meteo, x_images, y
    # filter data for hour and days
    def filter_data(self, data, start_date, end_date, start_hour, end_hour, take_all_seasons=True):
        months_to_take = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] if take_all_seasons else [1, 2,3,4,9,10,11,12]
        filtered_data = []
        for dt in data["datetime"]:
            date_part, time_part = dt.split('T')
            hour = int(time_part.split(':')[0])
            month = int(date_part.split('-')[1])
            if start_date <= date_part <= end_date and start_hour <= hour <= end_hour and month in months_to_take:
                filtered_data.append(dt)
        # delete all row input data wich have a row nan
      
        return filtered_data
        
    def load_data(self, fp_weather):
        # Load the NPZ file
        npz_file = np.load(fp_weather, allow_pickle=True)
        
        # Convert to a regular dictionary
        data_all = {k: npz_file[k] for k in npz_file.files}
        
        # Get the 'dole' data
        dole_data = data_all['dole']
        
        # Initialize the result dictionary
        data = defaultdict(list)
        # Process each item in dole_data
        for item in dole_data:
                
            # Add all values to our dictionary of lists
            for key, value in item.items():
                data[key].append(value)
        
        # Convert defaultdict to regular dict
        data = dict(data)
        
        # Filter data
        filtered_data = self.filter_data(data, "2023-01-01", "2023-12-31", 9, 19, take_all_seasons=True)
        
        # Prepare the final datasets
        x_meteo, x_images, y = self.prepare_data(data, filtered_datatimes=filtered_data)
        print(f"Data Prepared: {fp_weather}")
        
        # Normalize data
        x_meteo, _ = self.normalize_data(x_meteo, var_order=[
            "gre000z0_nyon", "gre000z0_dole", "RR", "TD", "WG", "TT", 
            "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD"
        ])
        y, stats = self.normalize_data(y, var_order=["gre000z0_nyon", "gre000z0_dole"])
        
        return x_meteo, x_images, y, stats