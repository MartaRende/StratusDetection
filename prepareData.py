from collections import defaultdict
from datetime import datetime, timedelta
import torch    
import os
import numpy as np
import glob
from PIL import Image
import h5py
import pandas as pd
import random
class PrepareData:
    def __init__(self, fp_images="/home/marta/Projects/tb/data/images/mch/1159/2/", fp_weather="data/complete_data.npz", fp_global_rayonnement="data/rayonnement_global"):
        self.image_base_folder = fp_images
        self.fp_weather = fp_weather
        npz_file = np.load(fp_weather, allow_pickle=True)
        # Convert to a regular dictionary
        data_all = {k: npz_file[k] for k in npz_file.files}
        self.data= data_all['dole']
        self.test_data = []



    def get_image_for_datetime(self,dt):
        date_part, time_part = dt.split('T')
        year, month, day = date_part.split('-')
        hourmin = time_part.replace(':', '')[:4]  # e.g. '0310' from '03:10:00'

        img_filename = f"1159_2_{year}-{month}-{day}_{hourmin}.jpeg"
        # If your images are in subfolders by month/day:
        img_path = os.path.join(self.image_base_folder, year,month, day, img_filename)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_arr = np.array(img)
        else:
            img_arr = np.zeros((512, 512, 3), dtype=np.uint8)  # Placeholder for missing images
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
                if np.isnan(norm_row[-1]).any():
                    norm_row[-1] = 0
            x_norm.append(norm_row)
        
        return np.array(x_norm), stats
            
    def prepare_data(self,loaded, filtered_datatimes=None):
        x_images = []
        y = []
        x_meteo = []
        dt_to_idx = {}
        for i in range(len(loaded)):
            dt = loaded[i]["datetime"]
            dt_to_idx[dt]  = i
        for i in range(len(loaded)):
            dt = loaded[i]["datetime"]
            if filtered_datatimes is not None and dt not in filtered_datatimes:
                continue
            meteo_row = [
                loaded[i]["gre000z0_nyon"], # rayonnement global
                loaded[i]["gre000z0_dole"], # rayonnement global
                loaded[i]["RR"], # precipitation
                loaded[i]["TD"], # dew point temperature
                loaded[i]["WG"], # wind gust
                loaded[i]["TT"], # temperature
                loaded[i]["CT"], #  couverture du ciel 
                loaded[i]["FF"], #  wind speed
                loaded[i]["RS"], #  snowfall
                loaded[i]["TG"], #  ground temperature
                loaded[i]["Z0"], #  ground height
                loaded[i]["ZS"], #  snow height
                loaded[i]["SU"], #  sunshine duration
                loaded[i]["DD"], #  wind direction
            
            ]
            
            # Remove datetimes where any corresponding input data row has a NaN

            if not any(np.isnan(meteo_row)):
                x_meteo.append(meteo_row)
                x_images.append(self.get_image_for_datetime(dt))
                # if imge contains 0 values, remove the row
                if np.all(x_images[-1]==0):
                    x_meteo.pop()
                    x_images.pop()
                    self.data = [row for row in self.data if row["datetime"] != dt]
                    # print(f"Image for {dt} is empty, removing row.")
                    continue

                dt_next = (datetime.fromisoformat(dt) + timedelta(minutes=60)).isoformat()

                if dt_next in dt_to_idx:
                    idx_next = dt_to_idx[dt_next]
                    temp = [loaded[idx_next]["gre000z0_nyon"], loaded[idx_next]["gre000z0_dole"]]
                    y.append(temp)
                else:
                    # print(f"Warning: {dt_next} not found in data.")

                    x_meteo.pop()
                    x_images.pop()
                    # Remove row from self.data
                    self.data = [row for row in self.data if row["datetime"] != dt]
                    


        print(f"Filtered data: {len(x_meteo)} rows")
        x_meteo = np.array(x_meteo)
        x_images = np.array(x_images)
        y = np.array(y)
        return x_meteo, x_images, y
    # filter data for hour and days
    def filter_data(self, data, start_date, end_date,take_all_seasons=True):
        months_to_take = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] if take_all_seasons else [1, 2,3,4,9,10,11,12]
        filtered_data = []
        for i in range(len(data)):
            dt = data[i]["datetime"]
            date_part, time_part = dt.split('T')
            month = int(date_part.split('-')[1])
            if start_date <= date_part <= end_date  and month in months_to_take:
                filtered_data.append(dt)
    # delete all row input data wich have a row nan
      
        return filtered_data
    # def sort_data_by_day(self, data):
    #     day_to_indices = defaultdict(list)
    #     for idx, dt in enumerate(data["datetime"]):
    #         day = dt.split('T')[0]
    #         day_to_indices[day].append(idx)

    #     days = sorted(day_to_indices.keys())
    #     return days
    def find_startus_days(self,data):
        
        dole_data = {}
        nyon_data = {}
  
        for i in range(len(data)):
            dole_data[data[i]["datetime"]] = data[i]["gre000z0_dole"]
            nyon_data[data[i]["datetime"]] = data[i]["gre000z0_nyon"]

        dole_df = pd.DataFrame.from_dict(dole_data, orient='index', columns=['gre000z0_dole'])
        nyon_df = pd.DataFrame.from_dict(nyon_data, orient='index', columns=['gre000z0_nyon'])

        # Step 3: Convert the index to datetime and group by day
        dole_df.index = pd.to_datetime(dole_df.index)
        nyon_df.index = pd.to_datetime(nyon_df.index)

        dole_daily = dole_df.resample('D').median()  # Daily mean
        nyon_daily = nyon_df.resample('D').median()

        daily_diff = dole_daily['gre000z0_dole'] - nyon_daily['gre000z0_nyon']
        stratus_days = daily_diff[daily_diff > 120].index
        all_days = []
        for i in stratus_days:
            day = i.strftime('%Y-%m-%d')  # Convert Timestamp to string
            all_days.append(day)
        return stratus_days
        # Find the start date
    def get_test_train_days(self, split_ratio=0.8):
        # npz_file = np.load("data/complete_data.npz", allow_pickle=True)
        # data = {k: npz_file[k] for k in npz_file.files}
        # data = data['dole']
        all_days = list(set([
            d["datetime"].split('T')[0] if 'T' in d["datetime"] else d["datetime"].split(' ')[0]
            for d in self.data
        ]))

        random.seed(42)
        all_days.sort()
        random.shuffle(all_days)
        split_index = int(split_ratio * len(all_days))
        train_days = set(all_days[:split_index])
        test_days = set(all_days[split_index:])
        
        return train_days, test_days

    def get_indices_for_days(self, data, days):
        indices = []
        for i in range(len(self.data)):
    
            dt = self.data[i]["datetime"]
            day = dt.split('T')[0] if 'T' in dt else dt.split(' ')[0]

            if day in days:
                indices.append(i)
    
        return indices
   
    def split_data(self, x_meteo, x_images, y):
        # Split the data into training and testing sets
        train_days, test_days = self.get_test_train_days()
        train_indices = self.get_indices_for_days(self.data, train_days)
        test_indices = self.get_indices_for_days(self.data, test_days)
    
        x_meteo_train = x_meteo[train_indices]
        x_images_train = x_images[train_indices]
        y_train = y[train_indices]

        x_meteo_test = x_meteo[test_indices]
        x_images_test = x_images[test_indices]
        y_test = y[test_indices]
        
        # Save test data with datetime information
        test_datetimes = [self.data[i]["datetime"] for i in test_indices]
        # Define column names for the data
        column_names = [
            "gre000z0_nyon",  # Global radiation at Nyon
            "gre000z0_dole",  # Global radiation at Dole
            "RR",             # Precipitation
            "TD",             # Dew point temperature
            "WG",             # Wind gust
            "TT",             # Temperature
            "CT",             # Sky coverage
            "FF",             # Wind speed
            "RS",             # Snowfall
            "TG",             # Ground temperature
            "Z0",             # Ground height
            "ZS",             # Snow height
            "SU",             # Sunshine duration
            "DD",             # Wind direction
        ]

        # Combine x_meteo_test and test_datetimes
        test_datetimes = np.array(test_datetimes).reshape(-1, 1)
        combined_data = np.concatenate((x_meteo_test, test_datetimes), axis=1)

        # Build list of dictionaries
        dole = []
        dole = []
        for row in combined_data:
            datetime_value = row[-1]  
            entry = {
                column_names[i]: float(row[i]) if '.' in str(row[i]) else int(row[i])
                for i in range(len(column_names))
            }
            entry['datetime'] = datetime_value
            dole.append(entry)
        # to save data in training process
        self.test_data = dole
        print(f"Train data shape: {x_meteo_train.shape}, Test data shape: {x_meteo_test.shape}")
        print(f"len of test data: {len(self.test_data)}")
   
        return x_meteo_train, x_meteo_test, x_images_train, x_images_test, y_train, y_test
    def rebuild_data_with_filtered_datetimes(self, filtered_datetimes):
        filtered_set = set(filtered_datetimes)
        self.data = [row for row in self.data if row["datetime"] in filtered_set]
        
    def load_data(self, fp_weather):
 
        # Filter data
        filtered_datetimes = self.filter_data(self.data, "2023-01-01", "2023-12-31", take_all_seasons=False)
        self.rebuild_data_with_filtered_datetimes(filtered_datetimes)
     
        # Prepare the final datasets
        x_meteo, x_images, y = self.prepare_data(self.data, filtered_datatimes=filtered_datetimes)
        
        print(f"Data Prepared: {fp_weather}")
        
        
        return x_meteo, x_images, y