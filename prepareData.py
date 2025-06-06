import os
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
import random

class PrepareData:
    def __init__(self, fp_images, fp_weather):
        self.image_base_folder = fp_images
        self.fp_weather = fp_weather
        self.data = self._load_weather_data()
        self.test_data = []

    def _load_weather_data(self):
        npz_file = np.load(self.fp_weather, allow_pickle=True)

        data_all = {k: npz_file[k] for k in npz_file.files}
        df = pd.DataFrame(data_all['dole'])
        # Normalize the DataFrame to expand dictionary values into columns
        df = pd.json_normalize(df[0])

        df['datetime'] = pd.to_datetime(df['datetime'])
    
        
        return df

    def get_image_for_datetime(self, dt):
        date_str = dt.strftime('%Y-%m-%d')
        time_str = dt.strftime('%H%M')
        img_filename = f"1159_2_{date_str}_{time_str}.jpeg"
        img_path = os.path.join(self.image_base_folder, dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'), img_filename)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img, dtype=np.float32) / 255.0 # Normalize to [0, 1]
            return img_array
        else:
            return np.zeros((512, 512, 3), dtype=np.uint8)

    def normalize_data(self, train_df, validation_df, test_df, var_order=None):
        log_vars = ["RR", "RS"]
        angle_var = "DD"
        stats = {}

        if var_order is None:
            min_vals = train_df.min()
            max_vals = train_df.max()
            range_vals = max_vals - min_vals
            range_vals = range_vals.replace(0, 1e-8)  # Avoid division by zero
            train_norm = (train_df - min_vals) / range_vals
            validation_norm = (validation_df - min_vals) / range_vals
            test_norm = (test_df - min_vals) / range_vals
            return train_norm.fillna(0), validation_norm.fillna(0), test_norm.fillna(0), {"min": min_vals, "max": max_vals}
        for var in var_order:
            if var == angle_var:
                continue
            values = train_df[var]
            stats[var] = {"min": values.min(), "max": values.max()}

        def process(df):
            df_processed = pd.DataFrame()
            for var in var_order:
                if var == angle_var:
                    angle_rad = np.deg2rad(pd.to_numeric(df[var], errors="coerce").fillna(0))
                    df_processed[f"{var}_cos"] = np.cos(angle_rad)
                    df_processed[f"{var}_sin"] = np.sin(angle_rad)
                else:
                    min_val = stats[var]["min"]
                    max_val = stats[var]["max"]
                    range_val = max_val - min_val
                    if range_val == 0:
                        range_val = 1e-8  # Avoid division by zero
                    df_processed[var] = ((df[var] - min_val) / range_val).fillna(0)
            return df_processed

        train_norm = process(train_df)
        validation_norm = process(validation_df)
        test_norm = process(test_df)
        return train_norm.values, validation_norm.values, test_norm.values, stats


    def filter_data(self, start_date, end_date, take_all_seasons=True):
        months_to_take = list(range(1, 13)) if take_all_seasons else [1, 2, 3, 9, 10, 11, 12]        

        mask = (self.data['datetime'].dt.date >= pd.to_datetime(start_date).date()) & \
               (self.data['datetime'].dt.date <= pd.to_datetime(end_date).date()) & \
               (self.data['datetime'].dt.month.isin(months_to_take))
        self.data = self.data[mask].copy()
  
        return self.data

    def prepare_data(self, df):
        x_images = []
        x_meteo = []
        y = []

        df = df.sort_values('datetime').reset_index(drop=True)
        df['datetime_next'] = df['datetime'] + timedelta(minutes=10)

        valid_indices = []
      
        for idx, row in df.iterrows():
            meteo_row = row[[
                "gre000z0_nyon", "gre000z0_dole", "RR", "TD", "WG", "TT",
                "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD", "pres"
            ]]

            if meteo_row.isnull().any():
                print(f"Skipping row {idx} due to NaN values in meteo data.")
                continue

            img_array = self.get_image_for_datetime(row['datetime'])
            if np.all(img_array == 0):
                print(f"Skipping row {idx} due to missing image data.")
                continue

            next_row = df[df['datetime'] == row['datetime_next']]
            if next_row.empty:
                print(f"Skipping row {idx} due to missing next row data.")
                continue

            x_meteo.append(meteo_row.values)
            x_images.append(img_array)
            y.append(next_row.iloc[0][["gre000z0_nyon", "gre000z0_dole"]].values)
            valid_indices.append(idx)

        # Convert to numpy arrays
        x_meteo = np.array(x_meteo)
        x_images = np.array(x_images)
        y = np.array(y)

        # Save filtered data
        self.data = df.loc[valid_indices].reset_index(drop=True)
       

        return x_meteo, x_images, y


    def find_stratus_days(self, df=None):
        if df is None:
            df = self.data
        df = df.copy()
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index)
        daily_dole = df['gre000z0_dole'].resample('D').median()
        daily_nyon = df['gre000z0_nyon'].resample('D').median()
        daily_diff = daily_dole - daily_nyon
        stratus_days = daily_diff[daily_diff > 80].index.strftime('%Y-%m-%d').tolist()
        return stratus_days
    
    def get_train_validation_days(self, train_days, split_ratio=0.2):
        train_days = list(train_days)
        # Use self.data to find stratus days
        stratus_days = self.find_stratus_days(self.data[self.data['date_str'].isin(train_days)])
        print("Stratus days found:", len(stratus_days))
        random.shuffle(stratus_days)
        split_index = int(split_ratio * len(stratus_days))
        train_stratus_days = set(stratus_days[split_index:])
        test_stratus_days = set(stratus_days[:split_index])
        
        non_stratus_days = [d for d in train_days if d not in stratus_days]
        random.shuffle(non_stratus_days)
        remaining_train = int(len(train_days) * (1 - split_ratio)) - len(train_stratus_days)

        train_days = set(list(train_stratus_days) + non_stratus_days[:remaining_train])
        test_days = set(list(test_stratus_days) + non_stratus_days[remaining_train:])
        print(f"Train days: {len(train_days)}, Test days: {len(test_days)}")
        return train_days, test_days
    
    def get_test_train_days(self, split_ratio=0.8):
        stratus_days = self.find_stratus_days()
        all_days = self.data['datetime'].dt.strftime('%Y-%m-%d').unique().tolist()
        print("Stratus days found:", len(stratus_days))
        random.shuffle(stratus_days)
        split_index = int(split_ratio * len(stratus_days))
        train_stratus_days = set(stratus_days[:split_index])
        test_stratus_days = set(stratus_days[split_index:])
       
        non_stratus_days = [d for d in all_days if d not in stratus_days]
        random.shuffle(non_stratus_days)
        remaining_train = int(len(all_days) * split_ratio) - len(train_stratus_days)
        train_days = set(list(train_stratus_days) + non_stratus_days[:remaining_train])
        test_days = set(list(test_stratus_days) + non_stratus_days[remaining_train:])
        return train_days, test_days

    def split_data(self, x_meteo, x_images, y):
        train_days, test_days = self.get_test_train_days()

        # Ensure indices in self.data are reset and consistent
        self.data = self.data.reset_index(drop=True)
        self.data['date_str'] = self.data['datetime'].dt.strftime('%Y-%m-%d')

        # Get train and test indices based on date_str
        train_indices = self.data[self.data['date_str'].isin(train_days)].index
        test_indices = self.data[self.data['date_str'].isin(test_days)].index
        column_names = [
            "gre000z0_nyon", "gre000z0_dole", "RR", "TD", "WG", "TT",
            "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD","pres"
        ]
        label_names = ["gre000z0_nyon", "gre000z0_dole"]

        # Prepare train and test DataFrames for x_meteo
        x_meteo_train_df = pd.DataFrame(x_meteo[train_indices.to_numpy()], columns=column_names)
        x_meteo_test_df = pd.DataFrame(x_meteo[test_indices.to_numpy()], columns=column_names)

        # Prepare train and test DataFrames for y (labels)
        y_train_df = pd.DataFrame(y[train_indices.to_numpy()], columns=label_names)
        y_test_df = pd.DataFrame(y[test_indices.to_numpy()], columns=label_names)

        x_images_train = x_images[train_indices.to_numpy()]
        x_images_test = x_images[test_indices.to_numpy()]

        # Prepare test data for evaluation
        test_datetimes = self.data.loc[test_indices, 'datetime'].values
        x_meteo_test_df['datetime'] = test_datetimes
        y_test_df['datetime'] = test_datetimes
        self.test_data = x_meteo_test_df.to_dict('records')

        # Also add datetime to train df for reference
        train_datetimes = self.data.loc[train_indices, 'datetime'].values
        x_meteo_train_df['datetime'] = train_datetimes
        y_train_df['datetime'] = train_datetimes
        print(f"Train days: {len(train_days)}, Test days: {len(test_days)}")
        
        
        return x_meteo_train_df, x_meteo_test_df, x_images_train, x_images_test, y_train_df, y_test_df

    def split_train_validation(self, x_meteo_df, x_images, y_df, validation_ratio=0.2):
        # Get training days and validation days from existing dates
        train_days = x_meteo_df['datetime'].dt.strftime('%Y-%m-%d').unique()
        train_day_set, val_day_set = self.get_train_validation_days(train_days, validation_ratio)

        x_meteo_df = x_meteo_df.copy()
        y_df = y_df.copy()
        x_meteo_df['date_str'] = x_meteo_df['datetime'].dt.strftime('%Y-%m-%d')
        y_df['date_str'] = y_df['datetime'].dt.strftime('%Y-%m-%d')

        train_mask = x_meteo_df['date_str'].isin(train_day_set)
        val_mask = x_meteo_df['date_str'].isin(val_day_set)

        column_names = [col for col in x_meteo_df.columns if col not in ["datetime", "date_str"]]
        label_names = [col for col in y_df.columns if col not in ["datetime", "date_str"]]

        # Prepare DataFrames for train/val splits
        x_meteo_train_df = x_meteo_df[train_mask].reset_index(drop=True)[column_names]
        x_meteo_val_df = x_meteo_df[val_mask].reset_index(drop=True)[column_names]
        y_train_df = y_df[train_mask].reset_index(drop=True)[label_names]
        y_val_df = y_df[val_mask].reset_index(drop=True)[label_names]

        # Add datetime back for reference
        x_meteo_train_df['datetime'] = x_meteo_df[train_mask].reset_index(drop=True)['datetime']
        x_meteo_val_df['datetime'] = x_meteo_df[val_mask].reset_index(drop=True)['datetime']
        y_train_df['datetime'] = y_df[train_mask].reset_index(drop=True)['datetime']
        y_val_df['datetime'] = y_df[val_mask].reset_index(drop=True)['datetime']

        img_train = x_images[train_mask.to_numpy()]
        img_val = x_images[val_mask.to_numpy()]
        return x_meteo_train_df, x_meteo_val_df, img_train, img_val, y_train_df, y_val_df


    def normalize_data_test(self, data, var_order=None, stats=None):
        df = pd.DataFrame(data, columns=var_order)
        angle_var = "DD"
        df_processed = pd.DataFrame()

        for var in var_order:
            if var == angle_var:
                # Convert degrees to radians, safely handling NaNs and non-numeric data
                angle_rad = np.deg2rad(pd.to_numeric(df[var], errors="coerce").fillna(0))
                df_processed[f"{var}_cos"] = np.cos(angle_rad)
                df_processed[f"{var}_sin"] =  np.sin(angle_rad)
            else:
                min_val = stats[var]["min"]
                max_val = stats[var]["max"]
                range_val = max_val - min_val if max_val != min_val else 1e-8
                df_processed[var] = ((df[var] - min_val) / range_val).fillna(0)

        return df_processed.values

    def load_data(self, start_date="2023-01-01", end_date="2024-12-31", take_all_seasons=False):
        filtered_df = self.filter_data(start_date, end_date, take_all_seasons)
        print(f"Filtered data shape: {filtered_df.shape}")
        x_meteo, x_images, y = self.prepare_data(filtered_df)
        return x_meteo, x_images, y
