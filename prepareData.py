import os
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
import random
from itertools import groupby
from operator import itemgetter
class PrepareData:
    def __init__(self, fp_images, fp_weather, num_views=1):
        self.image_base_folder = fp_images
        self.fp_weather = fp_weather
        self.data = self._load_weather_data()
        self.test_data = []
        self.num_views = num_views
        self.stats_stratus_days = None
        
    def _load_weather_data(self):
        npz_file = np.load(self.fp_weather, allow_pickle=True)

        data_all = {k: npz_file[k] for k in npz_file.files}
        df = pd.DataFrame(data_all['dole'])
        # Normalize the DataFrame to expand dictionary values into columns
        df = pd.json_normalize(df[0])

        df['datetime'] = pd.to_datetime(df['datetime'])
    
        
        return df

    def get_image_for_datetime(self, dt, view=1):
        date_str = dt.strftime('%Y-%m-%d')
        time_str = dt.strftime('%H%M')
        img_filename = f"1159_{view}_{date_str}_{time_str}.jpeg"
        img_path = os.path.join(self.image_base_folder, str(view), dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'), img_filename)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            #normalize images
            # img = img.resize((512, 512))
            img_array = np.array(img)
            # img_array = np.array(img)
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
        df['datetime_next'] = df['datetime'] + timedelta(minutes=60)

        valid_indices = []
    
        for idx, row in df.iterrows():
            meteo_row = row[[
                "gre000z0_nyon", "gre000z0_dole", "RR", "TD", "WG", "TT",
                "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD", "pres"
            ]]

            if meteo_row.isnull().any():
                print(f"Skipping row {idx} due to NaN values in meteo data.")
                continue
            img_array_1 = self.get_image_for_datetime(row['datetime'])
            if self.num_views > 1:
                img_array_2 = self.get_image_for_datetime(row['datetime'], view=2)
                img_array = [img_array_1, img_array_2]  # Store as a list of two images
            else:
                img_array = img_array_1  # Store as a single image
            if np.all(img_array_1 == 0) or (self.num_views > 1 and np.all(img_array_2 == 0)):
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


    def find_stratus_days(self, df=None, median_gap=None, mad_gap=None):
        if df is None:
            df = self.data
        df = df.copy()
        
        weather_df = df.reset_index()[['datetime', 'gre000z0_dole', 'gre000z0_nyon']].copy()
        # Suppose we have a DataFrame 'weather_df' with columns 'gre000z0_dole' and 'gre000z0_nyon'
        # Calculate the absolute difference between the two columns
        weather_df['gap_abs'] = weather_df['gre000z0_dole'] - weather_df['gre000z0_nyon']

        # Calculate the median and MAD of the difference
        if median_gap is None and mad_gap is None:
            median_gap = np.median(weather_df['gap_abs'])
            mad_gap = np.median(np.abs(weather_df['gap_abs'] - median_gap))
        # Calculate the Modified Z-Score
        weather_df['gap_abs_mod_zscore'] = 0.6745 * (weather_df['gap_abs'] - median_gap) / mad_gap

        # Define a threshold to identify outliers
        threshold = 3
        weather_df['large_gap_mod_zscore'] = weather_df['gap_abs_mod_zscore'] > threshold

        # Filter the data considered outliers
        large_gap_data = weather_df[weather_df['large_gap_mod_zscore']]
        # Print the results
        # Find sequences where there are more than 5 consecutive large differences
        large_gap_data = weather_df[weather_df['large_gap_mod_zscore']].copy()
        large_gap_data = large_gap_data.sort_values('datetime')

        # Create a boolean mask for large differences
        mask = weather_df['large_gap_mod_zscore'].values

        # Find runs of consecutive True values

        indices = np.where(mask)[0]
        groups = []
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            if len(group) > 2:
                groups.append(group)

        # Get the corresponding datetimes for each group
        consecutive_large_diff_dates = []
        for group in groups:
            dates = weather_df.iloc[group]['datetime'].dt.date.unique()
            consecutive_large_diff_dates.extend(dates)

        consecutive_large_diff_dates = np.unique(consecutive_large_diff_dates)


        # Find days with at least 8 large differences in total
        counts = large_gap_data['datetime'].dt.date.value_counts()
        days_with_8_or_more = counts[counts >= 3].index

            # Find intersection of days with >5 consecutive large differences and days with at least 8 large differences
        days_consecutive = set(consecutive_large_diff_dates)
        days_8_or_more = set(days_with_8_or_more)
        stratus_days = sorted(days_consecutive & days_8_or_more)
        stratus_days = [str(d) for d in stratus_days]
        non_stratus_days = sorted(set(df['datetime'].dt.strftime('%Y-%m-%d').unique()) - set(stratus_days))
  
        
        return stratus_days,non_stratus_days, (median_gap, mad_gap)
    
    def get_train_validation_days(self, train_days, split_ratio=0.2):
        train_days = list(train_days)
        # Use self.data to find stratus days
        stratus_days,_,_= self.find_stratus_days(self.data[self.data['date_str'].isin(train_days)])
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
        stratus_days, _,self.stats_stratus_days= self.find_stratus_days()
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
