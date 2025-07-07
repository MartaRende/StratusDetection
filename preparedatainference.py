from collections import defaultdict
import os
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
import random
from itertools import groupby
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor
import functools

class PrepareData:
    def __init__(self, fp_images, fp_weather, num_views=1, seq_length=3, prediction_time=6):
        
        self.image_base_folder = fp_images
        self.fp_weather = fp_weather
        if fp_weather.endswith('.npz'):
            self.data = self._load_weather_data()
            self.data_backup=self.data           
        self.test_data = []
        self.num_views = num_views
        self.stats_stratus_days = None
        self.seq_length = seq_length
        self.prediction_time = prediction_time # number of 10-minute intervals to predict ahead (default is 6 for 60 minutes)
        self.meteo_feats = ["gre000z0_nyon", "gre000z0_dole",
             "RR", "TD", "WG", "TT",
            "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD", "pres"
        ]
    
    def _load_weather_data(self):
                # load data test of npz file
        complete_data_gen_fp = f"data/complete_data_gen.npz"
        complete_data_gen = np.load(complete_data_gen_fp, allow_pickle=True)
        npz_file = np.load(self.fp_weather, allow_pickle=True)
        data_all = {k: npz_file[k] for k in npz_file.files}
        complete_data_gen = {k: complete_data_gen[k] for k in complete_data_gen.files}
        complete_df = pd.DataFrame(complete_data_gen['dole'])
        complete_df = pd.json_normalize(complete_df[0])
        complete_df['datetime'] = pd.to_datetime(complete_df['datetime'])
        df = pd.DataFrame(data_all['dole'])
        df = pd.json_normalize(df[0])
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Find intersection of dates (ignore hour) between complete_df and df
        complete_dates = complete_df['datetime'].dt.date
        df_dates = df['datetime'].dt.date
        common_dates = np.intersect1d(complete_dates, df_dates)
                # Filter complete_df to only include rows where the date is in common_dates
        complete_df = complete_df[complete_df['datetime'].dt.date.isin(common_dates)]
        return complete_df

    def get_image_path(self, dt,   view=2):
        """Get the path for an image without loading it"""
        if isinstance(dt, np.datetime64):
            dt = pd.Timestamp(dt)  # Convert to pandas Timestamp, which supports strftime

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

    def image_exists(self, dt, view=2):
        """Check if image exists without loading it"""
        return os.path.exists(self.get_image_path(dt, view))



    def filter_data(self, start_date, end_date, take_all_seasons=True):
        months_to_take = list(range(1, 13)) if take_all_seasons else [1, 2, 3, 9, 10, 11, 12]        

        mask = (self.data['datetime'].dt.date >= pd.to_datetime(start_date).date()) & \
               (self.data['datetime'].dt.date <= pd.to_datetime(end_date).date()) & \
               (self.data['datetime'].dt.month.isin(months_to_take))
        self.data = self.data[mask].copy()
        return self.data
    def get_image_for_datetime(self, dt, view=2):
        """Get the image for a specific datetime without loading it into memory"""
        if isinstance(dt, np.datetime64):
            dt = pd.Timestamp(dt)
        image_path = self.get_image_path(dt, view)
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
           # img = img.crop((0, 0, 512, 200))  # Crop to 512x200
            return np.array(img)
        else:
            print(f"Image not found for datetime {dt} at view {view}. Returning empty image.")
            return np.zeros((512, 512, 3), dtype=np.uint8)
    def prepare_data(self, df):
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create sequences
        x_images_seq = []
        x_meteo_seq = []
        y_seq = []
        valid_indices = []
        
        # Define the meteorological features to use
        meteo_features = ["gre000z0_nyon", "gre000z0_dole",
             "RR", "TD", "WG", "TT",
            "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD", "pres"
        ]
        
        # Iterate through possible sequence starting points
        for i in range(len(df) - self.seq_length):
            # Get the sequence window
            seq_window = df.iloc[i:i+self.seq_length]
            
            next_t_start = i + self.seq_length
            next_t_end = next_t_start + self.prediction_time  # prediction_time is the number of output points
            if next_t_end > len(df):
                break
            next_points = df.iloc[next_t_start:next_t_end]

            # if len(next_points) < 3:
            #     print(f"Skipping sequence starting at index {i} due to insufficient next points.")
            #     continue
            # Ensure next_points have 10-minute intervals
            next_time_diffs = np.diff(next_points['datetime'].values) / np.timedelta64(1, 'm')
            if not all(d == 10 for d in next_time_diffs):
            
                print(f"Skipping sequence starting at index {i} due to non-10-minute intervals in next_points.at datetime {next_points['datetime'].values[0]}.")
                # Interpolate missing values in next_points
                # next_points = next_points.interpolate(method='linear', limit_direction='both')
                # import ipdb
                # ipdb.set_trace()
                continue
            # Use the three next points as the target
            target = next_points[["gre000z0_nyon", "gre000z0_dole"]].values
     
            # Check for continuity (10-minute intervals)
            time_diffs = np.diff(seq_window['datetime'].values) / np.timedelta64(1, 'm')
           
            if not all(diff == 10 for diff in time_diffs):
                print(f"Skipping sequence starting at index {i} due to non-10-minute intervals. at datetime {seq_window['datetime'].values[0]}.")
           
                continue

            # Check if next point is exactly 60 minutes after last sequence point
            last_seq_time = seq_window.iloc[-1]['datetime']
          
            
            if (next_points['datetime'].values[0] - last_seq_time) != timedelta(minutes=10):
                print(f"Skipping sequence starting at index {i} due to non-60-minute gap to next point. at datetime {seq_window['datetime'].values[-1]}.")
                continue
           
            # Prepare meteorological data sequence
            meteo_sequence = seq_window[meteo_features].values
            if np.isnan(meteo_sequence).any():
                print(f"Skipping sequence starting at index {i} due to NaN values in meteorological data.at datetime {seq_window['datetime'].values[0]}.")
                continue
                
            # Prepare image sequence
            img_sequence = []
            valid_images = True
            for _, row in seq_window.iterrows():
                img = self.get_image_for_datetime(row['datetime'])
                if np.all(img == 0):  # Missing image
                    print(f"Skipping sequence starting at index {i} due to missing image for datetime {row['datetime']}.")
                    valid_images = False
                    break
                if self.num_views > 1:
                    img2 = self.get_image_for_datetime(row['datetime'], view=1)
                    if np.all(img2 == 0):
                        print(f"Skipping sequence starting at index {i} due to missing second view image for datetime {row['datetime']}.")
                        valid_images = False
                        break
                    img_sequence.append([img, img2])
                else:
                    img_sequence.append(img)
            
            if not valid_images:
                print(f"Skipping sequence starting at index {i} due to missing images.")
                continue
                
            # Prepare target (next time step after sequence)
            
          
            # Use pd.isnull to handle all types safely
            if pd.isnull(target).any():
                print(f"Skipping sequence starting at index {i} due to NaN values in target data. at datetime {seq_window['datetime'].values[0]}.")
                continue
        
            # Add to sequences
            x_meteo_seq.append(meteo_sequence)
            x_images_seq.append(np.array(img_sequence))
            y_seq.append(target)
            valid_indices.append(i)
           
        
        # Convert to numpy arrays
        x_meteo_seq = np.array(x_meteo_seq)
        x_images_seq = np.array(x_images_seq)
        y_seq = np.array(y_seq)
        # Save filtered data
        self.data = df.loc[valid_indices].reset_index(drop=True)
        self.data['date_str'] = self.data['datetime'].dt.strftime('%Y-%m-%d')
        print(len(self.data), "valid sequences found after filtering")
        return x_meteo_seq, x_images_seq, y_seq


    

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
        print(f"Median gap: {median_gap}, MAD gap: {mad_gap}")
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
    
    def normalize_data_test(self, data, var_order=None, stats=None):
        arr = np.array(data)
        original_ndim = arr.ndim

        if arr.ndim == 2:
            arr = arr[:, np.newaxis, :]  # Add the time dimension

        N, T, F = arr.shape
        flat = arr.reshape(N, T * F)
     
        df = pd.DataFrame(flat, columns=var_order)
        df_out = pd.DataFrame()

        for var in var_order:
            base_var = var.split('_')[0]  # es. 'T_0' → 'T'
            
            if base_var not in stats:
                raise ValueError(f"Missing stats for variable base '{base_var}'")

            col = df[var].astype(float).fillna(0)
            mn = stats[base_var]["min"]
            mx = stats[base_var]["max"]
            rng = mx - mn if mx != mn else 1e-8
            df_out[var] = ((col - mn) / rng).fillna(0)

        flat_out = df_out.values
        reshaped = flat_out.reshape(N, T, F)
     
        if original_ndim == 2:
            return reshaped[:, 0, :]  # Back to 2D
        return reshaped


    def load_data_test(self, start_date="2023-01-01", end_date="2024-12-31", take_all_seasons=False):
        filtered_df = self.filter_data(start_date, end_date, take_all_seasons)
        print(f"Filtered data shape: {filtered_df.shape}")
        x_meteo, x_images, y = self.prepare_data(filtered_df)
        return x_meteo, x_images, y
    