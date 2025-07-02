import os
import numpy as np
import pandas as pd
from PIL import Image
from datetime import timedelta
import random
from itertools import groupby
from operator import itemgetter

class PrepareData:
    def __init__(self, fp_images, fp_weather, num_views=1, seq_length=3, prediction_minutes=10):
        self.image_base_folder = fp_images
        self.fp_weather = fp_weather
        if fp_weather.endswith('.npz'):
            self.data = self._load_weather_data()
            self.data_backup=self.data           
        self.test_data = []
        self.num_views = num_views
        self.stats_stratus_days = None
        self.seq_length = seq_length
        self.meteo_feats = ["gre000z0_nyon", "gre000z0_dole",
             "RR", "TD", "WG", "TT",
            "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD", "pres"
        ]
        self.desired_prediction = prediction_minutes  # minutes for prediction

    def _load_weather_data(self):
        npz_file = np.load(self.fp_weather, allow_pickle=True)
        data_all = {k: npz_file[k] for k in npz_file.files}
        df = pd.DataFrame(data_all['dole'])
        df = pd.json_normalize(df[0])
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    def get_image_path(self, dt, view=2):
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
    
    def get_valid_sequences(self):
        """Identify valid sequences without modifying self.data during iteration"""
        df = self.data.sort_values("datetime").reset_index(drop=True)
        valid_seqs = []
        valid_indices = []

        for i in range(len(df) - self.seq_length):
            seq = df.iloc[i:i+self.seq_length]
            # Compute the next point using self.desired_prediction (in minutes)
            next_time = seq.iloc[-1]['datetime'] + timedelta(minutes=self.desired_prediction)
            next_t_row = df[df['datetime'] == next_time]
            if next_t_row.empty:
                print(f"Skipping sequence starting at index {i} due to missing next_t at {next_time}.")
                continue
            next_t = next_t_row.iloc[0]

            # Check time continuity
            time_diffs = np.diff(seq['datetime'].values) / np.timedelta64(1, 'm')
            if not all(d == 10 for d in time_diffs):
                print(f"Skipping sequence starting at index {i} due to non-10-minute intervals. at hour {seq.iloc[0]['datetime']}")
                continue
           
            if (next_t['datetime'] - seq.iloc[-1]['datetime']) != timedelta(minutes=self.desired_prediction):
                print(f"Skipping sequence starting at index {i} due to non-{self.desired_prediction}-minute gap to next point.at hour {seq.iloc[-1]['datetime']}")
                continue

            # Check for missing meteo data
            if seq[self.meteo_feats].isna().any().any():
                print(f"Skipping sequence starting at index {i} due to NaN values in meteorological data. at hour {seq.iloc[0]['datetime']}")
                continue

            # Check for missing target values
            if next_t[["gre000z0_nyon", "gre000z0_dole"]].isna().any():
                print(f"Skipping sequence starting at index {i} due to NaN values in target data. at hour {seq.iloc[-1]['datetime']}")
                continue

            # Check if all images exist
            all_images_exist = True
            for _, row in seq.iterrows():
                if not self.image_exists(row['datetime']):
                    all_images_exist = False
                    print(f"Skipping sequence starting at index {i} due to missing image for datetime {row['datetime']}.")
                    break
                if self.num_views > 1 and not self.image_exists(row['datetime'], view=1):
                    all_images_exist = False
                    print(f"Skipping sequence starting at index {i} due to missing second view image for datetime {row['datetime']}.")
                    break

            if not all_images_exist:
                continue

            # Save valid sequence
            valid_indices.append(i)
            valid_seqs.append({
                "indices": list(range(len(valid_indices) - self.seq_length, len(valid_indices))),
                "datetimes": seq['datetime'].tolist(),
                "target": next_t[["gre000z0_nyon", "gre000z0_dole"]].values,
                "next_datetime": next_t['datetime']
            })
   
        # Update self.data only after collecting all valid sequences
        self.data = self.data.loc[valid_indices].reset_index(drop=True)

        for seq in valid_seqs:
            seq["indices"] = [self.data.index[i] for i in seq["indices"]]
      
        return valid_seqs
    def load_data(self, start_date="2023-01-01", end_date="2024-12-31"):
        """Load and prepare data without loading images into memory"""
        df = self.filter_data(start_date, end_date, take_all_seasons=False)
        self.data = df.sort_values("datetime").reset_index(drop=True)
        self.data_backup = self.data.copy()
        valid_seqs = self.get_valid_sequences()
        
        # Prepare metadata and targets
        meteo_data = []
        targets = []
        datetime_info = []
        datetime_next = []

        for seq in valid_seqs:
            meteo_data.append(self.data.loc[seq["indices"]][self.meteo_feats].values)
            targets.append(seq["target"])
            datetime_info.append(self.data.loc[seq["indices"][-1], 'datetime'])
            datetime_next.append(seq["next_datetime"])
        # Convert to numpy arrays
        x_meteo = np.array(meteo_data)
        y = np.array(targets)
  
        # Store datetime information for later use
        self.datetime_info = datetime_info
        self.datetime_next = datetime_next
        
        print(f"Number of valid sequences: {len(valid_seqs)}")

        return x_meteo, valid_seqs, y


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
          
            values = train_df[var]
            stats[var] = {"min": values.min(), "max": values.max()}
        
        # if len(var_order) > 3:
        #     var_order = [var for var in var_order if not var.startswith('gre000z0_nyon') and not var.startswith('gre000z0_dole')]

        def process(df):
            df_processed = pd.DataFrame()
            for var in var_order:
             
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

        train_days = set(list(train_stratus_days)+ non_stratus_days[:remaining_train])
        test_days = set(list(test_stratus_days)+ non_stratus_days[remaining_train:])
        print(f"Train days: {len(train_days)}, Test days: {len(test_days)}")
      
        return train_days, test_days
    
    def get_test_train_days(self, split_ratio=0.8):
        # Find stratus and non-stratus days
        stratus_days, non_stratus_days, self.stats_stratus_days = self.find_stratus_days()
        all_days = sorted(self.data['datetime'].dt.strftime('%Y-%m-%d').unique().tolist())
        
        # Specific test days to always include (including some stratus days)
        specific_test_days = [
            "2023-03-02", "2024-12-26", "2023-02-13", "2024-10-25", "2024-11-03", 
            "2024-11-08", "2023-01-27", "2023-01-25", "2023-02-09", "2024-10-30",
            "2024-11-09", "2024-10-19", "2024-11-16", "2023-03-02"
        ]
        # Only keep specific test days that are in all_days
        specific_test_days = [d for d in specific_test_days if d in all_days]
        
        # Calculate required test size (20% of all days)
        total_days = len(all_days)
        required_test_size = int(total_days * (1 - split_ratio))
        
        # Start with specific test days
        test_days = set(specific_test_days)
        
        # Calculate how many more days we need for test set
        remaining_test_needed = max(0, required_test_size - len(test_days))
        
        # Get non-stratus days not in specific test days
        available_non_stratus = [d for d in non_stratus_days if d not in specific_test_days]
        random.shuffle(available_non_stratus)
        
        # Add required number of non-stratus days to reach 20%
        if remaining_test_needed > 0:
            additional_test_days = available_non_stratus[:remaining_test_needed]
            test_days.update(additional_test_days)
        
        # All remaining days go to train set
        train_days = set([d for d in all_days if d not in test_days])
        
        # Verify no stratus days in test except specific ones
        test_stratus_days = [d for d in test_days if d in stratus_days]
        non_specific_test_stratus = [d for d in test_stratus_days if d not in specific_test_days]
        
        if non_specific_test_stratus:
            print(f"Warning: Found {len(non_specific_test_stratus)} stratus days in test set not in specific list")
        
        print(f"Total days: {total_days}")
        print(f"Train days: {len(train_days)} ({len(train_days)/total_days:.1%})")
        print(f"Test days: {len(test_days)} ({len(test_days)/total_days:.1%})")
        print(f" - Specific test days: {len(specific_test_days)}")
        print(f" - Additional non-stratus test days: {len(test_days) - len(specific_test_days)}")
        
        return train_days, test_days

    def split_data(self, x_meteo, x_images, y):
        train_days, test_days = self.get_test_train_days()
     
        # Ensure indices in self.data are reset and consistent
        self.data = self.data.reset_index(drop=True)
        self.data['date_str'] = self.data['datetime'].dt.strftime('%Y-%m-%d')
        self.data_backup['date_str'] = self.data_backup['datetime'].dt.strftime('%Y-%m-%d')
          # Find all rows in self.data present in test_days
        test_rows = self.data_backup[self.data_backup['date_str'].isin(test_days)]
      
        self.test_data = test_rows
        self.test_data = self.test_data.to_dict('records')
       
        # Get train and test indices based on date_str
        train_indices = self.data[self.data['date_str'].isin(train_days)].index
        test_indices = self.data[self.data['date_str'].isin(test_days)].index
        
        # Create sequence indices (3 timestamps per sequence)
        def generate_sequence_indices():
            sequence_indices = []
            for i in range(len(self.data) - self.seq_length):
                seq_window = self.data.iloc[i:i+self.seq_length]
                time_diffs = np.diff(seq_window['datetime'].values) / np.timedelta64(1, 'm')
                if all(diff == 10 for diff in time_diffs):  # Ensure 10-minute intervals
                    sequence_indices.append(range(i, i + self.seq_length))
            return sequence_indices
        num_sequences = len(x_meteo)
        sequence_indices = generate_sequence_indices()

        # Filter sequences based on train and test indices
        train_sequences = [indices for indices in sequence_indices if all(idx in train_indices for idx in indices)]
        test_sequences = [indices for indices in sequence_indices if all(idx in test_indices for idx in indices)]
        
        # Prepare train and test DataFrames for x_meteo
        column_names = [
            f"{feat}_t{t}" for t in range(self.seq_length) for feat in [
                "gre000z0_nyon", "gre000z0_dole", "RR", "TD", "WG", "TT",
                "CT", "FF", "RS", "TG", "Z0", "ZS", "SU", "DD", "pres"
            ]
        ]
        label_names = ["gre000z0_nyon", "gre000z0_dole"]
        self.datetime_next = pd.Series(self.datetime_next)

        # Fix: Use the sequence index's last element to select the correct sample (not the whole sequence)
        x_meteo_train = np.array([x_meteo[indices[-1]] for indices in train_sequences]).reshape(-1, self.seq_length * len(column_names) // self.seq_length)
        x_meteo_test = np.array([x_meteo[indices[-1]] for indices in test_sequences]).reshape(-1, self.seq_length * len(column_names) // self.seq_length)
        feature_columns = [col for col in column_names if col not in ['gre000z0_nyon_t0', 'gre000z0_dole_t0', 'gre000z0_nyon_t1', 'gre000z0_dole_t1', 'gre000z0_nyon_t2', 'gre000z0_dole_t2']]
        x_meteo_train_features_df = pd.DataFrame(x_meteo_train, columns=column_names)[feature_columns]
        x_meteo_test_features_df = pd.DataFrame(x_meteo_test, columns=column_names)[feature_columns]
        x_meteo_train_df = pd.DataFrame(x_meteo_train, columns=column_names)
        x_meteo_test_df = pd.DataFrame(x_meteo_test, columns=column_names)
        
        # Prepare train and test DataFrames for y (labels)
        y_train = np.array([y[indices[0]] for indices in train_sequences])
        y_test = np.array([y[indices[0]] for indices in test_sequences])
  
        y_train_df = pd.DataFrame(y_train, columns=label_names)
        y_test_df = pd.DataFrame(y_test, columns=label_names)

        # Prepare train and test arrays for x_images
        
        x_images_train = np.array([x_images[indices[-1]] for indices in train_sequences])
        x_images_test = np.array([x_images[indices[-1]] for indices in test_sequences])
    
        # Prepare test data for evaluation
        test_datetimes = self.data.loc[[indices[0] for indices in test_sequences], 'datetime'].values
        x_meteo_test_df['datetime'] = test_datetimes
        test_t_datetimes = self.datetime_next[[indices[0] for indices in test_sequences]].values

        y_test_df['datetime'] = test_t_datetimes
      
    # Find the datetime sequence for each train and test sample
        train_datetime_seq = [self.data.loc[list(indices), 'datetime'].tolist() for indices in train_sequences]
        test_datetime_seq = [self.data.loc[list(indices), 'datetime'].tolist() for indices in test_sequences]
   
        # Also add datetime to train df for reference
        train_datetimes = self.data.loc[[indices[0] for indices in train_sequences], 'datetime'].values
        x_meteo_train_features_df['datetime'] = train_datetimes
        x_meteo_test_features_df['datetime'] = test_datetimes
        x_meteo_train_df['datetime'] = train_datetimes
        # Compute t datetimes: the datetime corresponding to each sequence's prediction time (i.e., the time step after the sequence)
  

        t_datetimes_train = self.datetime_next[[indices[0] for indices in train_sequences]].values

        y_train_df['datetime'] = t_datetimes_train
        
        # Find all rows in self.data present in test_days
        test_rows = self.data[self.data['date_str'].isin(test_days)]
        print(f"Number of rows in self.data present in test_days: {len(test_rows)}")
        
        test_data = x_meteo_test_df.drop(columns=[c for c in x_meteo_test_df.columns if c.endswith('_t1') or c.endswith('_t2')])

        # Rename columns to remove '_t0' suffix
        test_data.columns = [c[:-3] if c.endswith('_t0') else c for c in test_data.columns]
        self.test_data = test_data.to_dict('records')
 
        return x_meteo_train_df, x_images_train, y_train_df, x_meteo_test_df, x_images_test, y_test_df, train_datetime_seq, test_datetime_seq
    def split_train_validation(self, x_meteo_seq, x_images_seq, y_seq, validation_ratio=0.2):
        # Ensure datetime and date_str columns exist
        if 'date_str' not in x_meteo_seq.columns:
            x_meteo_seq['date_str'] = x_meteo_seq['datetime'].dt.strftime('%Y-%m-%d')
        
        # Get train and validation days
        unique_days = x_meteo_seq['date_str'].unique()
        train_days, val_days = self.get_train_validation_days(unique_days, validation_ratio)

        # Generate valid sequences (10-minute intervals)
        sequence_indices = []
        for i in range(len(x_meteo_seq) - self.seq_length + 1):
            seq_window = x_meteo_seq.iloc[i:i+self.seq_length]
            time_diffs = np.diff(seq_window['datetime'].values) / np.timedelta64(1, 'm')
            if all(diff == 10 for diff in time_diffs):
                sequence_indices.append(list(range(i, i + self.seq_length)))

        # Split sequences into train/validation
        train_sequences = [
            indices for indices in sequence_indices 
            if all(x_meteo_seq.iloc[idx]['date_str'] in train_days for idx in indices)
        ]
        val_sequences = [
            indices for indices in sequence_indices 
            if all(x_meteo_seq.iloc[idx]['date_str'] in val_days for idx in indices)
        ]

        # Prepare features and labels
        column_names = [col for col in x_meteo_seq.columns if col not in ['datetime_t0','datetiem_t1','datetime_t2' 'date_str_t0', 'date_str_t2', 'date_str_t1']]
        
        # Use iloc for positional indexing
        x_meteo_train = np.array([x_meteo_seq.loc[indices[0], column_names].values
                         for indices in train_sequences])
        x_meteo_val = np.array([x_meteo_seq.loc[indices[0], column_names].values
                         for indices in val_sequences])
   
        x_images_train = np.array([x_images_seq[indices[0]] for indices in train_sequences])
        x_images_val = np.array([x_images_seq[indices[0]] for indices in val_sequences])
    
        
        y_train = np.array([y_seq.iloc[indices[0]] for indices in train_sequences])
        y_val = np.array([y_seq.iloc[indices[0]] for indices in val_sequences])
    
        # Create DataFrames
        x_meteo_train_df = pd.DataFrame(x_meteo_train, columns=column_names)
        x_meteo_val_df = pd.DataFrame(x_meteo_val, columns=column_names)
        # Extract train and validation datetimes for reference
        train_datetimes = x_meteo_seq.loc[[indices[-1] for indices in train_sequences], 'datetime'].values
        val_datetimes = x_meteo_seq.loc[[indices[-1] for indices in val_sequences], 'datetime'].values

        train_datetime_seq = [x_meteo_seq.loc[list(indices), 'datetime'].tolist() for indices in train_sequences]
        val_datetime_seq = [x_meteo_seq.loc[list(indices), 'datetime'].tolist() for indices in val_sequences]

        y_train_df = pd.DataFrame(y_train, columns=["gre000z0_nyon", "gre000z0_dole", "datetime"])
        y_val_df = pd.DataFrame(y_val, columns=["gre000z0_nyon", "gre000z0_dole", "datetime"])
        return x_meteo_train_df, x_images_train, y_train_df, x_meteo_val_df, x_images_val, y_val_df, train_datetime_seq, val_datetime_seq

   