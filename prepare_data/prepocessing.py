import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from pyproj import Transformer
from netCDF4 import Dataset, num2date
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

def get_grid_indices(x_grid, y_grid, coordinates):
    """Cache grid indices for each coordinate."""
    indices = {}
    for name, (x_target, y_target) in coordinates.items():
        ix = np.abs(x_grid - x_target).argmin()
        iy = np.abs(y_grid - y_target).argmin()
        indices[name] = (ix, iy)
    return indices

import traceback

def process_file(fp, datetimes, coordinates, idaweb_data):
    """Process a single NetCDF file for a list of datetime strings."""
    try:
        with Dataset(fp) as nc:
            time_var = nc.variables['datetime']
            all_datetimes = num2date(time_var[:], units=time_var.units)
            all_datetimes_str = [dt.isoformat() for dt in all_datetimes]
            x_grid = nc.variables['x'][:]
            y_grid = nc.variables['y'][:]
            grid_indices = get_grid_indices(x_grid, y_grid, coordinates)

            results = []
            for dt_str in datetimes:
                if dt_str not in all_datetimes_str:
                    continue
                idx = all_datetimes_str.index(dt_str)
                data_entry = {}

                for name, (ix, iy) in grid_indices.items():
                    point_data = {}
                    point_data["datetime"] = dt_str
                    has_nan = False
                    for var in ['RR', 'TD', 'WG', 'TT', 'CT', 'FF', 'RS', 'TG', 'Z0', 'ZS', 'SU', 'DD']:
                        try:
                            value = float(nc.variables[var][idx, iy, ix]) if var in nc.variables else np.nan
                        except Exception as e:
                            print(f"Error processing variable '{var}' in file '{fp}' at datetime '{dt_str}': {e}")
                            traceback.print_exc()  # Print the full stack trace
                            value = np.nan
                        point_data[var] = value
                        if np.isnan(value):
                            has_nan = True
                    # Merge IDAWeb if available
                    for loc in ["nyon", "dole"]:
                        key = f"gre000z0_{loc}"
                        if loc in idaweb_data and dt_str in idaweb_data[loc]:
                            point_data[key] = idaweb_data[loc][dt_str]["gre000z0"]
                        else:
                            point_data[key] = np.nan
                    if not has_nan:
                        data_entry = point_data
                if data_entry:
                    results.append(data_entry)
            return results
    except Exception as e:
        print(f"Error processing file '{fp}': {e}")
        traceback.print_exc()  # Print the full stack trace
        return []
def group_data_by_location(processed_data):
    grouped = {"nyon": [], "dole": []}
    for entry in processed_data:
        for location in grouped.keys():
            if location in entry:
                grouped[location].append(entry[location])
    return grouped

class PreProcessData:
    def __init__(self, start_date, end_date, fp_inca, fp_images, fp_global_rayonnement):
        self.start_date = start_date
        self.end_date = end_date
        self.fp_inca = fp_inca
        self.fp_images = fp_images
        self.fp_global_rayonnement = fp_global_rayonnement

        self.cache_dir = Path(".data_cache")
        self.cache_dir.mkdir(exist_ok=True)

        self.points = {"dole": (46.424797, 6.099136)}
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:21781")
        self.coordinates = {
            name: self.transformer.transform(lat, lon)
            for name, (lat, lon) in self.points.items()
        }

        self.datetime_index = {}
        self.idaweb_data = {}
        cache_key = hashlib.md5(fp_inca.encode()).hexdigest()
        self.index_cache_file = f"data/nc_index_cache_{cache_key}.pkl"

        self._load_or_build_index()
        self._preload_idaweb_data()

    def _load_or_build_index(self):
        if os.path.exists(self.index_cache_file):
            try:
                with open(self.index_cache_file, 'rb') as f:
                    self.datetime_index = pickle.load(f)
                print("Loaded datetime index from cache")
                return
            except Exception as e:
                print(f"Error loading index cache: {e}. Rebuilding...")

        self._build_datetime_index()
        try:
            with open(self.index_cache_file, 'wb') as f:
                pickle.dump(self.datetime_index, f)
            print("Saved new datetime index to cache")
        except Exception as e:
            print(f"Warning: Could not save index cache: {e}")

    def _build_datetime_index(self):
        self.datetime_index = {}
        nc_files = sorted(glob.glob(os.path.join(self.fp_inca, "/**/*.nc")))
        for fp in tqdm(nc_files, desc="Building datetime index"):
            try:
                with Dataset(fp) as nc:
                    time_var = nc.variables['datetime']
                    datetimes = num2date(time_var[:], units=time_var.units)
                    for dt in datetimes:
                        dt_str = dt.isoformat()
                        self.datetime_index[dt_str] = fp
            except Exception as e:
                print(f"Error processing {fp}: {e}")
                continue

    def _preload_idaweb_data(self):
        txt_files = glob.glob(os.path.join(self.fp_global_rayonnement, "*.txt"))
        data = defaultdict(dict)

        for txt_file in txt_files:
            name = "dole" if "dole" in txt_file.lower() else "nyon" if "nyon" in txt_file.lower() else None
            if not name:
                continue
            try:
                df = pd.read_csv(txt_file, sep=";", comment="#", engine="python")
                df['time'] = pd.to_datetime(df['time'], format="%Y%m%d%H%M", errors='coerce')
                df = df.dropna(subset=['time'])
                for _, row in df.iterrows():
                    data[name][row['time'].isoformat()] = {"gre000z0": row['gre000z0']}
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
        self.idaweb_data = data

    def get_range_data(self, workers=None):
        workers = workers or multiprocessing.cpu_count()
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='10T')
        grouped = defaultdict(list)
        for dt in date_range:
            dt_str = dt.isoformat()
            fp = self.datetime_index.get(dt_str)
            if fp:
                grouped[fp].append(dt_str)

        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_file, fp, dts, self.coordinates, self.idaweb_data): fp
                for fp, dts in grouped.items()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing NetCDF files"):
                result = future.result()
                if result:
                    results.extend(result)
        return results


if __name__ == "__main__":
    import time
    start_date = "2023-01-01T00:00:00"
    end_date = "2023-01-01T23:50:00"
    fp_inca = "/home/marta/Projects/tb/data/weather/inca/"
    fp_images = "/home/marta/Projects/tb/data/images/mch/1159/2/2023"
    fp_global_rayonnement = "data/rayonnement_global"

    processor = PreProcessData(start_date, end_date, fp_inca, fp_images, fp_global_rayonnement)

    start_time = time.time()
    processed_data = processor.get_range_data(workers=4)
    print(f"Processed {len(processed_data)} records in {time.time() - start_time:.2f}s")


    data_by_day = defaultdict(list)
    for entry in processed_data:
            dt = entry['datetime']
            day = dt[:10]  # 'YYYY-MM-DD'
            data_by_day[day].append(entry)
    all_days = list(data_by_day.keys())
    all_days = [record for day in all_days for record in data_by_day[day]]

    np.savez("data/data.npz", dole=np.array(all_days))
