import numpy as np
import pandas as pd

def read_txt_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

data = np.load("data/complete_data.npz", allow_pickle=True)
df = pd.DataFrame(data['dole'])
df = pd.json_normalize(df[0])

idaweb_data = read_txt_file("data/rayonnement_global/geneva.txt")

# Convert the list to a DataFrame
idaweb_df = pd.DataFrame(idaweb_data)
# Split the single column into multiple columns using the delimiter ';'
idaweb_df = idaweb_df[0].str.split(';', expand=True)

# Rename the columns based on the header row (if available)
idaweb_df.columns = idaweb_df.iloc[2]  # Assuming the third row contains column names
idaweb_df = idaweb_df.drop(index=2)  # Drop the header row from the data

# Reset the index
idaweb_df = idaweb_df.reset_index(drop=True)
# filter idaweb_df to have only rows where stn == 'dol'
idaweb_df = idaweb_df[idaweb_df['gre0'] == 'DOL']

idaweb_df = idaweb_df[['time', 'prestas0']]
# Transform 'time' column from 'YYYYMMDDHHMM' to 'YYYY-MM-DDTHH:MM:SS'
idaweb_df['time'] = pd.to_datetime(idaweb_df['time'], format='%Y%m%d%H%M').dt.strftime('%Y-%m-%dT%H:%M:%S')
# Ensure both DataFrames have datetime columns in the same format
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
idaweb_df['time'] = pd.to_datetime(idaweb_df['time'], errors='coerce')

# Merge pressure data into df based on datetime
df = df.merge(idaweb_df.rename(columns={'time': 'datetime', 'prestas0': 'pres'}), on='datetime', how='left')
if 'pres_x' in df.columns and 'pres_y' in df.columns:
    df['pres'] = df['pres_y']  # Keep 'pres_y' or choose 'pres_x'
    df = df.drop(columns=['pres_x', 'pres_y'])  # Drop the duplicates
import ipdb
ipdb.set_trace()
df['pres'] = pd.to_numeric(df['pres'], errors='coerce')
npz_file = "data/complete_data_pres.npz"
np.savez_compressed(npz_file, dole=df.to_dict(orient='records'))
# load the npz file to check if it worked
npz_data = np.load(npz_file, allow_pickle=True)

import ipdb
ipdb.set_trace()

