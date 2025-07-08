import numpy as np
import pandas as pd

# def read_txt_file(filepath):
#     with open(filepath, 'r') as file:
#         lines = file.readlines()
#     return [line.strip() for line in lines]

# data = np.load("data/complete_data.npz", allow_pickle=True)
# df = pd.DataFrame(data['dole'])
# df = pd.json_normalize(df[0])

# idaweb_data = read_txt_file("data/rayonnement_global/geneva.txt")

# # Convert the list to a DataFrame
# idaweb_df = pd.DataFrame(idaweb_data)
# import ipdb
# ipdb.set_trace()
# # Split the single column into multiple columns using the delimiter ';'
# idaweb_df = idaweb_df[0].str.split(';', expand=True)

# # Rename the columns based on the header row (if available)
# idaweb_df.columns = idaweb_df.iloc[2]  # Assuming the third row contains column names
# idaweb_df = idaweb_df.drop(index=2)  # Drop the header row from the data

# # Reset the index
# idaweb_df = idaweb_df.reset_index(drop=True)
# # filter idaweb_df to have only rows where stn == 'dol'
# idaweb_df = idaweb_df.drop(index=[0, 1]).reset_index(drop=True)
# idaweb_df = idaweb_df[['time', 'gre000z0']]
# # Transform 'time' column from 'YYYYMMDDHHMM' to 'YYYY-MM-DDTHH:MM:SS'
# idaweb_df['time'] = pd.to_datetime(idaweb_df['time'], format='%Y%m%d%H%M').dt.strftime('%Y-%m-%dT%H:%M:%S')
# # Ensure both DataFrames have datetime columns in the same format
# df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
# idaweb_df['time'] = pd.to_datetime(idaweb_df['time'], errors='coerce')

# # Merge pressure data into df based on datetime
# df = df.merge(idaweb_df.rename(columns={'time': 'datetime', 'gre000z0': 'gre000z0_gen'}), on='datetime', how='left')

# import ipdb
# ipdb.set_trace()

npz_file = "data/complete_data_gen.npz"
# np.savez_compressed(npz_file, dole=df.to_dict(orient='records'))
# load the npz file to check if it worked
npz_data = np.load(npz_file, allow_pickle=True)
df_gen = pd.DataFrame(npz_data['dole'])
df_gen= pd.json_normalize(df_gen[0])

import ipdb
ipdb.set_trace()
df_gen['gre000z0_nyon'] = df_gen['gre000z0_gen']
df_gen['gre000z0_dole'] = pd.to_numeric(df_gen['gre000z0_dole'], errors='coerce')
df_gen['gre000z0_nyon'] = pd.to_numeric(df_gen['gre000z0_nyon'], errors='coerce')
np.savez_compressed(npz_file, dole=df_gen.to_dict(orient='records'))
import ipdb
ipdb.set_trace()

