import csv
import pandas as pd
import ipdb

file_path = 'data/rayonnement_global/dole.txt'  # Replace with your actual file path
df_dole = pd.read_csv(file_path, delimiter=';')

file_path = 'data/rayonnement_global/nyon.txt'  # Replace with your actual file path
df_nyon = pd.read_csv(file_path, delimiter=';')

file_path = 'data/rayonnement_global/geneva.txt'  # Replace with your actual file path
df_geneva = pd.read_csv(file_path, delimiter=';')
df_dole['location'] = 'dole'
df_nyon['location'] = 'nyon'
df_geneva['location'] = 'geneva'

import ipdb
ipdb.set_trace()  # Set a breakpoint here to inspect the DataFrame