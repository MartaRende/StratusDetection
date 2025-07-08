import numpy as np
import pandas as pd

# Load the test data
test_data_path = "models/model_28/test_data.npz"
data = np.load(test_data_path, allow_pickle=True)
data = data['dole']

# Convert the dictionary-like data into a DataFrame
data = pd.DataFrame(data.tolist())

# Drop columns ending with '_t1' or '_t2'
filtered_data = data.drop(columns=[c for c in data.columns if c.endswith('_t1') or c.endswith('_t2')])

# Rename columns to remove '_t0' suffix
filtered_data.columns = [c[:-3] if c.endswith('_t0') else c for c in filtered_data.columns]

# Save the filtered DataFrame to a new npz file
filtered_data_npz_path = "models/model_28/test_data.npz"
np.savez(filtered_data_npz_path, dole=filtered_data.to_dict('records'))

