import ipdb
import numpy as np
from datetime import datetime

import numpy as np


data = np.load("data/complete_data.npz", allow_pickle=True)
data_test = np.load("models/model_9/test_data.npz", allow_pickle=True)
stats = np.load("models/model_9/stats.npz", allow_pickle=True)

ipdb.set_trace()