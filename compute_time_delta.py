import pandas as pd
import numpy as np
from matplotlib.patches import Patch
MODEL_NUM = 1
MODEL_PATH = f"models/model_{MODEL_NUM}/matches.csv"
df = pd.read_csv(MODEL_PATH)
df = df.sort_values(by="expected_time")
df["expected_time"] = pd.to_datetime(df["expected_time"], errors='coerce')
df["date"] = df["expected_time"].dt.date
# Mark days with 0 values and NaN values differently in the plot

mean_time_diff_by_day = df.groupby("date")["time_difference_sec"].mean() / 60  # convert to minutes
mean_time_diff_by_day = mean_time_diff_by_day.sort_values(ascending=False)
mean_time_diff_by_day = mean_time_diff_by_day.rename("Mean_time_by_day")
print(mean_time_diff_by_day)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Prepare bar colors: gray for NaN, orange for 0, blue for others

ax = mean_time_diff_by_day.plot(kind='bar')
plt.ylabel('Mean Time Difference (min)')
plt.title('Mean Time Difference by Day')

# Calculate and plot global mean
global_mean = df["time_difference_sec"].mean() / 60
plt.axhline(global_mean, color='red', linestyle='--', label='Global Mean (min)')

# Mark NaN values with an "X"
for i, val in enumerate(mean_time_diff_by_day):
    if np.isnan(val):
        ax.scatter(i, 0.5, marker='x', color='black', s=100, zorder=5)  # 0.5 is just above the x-axis

# Custom legend
legend_elements = [
    Patch(facecolor='blue', label='Mean Time Difference By Day'),
    plt.Line2D([0], [0], color='black', marker='x', linestyle='', markersize=10, label='NaN Days (X)'),
    plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Global Mean (min)')
]
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(f"models/model_{MODEL_NUM}/mean_time_diff_by_day.png")