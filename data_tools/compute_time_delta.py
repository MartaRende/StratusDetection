import pandas as pd
import numpy as np
from matplotlib.patches import Patch
MODEL_NUM = 2
MODEL_PATH = f"models/model_{MODEL_NUM}/matches.csv"
df = pd.read_csv(MODEL_PATH)
df = df.sort_values(by="expected_time")
df["expected_time"] = pd.to_datetime(df["expected_time"], errors='coerce')
df["date"] = df["expected_time"].dt.date
# Mark days with 0 values and NaN values differently in the plot


mean_time_diff_by_day = (
    df.groupby("date")["time_difference_sec"].sum() /
    df.groupby("date")["num_datetimes_for_day"].first() / 60  # convert to minutes
)
mean_time_diff_by_day = mean_time_diff_by_day.sort_values(ascending=False)
mean_time_diff_by_day = mean_time_diff_by_day.rename("Mean_time_by_day")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Prepare bar colors: gray for NaN, orange for 0, blue for others

ax = mean_time_diff_by_day.plot(kind='bar')
plt.ylabel('Mean Time Difference (min)')
plt.title('Mean Time Difference by Day')

# Calculate and plot global mean
global_mean = df["time_difference_sec"].sum() / df["num_datetimes_for_day"].sum() / 60

# Compute per-row mean (in minutes), then take the median of those
per_row_mean = df["time_difference_sec"] / df["num_datetimes_for_day"] / 60
global_median = per_row_mean.median()
plt.axhline(global_mean, color='red', linestyle='--', label='Global Mean (min)')
plt.axhline(global_median, color='orange', linestyle='--', label='Global Median (min)')

# Mark NaN values with an "X"
for i, val in enumerate(mean_time_diff_by_day):
    if np.isnan(val):
        ax.scatter(i, 0.5, marker='x', color='black', s=100, zorder=5)  # 0.5 is just above the x-axis

# Custom legend
legend_elements = [
    Patch(facecolor='blue', label='Mean Time Difference By Day'),
    plt.Line2D([0], [0], color='black', marker='x', linestyle='', markersize=10, label='NaN Days (X)'),
    plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Global Mean (min)'),
    plt.Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Global Median (min)')
]
plt.legend(handles=legend_elements)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MODEL_NUM = 2
MODEL_PATH = f"models/model_{MODEL_NUM}/matches.csv"
df = pd.read_csv(MODEL_PATH)

# Seleziona solo le colonne numeriche rilevanti per la heatmap
corr_cols = [
    'expected_confidence', 
    'predicted_confidence',
    'time_difference_sec',
    'confidence_similarity',
    'combined_score',
    'expected_slope',
    'predicted_slope',
    'expected_z_score',
    'predicted_z_score'
]

# Filtra solo i match completi (escludi unmatched)
matched = df[df['match_status'] == 'matched'].copy()

# Calcola la matrice di correlazione
corr_matrix = matched[corr_cols].corr()

# Crea la heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={'label': 'Coefficiente di Correlazione'}
)

# Aggiungi titolo e formattazione
plt.title('Heatmap delle Correlazioni tra Variabili dei Picchi\n(Solo matched peaks)', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Salva la figura
plt.savefig(f"models/model_{MODEL_NUM}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
plt.tight_layout()
plt.savefig(f"models/model_{MODEL_NUM}/mean_time_diff_by_day.png")