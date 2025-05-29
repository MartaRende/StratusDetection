import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Metrics:
    def __init__(self, expected, predicted, data, save_path=None):
        self.expected = pd.DataFrame(expected, columns=["nyon", "dole"])
        self.predicted = pd.DataFrame(predicted, columns=["nyon", "dole"])
        self.data = pd.DataFrame(data["dole"])
        self.data = pd.json_normalize(self.data[0])

        self.save_path = save_path

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            metrics_folder = os.path.join(save_path, "metrics")
            os.makedirs(metrics_folder, exist_ok=True)
            self.save_path = metrics_folder

    def get_correctPredictions(self, tol=20):
        delta = (self.predicted - self.expected).abs()
        correct = ((delta <= tol).all(axis=1)).sum()
        return correct

    def get_accuracy(self, tol=20):
        return self.get_correctPredictions(tol) / len(self.expected)

    def find_datetime(self, expected_row):
        match = self.data[
            (np.abs(self.data["gre000z0_nyon"] - expected_row["nyon"]) <= 1e-6) &
            (np.abs(self.data["gre000z0_dole"] - expected_row["dole"]) <= 1e-6)
        ]
        return match["datetime"].iloc[0] if not match.empty else None

    def find_datetimes(self):
        return [self.find_datetime(row) for _, row in self.expected.iterrows()]

    def print_datetimes(self):
        datetimes = self.find_datetimes()
        for i, dt in enumerate(datetimes):
            if dt:
                print(f"Datetime: {dt} | Predicted: {self.predicted.iloc[i].tolist()} | Expected: {self.expected.iloc[i].tolist()}")
            else:
                print(f"Sample {i}: No matching datetime found")

    def get_mean_absolute_error(self):
        return (self.predicted - self.expected).abs().mean()

    def get_delta_between_expected_and_predicted(self):
        return (self.predicted - self.expected).abs().values.tolist()

    def plot_delta(self, title="Delta between Expected and Predicted", xlabel="Datetime", ylabel="Delta"):
        delta = (self.predicted - self.expected).abs()
        datetime = self.find_datetimes()
        xticks = [dt if i % 6 == 0 else "" for i, dt in enumerate(datetime)]

        plt.figure(figsize=(12, 10))
        for col in delta.columns:
            plt.plot(delta[col], marker="o", linestyle="-", label=col)

        plt.xticks(range(len(datetime)), xticks, rotation=90)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(f"{self.save_path}/delta.png")
        plt.close()

    def get_relative_error(self):
        return ((self.predicted - self.expected).abs() / self.expected.replace(0, np.nan)).fillna(0).values.tolist()

    def plot_relative_error(self, title="Relative Error", xlabel="Datetime", ylabel="Relative Error"):
        rel_error = ((self.predicted - self.expected).abs() / self.expected.replace(0, np.nan)).fillna(0)
        datetime = self.find_datetimes()
        xticks = [dt if i % 6 == 0 else "" for i, dt in enumerate(datetime)]

        plt.figure(figsize=(12, 10))
        for col in rel_error.columns:
            plt.plot(rel_error[col], marker="o", linestyle="-", label=col)

        plt.xticks(range(len(datetime)), xticks, rotation=90)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(f"{self.save_path}/relative_error.png")
        plt.close()

    def get_rmse(self):
        mse = ((self.predicted - self.expected) ** 2).mean()
        return np.sqrt(mse)

    def mean_relative_error(self):
        rel_error = ((self.predicted - self.expected).abs() / self.expected.replace(0, np.nan)).fillna(0)
        return rel_error.mean()

    def plot_day_curves(self, day, title="Day Curves", xlabel="Hour", ylabel="Value"):
        datetime_list = self.find_datetimes()
        df = pd.DataFrame({
            "datetime": datetime_list,
            "expected_nyon": self.expected["nyon"],
            "expected_dole": self.expected["dole"],
            "predicted_nyon": self.predicted["nyon"],
            "predicted_dole": self.predicted["dole"],
        })
        df = df[df["datetime"].notnull()]
        df["day"] = df["datetime"].astype(str).str[:10]
        df["hour"] = df["datetime"].astype(str).str[11:16]

        day_df = df[df["day"] == str(day)]
        if day_df.empty:
            print(f"No aligned data found for day {day}")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(day_df["hour"], day_df["expected_nyon"], "o-", label="Expected Nyon")
        plt.plot(day_df["hour"], day_df["predicted_nyon"], "x--", label="Predicted Nyon")
        plt.plot(day_df["hour"], day_df["expected_dole"], "o-", label="Expected Dole")
        plt.plot(day_df["hour"], day_df["predicted_dole"], "x--", label="Predicted Dole")
        plt.title(f"{title} - {day}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/day_curve_{day}.png")
        plt.close()

        # Plot differences
        plt.figure(figsize=(12, 6))
        plt.plot(day_df["hour"], (day_df["expected_nyon"] - day_df["predicted_nyon"]), "o-", label="Nyon Difference")
        plt.plot(day_df["hour"], (day_df["expected_dole"] - day_df["predicted_dole"]), "x--", label="Dole Difference")
        plt.title(f"{title} - {day} (Difference)")
        plt.xlabel(xlabel)
        plt.ylabel("Difference")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/day_curve_diff_{day}.png")
        plt.close()
