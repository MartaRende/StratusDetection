import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class Metrics:
    def __init__(self, expected, predicted, data, save_path=None, start_date=None, end_date=None, stats_for_month=True):

        self.expected = pd.DataFrame(expected, columns=["nyon", "dole"])
        self.predicted = pd.DataFrame(predicted, columns=["nyon", "dole"])
        self.data = pd.DataFrame(data["dole"])
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.json_normalize(self.data[0])

        self.save_path = save_path
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            metrics_folder = os.path.join(save_path, "metrics")
            os.makedirs(metrics_folder, exist_ok=True)
            self.save_path = metrics_folder
            
        self.stats_for_month = stats_for_month
        if stats_for_month:
            self.path = self.get_month_dir()
        else:
            self.path = self.get_global_dir()

    def get_month_dir(self):
        datetime_list = self.find_datetimes()
        valid_dates = [dt for dt in datetime_list if dt is not None]
        if not valid_dates:
            return self.save_path  # fallback if no dates
        month = str(valid_dates[0])[:7]  # 'YYYY-MM'
        month_dir = os.path.join(self.save_path, month)
        os.makedirs(month_dir, exist_ok=True)
        return month_dir
    def get_global_dir(self):
        if self.save_path:
            return self.save_path
        return os.getcwd()

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
        if self.start_date and self.end_date:
            match = match[
                (pd.to_datetime(match["datetime"]) >= pd.to_datetime(self.start_date)) &
                (pd.to_datetime(match["datetime"]) <= pd.to_datetime(self.end_date))
            ]
        return match["datetime"].iloc[0] if not match.empty else None

    def find_datetimes(self):
        return [
            self.find_datetime(row)
            for _, row in self.expected.iterrows()
        ]

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

    def plot_rmse(self, title="Rmse", xlabel="Datetime", ylabel="rmse"):
        delta = np.sqrt((self.predicted - self.expected) ** 2)
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

        plt.savefig(f"{self.path}/delta.png")
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

        plt.savefig(f"{self.path}/relative_error.png")
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
        df["month"] = df["datetime"].astype(str).str[:7]  # Extract month (YYYY-MM)

        month_dir = os.path.join(self.save_path, df["month"].iloc[0])
        os.makedirs(month_dir, exist_ok=True)

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
        plt.savefig(f"{month_dir}/day_curve_{day}.png")
        plt.close()

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
        plt.savefig(f"{month_dir}/day_curve_diff_{day}.png")
        plt.close()


    def find_unique_days_non_startus(self, stratus_days):
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
        unique_days = df["day"].unique().tolist()
        unique_days = [d for d in unique_days if d not in stratus_days]
        return unique_days
    def plot_random_days(self, num_days=3, exclude_days=None, title="Random Day Curves", xlabel="Hour", ylabel="Value"):
        unique_days = self.find_unique_days_non_startus(exclude_days) 

        if len(unique_days) < num_days:
            print("Not enough days to sample from after exclusion.")
            return

        sampled_days = np.random.choice(unique_days, size=num_days, replace=False)
        for day in sampled_days:
            self.plot_day_curves(day, title=title, xlabel=xlabel, ylabel=ylabel)

    def save_metrics(self):
        month_dir = self.path
        metrics_file = os.path.join(month_dir, "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Accuracy: {self.get_accuracy()}\n")
            f.write(f"Mean Absolute Error: {self.get_mean_absolute_error().tolist()}\n")
            f.write(f"Root Mean Squared Error: {self.get_rmse().tolist()}\n")
            f.write(f"Mean Relative Error: {self.mean_relative_error().tolist()}\n")
        print(f"Metrics saved to {metrics_file}")
    
    def get_rmse_for_specific_days(self, days):
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
       
        # Filter for requested days
        days = [str(day) for sublist in days for day in sublist]
        df_filtered = df[df["day"].isin(days)]
    
        # for i in range(len(days)):
        #     df_filtered = df[df["day"].isin([days[i]])]
        # if df_filtered.empty:
        #     print("No data available for the specified days.")
        #     return {}

        rmse_per_day = {}
        for day in days:
            day_df = df_filtered[df_filtered["day"] == day]
            if day_df.empty:
                continue
            mse_nyon = ((day_df["predicted_nyon"] - day_df["expected_nyon"]) ** 2).mean()
            mse_dole = ((day_df["predicted_dole"] - day_df["expected_dole"]) ** 2).mean()
            rmse_per_day[day] = {
                "nyon": np.sqrt(mse_nyon),
                "dole": np.sqrt(mse_dole)
            }
   
        return rmse_per_day

    def plot_rmse_for_specific_days(self, days):
        if len(days) == 0:
            print("No days provided for RMSE calculation.")
            return
        rmse_per_day = self.get_rmse_for_specific_days(days)
        if not rmse_per_day:
            print("No RMSE data available for the specified days.")
            return

        days_list = list(rmse_per_day.keys())
        rmse_nyon = [rmse_per_day[day]["nyon"] for day in days_list]
        rmse_dole = [rmse_per_day[day]["dole"] for day in days_list]

        plt.figure(figsize=(12, 6))
        plt.plot(days_list, rmse_nyon, marker='o', linestyle='-', label='RMSE Nyon')
        plt.plot(days_list, rmse_dole, marker='x', linestyle='--', label='RMSE Dole')
        plt.xlabel("Days")
        plt.ylabel("RMSE")
        plt.title("RMSE for Specific Days")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.path}/rmse_specific_days_{days}.png")
        plt.close()
        
