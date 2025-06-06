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



    def plot_absolute_error(self, title="Absolute Error", xlabel="Datetime", ylabel="Absolute Error"):
        abs_error = (self.predicted - self.expected).abs()
        datetimes = self.find_datetimes()
        
        plt.figure(figsize=(12, 6))
        for col in abs_error.columns:
            plt.plot(abs_error[col], marker="o", linestyle="-", label=col)

        # Adjust x-axis ticks using datetimes
        xticks = [dt if i % 12 == 0 else "" for i, dt in enumerate(datetimes)]
        plt.xticks(range(len(datetimes)), xticks, rotation=45)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.path}/absolute_error.png")
        plt.close()

    def get_relative_error(self):
        return ((self.predicted - self.expected).abs() / self.expected.replace(0, np.nan)).fillna(0).values.tolist()

    def plot_relative_error(self, title="Relative Error", xlabel="Datetime", ylabel="Relative Error"):
        rel_error = ((self.predicted - self.expected).abs() / self.expected.replace(0, np.nan)).fillna(0)
        datetime = self.find_datetimes()
        xticks = [dt if i % 12 == 0 else "" for i, dt in enumerate(datetime)]

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

    def get_rmse(self, var_1=None, var_2=None):
        if var_1 is None or var_2 is None:
            var_1 = self.predicted
            var_2 = self.expected
        mse = ((var_1 - var_2) ** 2).mean()
        return np.sqrt(mse)

    def mean_relative_error(self, var_1=None, var_2=None):
        if var_1 is None or var_2 is None:
            var_1 = self.predicted
            var_2 = self.expected
        rel_error = ((var_1 - var_2).abs() / var_2.replace(0, np.nan)).fillna(0)
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
        if isinstance(days, list) and len(days) > 0 and isinstance(days[0], list):
            # Flatten if 2D
            days = [item for sublist in days for item in sublist]
        else:
            days = [str(day) for day in days]
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
    def get_absolute_error_for_specific_days(self, days):
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

        # Flatten days if needed
        if isinstance(days, list) and len(days) > 0 and isinstance(days[0], list):
            days = [item for sublist in days for item in sublist]
        else:
            days = [str(day) for day in days]
        df_filtered = df[df["day"].isin(days)]

        abs_error_per_day = {}
        for day in days:
            day_df = df_filtered[df_filtered["day"] == day]
            if day_df.empty:
                continue
            abs_error_nyon = (day_df["predicted_nyon"] - day_df["expected_nyon"]).abs().mean()
            abs_error_dole = (day_df["predicted_dole"] - day_df["expected_dole"]).abs().mean()
            abs_error_per_day[day] = {
                "nyon": abs_error_nyon,
                "dole": abs_error_dole
            }
        return abs_error_per_day
   
    
    def get_relative_error_for_specific_days(self, days):
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

        # Flatten days if needed
        if isinstance(days, list) and len(days) > 0 and isinstance(days[0], list):
            days = [item for sublist in days for item in sublist]
        else:
            days = [str(day) for day in days]
        df_filtered = df[df["day"].isin(days)]

        rel_error_per_day = {}
        for day in days:
            day_df = df_filtered[df_filtered["day"] == day]
            if day_df.empty:
                continue
            # Avoid division by zero
            rel_error_nyon = ((day_df["predicted_nyon"] - day_df["expected_nyon"]).abs() / day_df["expected_nyon"].replace(0, np.nan)).fillna(0).mean()
            rel_error_dole = ((day_df["predicted_dole"] - day_df["expected_dole"]).abs() / day_df["expected_dole"].replace(0, np.nan)).fillna(0).mean()
            
            rel_error_per_day[day] = {
                "nyon": rel_error_nyon,
                "dole": rel_error_dole
            }
        return rel_error_per_day

    def plot_relative_error_for_specific_days(self, days, stratus_days="stratus_days_relative"):
        if len(days) == 0:
            print("No days provided for relative error calculation.")
            return
        rel_error_per_day = self.get_relative_error_for_specific_days(days)
        if not rel_error_per_day:
            print("No relative error data available for the specified days.")
            return

        days_list = list(rel_error_per_day.keys())
        rel_nyon = [rel_error_per_day[day]["nyon"] for day in days_list]
        rel_dole = [rel_error_per_day[day]["dole"] for day in days_list]

        plt.figure(figsize=(12, 6))
        plt.plot(days_list, rel_nyon, marker='o', linestyle='-', label='Relative Error Nyon')
        plt.plot(days_list, rel_dole, marker='x', linestyle='--', label='Relative Error Dole')
        plt.xlabel("Days")
        plt.ylabel("Relative Error")
        plt.title("Relative Error for Specific Days")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.path}/relative_error_specific_days_{stratus_days}.png")
        plt.close()
    
    def get_delta_btw_nyon_dole(self):
        delta_predicted = (self.predicted["nyon"] - self.predicted["dole"]).abs()
        delta_expected = (self.expected["nyon"] - self.expected["dole"]).abs()
        return delta_predicted, delta_expected
        
    def get_global_rmse_for_specific_days(self, days):
        rmse_per_day = self.get_rmse_for_specific_days(days)
        if not rmse_per_day:
            return {"nyon": None, "dole": None}
        nyon_rmse = [v["nyon"] for v in rmse_per_day.values() if v["nyon"] is not None]
        dole_rmse = [v["dole"] for v in rmse_per_day.values() if v["dole"] is not None]
        delta_predicted, delta_expected = self.get_delta_btw_nyon_dole()
        # Calculate RMSE for delta
        delta_rmse = np.sqrt(np.mean((delta_predicted - delta_expected) ** 2))
        global_rmse = {
            "nyon": np.mean(nyon_rmse) if nyon_rmse else None,
            "dole": np.mean(dole_rmse) if dole_rmse else None,
            "delta": delta_rmse
        }

        return global_rmse
    def get_global_relative_error_for_specific_days(self, days):
        rel_error_per_day = self.get_relative_error_for_specific_days(days)
        if not rel_error_per_day:
            return {"nyon": None, "dole": None}
        nyon_rel_error = [v["nyon"] for v in rel_error_per_day.values() if v["nyon"] is not None]
        dole_rel_error = [v["dole"] for v in rel_error_per_day.values() if v["dole"] is not None]
        delta_predicted, delta_expected = self.get_delta_btw_nyon_dole()
        # Calculate relative error for delta
        rel_error = [abs((pred - exp) / exp) if exp != 0 else 0 for pred, exp in zip(delta_predicted, delta_expected)]
        global_rel_error = {
            "nyon": np.mean(nyon_rel_error) if nyon_rel_error else None,
            "dole": np.mean(dole_rel_error) if dole_rel_error else None,
            "delta": np.mean(rel_error) if rel_error else None
        }
        return global_rel_error
    def save_metrics(self, stratus_days=None, non_stratus_days=None):
        month_dir = self.path
        metrics_file = os.path.join(month_dir, "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Accuracy: {self.get_accuracy()}\n")
            f.write(f"Mean Absolute Error: {self.get_mean_absolute_error().tolist()}\n")
            f.write(f"Root Mean Squared Error: {self.get_rmse().tolist()}\n")
            f.write(f"Mean Relative Error: {self.mean_relative_error().tolist()}\n")
            if stratus_days:
                f.write(f"Rmse stratus days: {self.get_global_rmse_for_specific_days(stratus_days)}\n")
                f.write(f"Relative Error stratus days: {self.get_global_relative_error_for_specific_days(stratus_days)}\n")
            if non_stratus_days:
                f.write(f"Rmse non-stratus days: {self.get_global_rmse_for_specific_days(non_stratus_days)}\n")
                f.write(f"Relative Error non-stratus days: {self.get_global_relative_error_for_specific_days(non_stratus_days)}\n")
        print(f"Metrics saved to {metrics_file}")
    def get_mae_for_specific_days(self, days):
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

        # Flatten days if needed
        if isinstance(days, list) and len(days) > 0 and isinstance(days[0], list):
            days = [item for sublist in days for item in sublist]
        else:
            days = [str(day) for day in days]
        df_filtered = df[df["day"].isin(days)]

        mae_per_day = {}
        for day in days:
            day_df = df_filtered[df_filtered["day"] == day]
            if day_df.empty:
                continue
            mae_nyon = (day_df["predicted_nyon"] - day_df["expected_nyon"]).abs().mean()
            mae_dole = (day_df["predicted_dole"] - day_df["expected_dole"]).abs().mean()
            mae_per_day[day] = {
                "nyon": mae_nyon,
                "dole": mae_dole
            }
        return mae_per_day
    def get_global_mae_for_specific_days(self, days):
        mae_per_day = self.get_mae_for_specific_days(days)
        if not mae_per_day:
            return {"nyon": None, "dole": None}
        nyon_mae = [v["nyon"] for v in mae_per_day.values() if v["nyon"] is not None]
        dole_mae = [v["dole"] for v in mae_per_day.values() if v["dole"] is not None]
        global_mae = {
            "nyon": np.mean(nyon_mae) if nyon_mae else None,
            "dole": np.mean(dole_mae) if dole_mae else None
        }
        return global_mae
    def compute_and_save_metrics_by_month_for_days(self, days, label="stratus_days"):
        # Flatten days
        if isinstance(days, list) and len(days) > 0 and isinstance(days[0], list):
            days = [item for sublist in days for item in sublist]
        else:
            days = [str(day) for day in days]

        # Group days by month
        from collections import defaultdict
        month_day_map = defaultdict(list)
        for day in days:
            month = day[:7]  # "YYYY-MM"
            month_day_map[month].append(day)

        for month, month_days in month_day_map.items():
            # Define output path for this month
            month_dir = os.path.join(self.save_path, month)
            os.makedirs(month_dir, exist_ok=True)
            output_file = os.path.join(month_dir, f"metrics_{label}.txt")

            # Compute metrics
            rmse = self.get_global_rmse_for_specific_days(month_days)
            rel_error = self.get_global_relative_error_for_specific_days(month_days)
            mae = self.get_global_mae_for_specific_days(month_days)
            with open(output_file, "w") as f:
                f.write(f"Metrics for {label} - {month}\n")
                f.write(f"Days: {month_days}\n")
                f.write(f"Global RMSE: {rmse}\n")
                f.write(f"Global Relative Error: {rel_error}\n")
                f.write(f"Mean Absolute Error: {mae}\n")

            print(f"Saved {label} metrics for {month} to {output_file}")

    def plot_rmse_for_specific_days(self, days, stratus_days="stratus_days"):
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
        plt.savefig(f"{self.path}/rmse_specific_days_{stratus_days}.png")
        plt.close()

    def get_absolute_error(self):
        return (self.predicted - self.expected).abs().mean().tolist()


    
    
    def plot_delta_btw_nyon_dole(self, title="Delta between Nyon and Dole", xlabel="Datetime", ylabel="Delta"):
        delta_predicted, delta_expected = self.get_delta_btw_nyon_dole()
        datetime_list = self.find_datetimes()
        # Show day only once per day on x-axis
        prev_day = None
        xticks = []
        for i, dt in enumerate(datetime_list):
            day = str(dt)[:10] if dt else ""
            if day != prev_day:
                xticks.append(day)
                prev_day = day
            else:
                xticks.append("")
        plt.figure(figsize=(12, 6))
        plt.plot(delta_predicted, marker='o', linestyle='-', label='Predicted Delta')
        plt.plot(delta_expected, marker='x', linestyle='--', label='Expected Delta')
        plt.xticks(range(len(datetime_list)), xticks, rotation=45)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.path}/delta_nyon_dole.png")
        plt.close()
    
    def plot_absolute_error_delta_btw_nyon_dole(self, title="Absolute Error Delta between Nyon and Dole", xlabel="Datetime", ylabel="Absolute Error Delta"):
        delta_predicted, delta_expected = self.get_delta_btw_nyon_dole()
        abs_error_delta = (delta_predicted - delta_expected).abs()
        datetime_list = self.find_datetimes()
        # Show day only once per day on x-axis
        prev_day = None
        xticks = []
        for i, dt in enumerate(datetime_list):
            day = str(dt)[:10] if dt else ""
            if day != prev_day:
                xticks.append(day)
                prev_day = day
            else:
                xticks.append("")
        plt.figure(figsize=(12, 6))
        plt.plot(abs_error_delta, marker='o', linestyle='-', label='Absolute Error Delta')
        plt.xticks(range(len(datetime_list)), xticks, rotation=45)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.path}/absolute_error_delta_nyon_dole.png")
        plt.close()