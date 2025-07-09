import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict
from PIL import Image
from scipy import stats
import seaborn as sns

from .config import PlotConfig
# Plotting utilities are implemented with the help of Deepseek AI
class Plotter:
    """Handles all plotting functionality for the Metrics class"""
    
    def __init__(self, metrics):
        self.metrics = metrics
        self.plot_config = PlotConfig()

    def plot_error_metrics(self, days: List[str], metric_type: str = "rmse", 
                         prefix: str = "stratus_days", subdirectory=None) -> None:
        """Plot specified error metrics for given days"""
        valid_metrics = ["mae", "rmse", "relative_error"]
        if metric_type not in valid_metrics:
            raise ValueError(f"metric_type must be one of {valid_metrics}")

        day_metrics = self.metrics.get_metrics_for_days(days)
        if not day_metrics:
            self.metrics.logger.warning("No data available for plotting")
            return

        days_list = sorted(day_metrics.keys())
        geneva_values = [day_metrics[day][metric_type]["geneva"] for day in days_list]
        dole_values = [day_metrics[day][metric_type]["dole"] for day in days_list]

        fig, ax = plt.subplots(figsize=self.plot_config.figsize)

        ax.plot(days_list, geneva_values, 
                marker='o', linestyle='-', 
                color=self.plot_config.colors["geneva"],
                markersize=self.plot_config.marker_size,
                linewidth=self.plot_config.line_width,
                label='Geneva')
                
        ax.plot(days_list, dole_values, 
                marker='x', linestyle='--', 
                color=self.plot_config.colors["dole"],
                markersize=self.plot_config.marker_size,
                linewidth=self.plot_config.line_width,
                label='Dole')

        ax.set_title(f"{metric_type.upper()} for Specific Days", 
                    fontsize=self.plot_config.fontsize["title"])
        ax.set_xlabel("Date", fontsize=self.plot_config.fontsize["labels"])
        ax.set_ylabel(metric_type.upper(), fontsize=self.plot_config.fontsize["labels"])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
      
        plt.tight_layout()
        if self.metrics.save_path:
            output_path = os.path.join(
                subdirectory if subdirectory else self.metrics.save_path, 
                f"{metric_type}_specific_days_{prefix}.png"
            )
            plt.savefig(output_path, dpi=self.plot_config.dpi, bbox_inches='tight')
            self.metrics.logger.info(f"Saved plot to {output_path}")
        plt.close()

    def plot_day_curves(self, days: List[str]) -> None:
        """Plot comparison curves for multiple specific days with images"""
        days = self.metrics._normalize_days_input(days)
        for day in days:
            day_df = self.metrics._prepare_day_metrics([day])
       
            if day_df.empty:
                self.metrics.logger.warning(f"No data found for day {day}")
                continue
     
            day_df = day_df.drop_duplicates(subset=["datetime"])
            month = day_df["month"].iloc[0]
            month_dir = os.path.join(self.metrics.save_path, month)
            os.makedirs(month_dir, exist_ok=True)
            print(f"Plotting day curves for {day} in {month_dir}")
            day_datetimes = day_df["datetime"].tolist()
            num_images = min(6, len(day_datetimes))
            indices = np.linspace(0, len(day_datetimes) - 1, num_images, dtype=int) if num_images > 1 else [0]

            fig = plt.figure(figsize=(self.plot_config.figsize[0], self.plot_config.figsize[1] * 1.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

            ax1 = fig.add_subplot(gs[0])

            for var in ["geneva", "dole"]:
                ax1.plot(day_df["hour"], day_df[f"expected_{var}"],
                        'o-', color=self.plot_config.colors[var],
                        markersize=self.plot_config.marker_size,
                        linewidth=self.plot_config.line_width,
                        label=f'Expected {var.capitalize()}')

                ax1.plot(day_df["hour"], day_df[f"predicted_{var}"],
                        'x--', color=self.plot_config.colors[var],
                        markersize=self.plot_config.marker_size,
                        linewidth=self.plot_config.line_width,
                        label=f'Predicted {var.capitalize()}')

            ax1.set_title(f"Day Curves - {day} - Prediction in {self.metrics.prediction_minutes} minutes", 
                         fontsize=self.plot_config.fontsize["title"])
            ax1.set_ylabel("Radiation (W/m²)", fontsize=self.plot_config.fontsize["labels"])
            ax1.set_xlabel("Hours", fontsize=self.plot_config.fontsize["labels"])
            ax1.legend()
            ax1.grid(True)

            times = pd.to_datetime(day_df["hour"], format="%H:%M").dt.time
            ax1.set_xticks([
                t.strftime("%H:%M") for t in times if pd.Timestamp(t.strftime("%H:%M")).minute % 10 == 0
            ])
            ax1.set_xticklabels([
                t.strftime("%H:%M") for t in times if pd.Timestamp(t.strftime("%H:%M")).minute % 10 == 0
            ], rotation=45)

            ax2 = fig.add_subplot(gs[1])
            ax2.axis('off')

            if num_images > 0:
                img_width = 1.0 / num_images
                for i, idx in enumerate(indices):
                    dt = day_datetimes[idx]
                    img = self.metrics.get_image_for_datetime(dt)
                 
                    if np.all(img == 0):
                        self.metrics.logger.warning(f"Image for {dt} is completely black.")
                    else:
                        if img.max() - img.min() < 1e-3:
                            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

                    if img.ndim == 2:
                        img = np.stack([img] * 3, axis=-1)
                    left = i * img_width
                    ax_img = fig.add_axes([left, -0.1, img_width, 0.25])
                    ax_img.imshow(img)
                    ax_img.set_title(dt.strftime("%H:%M"), fontsize=8)
                    ax_img.axis('off')

            plt.tight_layout()
            plt.savefig(
                os.path.join(month_dir, f"day_curve_{day}.png"),
                dpi=self.plot_config.dpi,
                bbox_inches='tight'
            )
            plt.close()

    def plot_delta_absolute_error(self, days: List[str], prefix: str = "stratus_days", subdirectory=None) -> None:
        """Plot absolute error of delta (geneva-dole) for given days"""
        df = self.metrics._prepare_day_metrics(days)
        if df.empty:
            self.metrics.logger.warning("No data found for the provided days.")
            return

        if "month" not in df.columns:
            df["month"] = df["datetime"].dt.strftime("%Y-%m")
        months = df["month"].unique()

        for month in months:
            month_df = df[df["month"] == month]
            delta_abs_error = ((month_df["predicted_geneva"] - month_df["predicted_dole"]) -
                             (month_df["expected_geneva"] - month_df["expected_dole"])).abs()

            fig, ax = plt.subplots(figsize=self.plot_config.figsize)
            x_vals = np.arange(len(month_df))
            dates_labels = month_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")

            ax.plot(x_vals, delta_abs_error[month_df.index],
                    'o-', color='red',
                    markersize=self.plot_config.marker_size,
                    linewidth=self.plot_config.line_width,
                    label='Absolute Error (Geneva - Dole)')

            step = max(1, len(x_vals) // 10)
            ax.set_xticks(x_vals[::step])
            ax.set_xticklabels(dates_labels[::step], rotation=45)

            ax.set_title(f"Absolute Error of Delta (Geneva-Dole) - {month}",
                         fontsize=self.plot_config.fontsize["title"])
            ax.set_xlabel("Date", fontsize=self.plot_config.fontsize["labels"])
            ax.set_ylabel("Absolute Error", fontsize=self.plot_config.fontsize["labels"])
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            if self.metrics.save_path:
                output_path = os.path.join(
                    subdirectory, f"delta_absolute_error_{prefix}.png"
                )
                plt.savefig(output_path, dpi=self.plot_config.dpi, bbox_inches='tight')
                self.metrics.logger.info(f"Saved delta absolute error plot to {output_path}")
            plt.close()

    def plot_delta_scatter(self, days: List[str], prefix: str = "delta_comparison", subdirectory: str = None) -> None:
        """Scatter plot comparing expected vs predicted deltas"""
        if not days:
            df = self.metrics._create_comparison_dataframe()
        else:
            days = self.metrics._normalize_days_input(days)
            df = self.metrics._prepare_day_metrics(days)
        
        if df.empty:
            self.metrics.logger.warning("No data found for the provided days.")
            return
        if prefix == "delta_comparison":
            df["expected_delta"] = df["expected_geneva"] - df["expected_dole"]
            df["predicted_delta"] = df["predicted_geneva"] - df["predicted_dole"]
            residuals = df["predicted_delta"] - df["expected_delta"]
            outlier_threshold = 1.5 * np.std(residuals)
            df["is_outlier"] = np.abs(residuals) > outlier_threshold
        elif prefix == "geneva":
            df["expected_delta"] = df["expected_geneva"]
            df["predicted_delta"] = df["predicted_geneva"]
            residuals = df["predicted_delta"] - df["expected_delta"]
            outlier_threshold = 1.5 * np.std(residuals)
            df["is_outlier"] = np.abs(residuals) > outlier_threshold
        elif prefix == "dole":
            df["expected_delta"] = df["expected_dole"]
            df["predicted_delta"] = df["predicted_dole"]
            residuals = df["predicted_delta"] - df["expected_delta"]
            outlier_threshold = 2.5 * np.std(residuals)
            df["is_outlier"] = np.abs(residuals) > outlier_threshold

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df["expected_delta"], df["predicted_delta"]
        )
        mae = np.mean(np.abs(residuals))
        
        plt.figure(figsize=(12, 10))
        plt.scatter(
            df["expected_delta"], 
            df["predicted_delta"],
            alpha=0.6,
            color='blue',
            label='Normal points'
        )
        
        outliers = df[df["is_outlier"]]
        plt.scatter(
            outliers["expected_delta"],
            outliers["predicted_delta"],
            alpha=0.8,
            color='red',
            label='Outliers'
        )
        
        for _, row in outliers.iterrows():
            plt.annotate(
                str(row["datetime"].date()) if "datetime" in row else str(row.name),
                xy=(row["expected_delta"], row["predicted_delta"]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='red'
            )
        
    
        max_val = max(df["expected_delta"].max(), df["predicted_delta"].max())
        min_val = min(df["expected_delta"].min(), df["predicted_delta"].min())
        plt.plot(
            [min_val, max_val], 
            [min_val, max_val], 
            color='green',
            linestyle=':',
            label='Perfect fit'
        )
        
        plt.title(f"Expected vs Predicted Delta (Geneva - Dole)\nOutliers labeled with date", 
                fontsize=self.plot_config.fontsize["title"])
        plt.xlabel("Expected Delta (W/m²)", fontsize=self.plot_config.fontsize["labels"])
        plt.ylabel("Predicted Delta (W/m²)", fontsize=self.plot_config.fontsize["labels"])
        plt.legend(fontsize=self.plot_config.fontsize["labels"])
        plt.grid(True, linestyle='--', alpha=0.3)
        
        stats_text = (
            f"MAE: {mae:.2f}\n"
            f"Outliers: ({len(outliers)/len(df)*100:.1f}%)\n"
        )
        plt.annotate(
            stats_text,
            xy=(0.05, 0.75),
            xycoords='axes fraction',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=self.plot_config.fontsize.get("annotations", 10)
        )
        
        plt.tight_layout()
        if self.metrics.save_path:
            output_path = os.path.join(
                subdirectory if subdirectory else self.metrics.save_path,
                f"{prefix}_scatter_outliers.png"
            )
            if not days:
                output_path = os.path.join(
                    subdirectory if subdirectory else self.metrics.save_path,
                    f"{prefix}_scatter_all.png"
                )
            else:
                output_path = os.path.join(
                    subdirectory if subdirectory else self.metrics.save_path,
                    f"{prefix}_scatter_stratus.png"
                )
            plt.savefig(output_path, dpi=self.plot_config.dpi, bbox_inches='tight')
        print(f"Saved delta scatter plot to {output_path}")
        plt.close()
    
    def plot_residual_errors(self, days, prefix: str = "residual_errors", subdirectory: str = None) -> None:
        """
        Plot residual errors for expected vs predicted deltas (Geneva - Dole).
        
        Args:
            days: List of days to include in the plot
            prefix: Prefix for filename
            subdirectory: Optional subdirectory to save plot
        """
        # Prepare data
        if not days:
            # If days is empty, use all available days in the data
            df = self.metrics_create_comparison_dataframe()
        else:
            days = self.metrics._normalize_days_input(days)
            df = self.metrics._prepare_day_metrics(days)
        if df.empty:
            self.logger.warning("No data found for the provided days.")
            return
        
        # Calculate deltas and residuals
        df["expected_delta"] = df["expected_geneva"] - df["expected_dole"]
        df["predicted_delta"] = df["predicted_geneva"] - df["predicted_dole"]
        df["residual"] = df["predicted_delta"] - df["expected_delta"]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Residuals scatter plot
        # Scatter plot of residuals
        plt.scatter(
            df["expected_delta"], 
            df["residual"],
            alpha=0.6,
            color='blue',
            label='Residuals'
        )
        # Histogram of residuals on a secondary y-axis
        ax = plt.gca()
        ax_hist = ax.twinx()
        ax_hist.hist(
            df["residual"],
            bins=30,
            color='orange',
            alpha=0.3,
            label='Residuals Histogram'
        )
        ax_hist.set_ylabel("Count", fontsize=self.plot_config.fontsize["labels"])
        ax_hist.legend(loc='upper right', fontsize=self.plot_config.fontsize["labels"])
        
        # Horizontal line at zero residual
        plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
        
        # Format plot
        plt.title(f"Residual Errors (Predicted - Expected Delta) - Stratus Days\n", 
                fontsize=self.plot_config.fontsize["title"])
        plt.xlabel("Expected Delta (W/m²)", fontsize=self.plot_config.fontsize["labels"])
        plt.ylabel("Residual Error (W/m²)", fontsize=self.plot_config.fontsize["labels"])
        plt.legend(fontsize=self.plot_config.fontsize["labels"])
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        if self.metrics.save_path:
            output_path = os.path.join(
                subdirectory if subdirectory else self.metrics.save_path,
                f"{prefix}_residuals_all.png"
            )
            plt.savefig(output_path, dpi=self.plot_config.dpi, bbox_inches='tight')
            print(f"Saved residual errors plot to {output_path}")
        plt.close()

    def plot_delta_error_heatmap(self, days: List[str], prefix: str = "delta_heatmap", subdirectory: str = None) -> None:
        """Plot heatmap of absolute delta errors per day and hour"""
        df = self.metrics._prepare_day_metrics(days)
        if df.empty:
            self.metrics.logger.warning("No data found for the provided days.")
            return

        # Calculate delta error
        df["expected_delta"] = df["expected_geneva"] - df["expected_dole"]
        df["predicted_delta"] = df["predicted_geneva"] - df["predicted_dole"]
        df["delta_abs_error"] = (df["predicted_delta"] - df["expected_delta"]).abs()

        # Extract day and hour (as strings)
        df["date"] = df["datetime"].dt.date.astype(str)
        df["hour"] = df["datetime"].dt.hour

        # Create pivot table: rows=days, columns=hours
        heatmap_data = df.pivot_table(
            index="date",
            columns="hour",
            values="delta_abs_error",
            aggfunc="mean"
        )

        # Plot heatmap
        plt.figure(figsize=(14, 6))
    
        sns.heatmap(
            heatmap_data,
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor='gray',
            vmin=0,
            vmax=400,
        )
        plt.title("Heatmap of Absolute Delta Error (Geneva - Dole) per Day and Hour", fontsize=self.plot_config.fontsize["title"])
        plt.xlabel("Hour of Day", fontsize=self.plot_config.fontsize["labels"])
        plt.ylabel("Day", fontsize=self.plot_config.fontsize["labels"])
        plt.tight_layout()
        # Plot mean of delta_abs_error across all days and hours
        mean_error = df["delta_abs_error"].mean()
        plt.axhline(mean_error, color='gray', linestyle='--', linewidth=1.5, label=f"Mean Delta Error: {mean_error:.2f} W/m²")
        plt.legend(loc='upper right', fontsize=self.plot_config.fontsize.get("labels", 10))
        # Save the file
        if self.metrics.save_path:
            output_path = os.path.join(
            subdirectory if subdirectory else self.metrics.save_path,
            f"{prefix}.png"
            )
            plt.savefig(output_path, dpi=self.plot_config.dpi, bbox_inches='tight')
            self.metrics.logger.info(f"Saved delta error heatmap to {output_path}")
        print(f"Saved delta error heatmap to {output_path}")
        plt.close()
    def plot_max_delta_jump_delay(self, days: List[str], prefix: str = "delta_jump_delay", subdirectory: str = None) -> None:
        """
        Plot the delay (in minutes) between the largest delta change (jump) time
        in expected vs predicted radiation deltas.
            """

            
        df = self.metrics._prepare_day_metrics(days)
        if df.empty:
            self.metrics.logger.warning("No data found for the provided days.")
            return

        df["expected_delta"] = df["expected_dole"] - df["expected_geneva"]
        df["predicted_delta"] = df["predicted_dole"] - df["predicted_geneva"]
        df["date"] = df["datetime"].dt.date

        daily_durations = []

        for day, group in df.groupby("date"):
            if group.empty:
                continue
            group = group.sort_values("datetime").reset_index(drop=True)

            # --- Expected
            exp_max_idx = group["expected_delta"].idxmax()
            exp_max_time = group.loc[exp_max_idx, "datetime"]
            expected_after_max = group[group["datetime"] > exp_max_time]
            if expected_after_max.empty:
                continue
            exp_min_idx = expected_after_max["expected_delta"].idxmin()
            exp_min_time = group.loc[exp_min_idx, "datetime"]
            exp_duration = (exp_min_time - exp_max_time).total_seconds() / 60

            # --- Predicted
            pred_max_idx = group["predicted_delta"].idxmax()
            pred_max_time = group.loc[pred_max_idx, "datetime"]
            predicted_after_max = group[group["datetime"] > pred_max_time]
            if predicted_after_max.empty:
                continue
            pred_min_idx = predicted_after_max["predicted_delta"].idxmin()
            pred_min_time = group.loc[pred_min_idx, "datetime"]
            pred_duration = (pred_min_time - pred_max_time).total_seconds() / 60

            # --- Error
            duration_error = pred_duration - exp_duration
            valid = pred_duration >= 0 and exp_duration >= 0

            daily_durations.append({
                "date": day,
                "expected_duration_min": exp_duration,
                "predicted_duration_min": pred_duration,
                "duration_error_min": duration_error,
                "expected_max_time": exp_max_time,
                "expected_min_time": exp_min_time,
                "predicted_max_time": pred_max_time,
                "predicted_min_time": pred_min_time,
                "valid_prediction": valid
            })


        duration_df = pd.DataFrame(daily_durations)
       
        global_mean_error = duration_df["duration_error_min"].mean()
        n_invalid = (~duration_df["valid_prediction"]).sum()

        print("Daily duration errors:")
        print(duration_df)
        print(f"\nGlobal mean error: {global_mean_error:.2f} min — Invalid matches: {n_invalid}")

        # --- Plot
        plt.figure(figsize=(14, 6))
        sns.barplot(
            data=duration_df,
            x="date",
            y="duration_error_min",
            hue="valid_prediction",
            palette={True: "skyblue", False: "salmon"},
            edgecolor="black"
        )
        plt.axhline(0, color="gray", linestyle="--", linewidth=1.5)
        plt.axhline(global_mean_error, color="blue", linestyle="--", linewidth=1.5,
                    label=f"Global Mean Error: {global_mean_error:.2f} min")

        plt.title("Peak-Valley Duration Error (Predicted - Expected)", fontsize=self.plot_config.fontsize["title"])
        plt.xlabel("Day", fontsize=self.plot_config.fontsize["labels"])
        plt.ylabel("Duration Error (min)", fontsize=self.plot_config.fontsize["labels"])
        plt.xticks(rotation=45)
        plt.legend(title="Valid Prediction")
        plt.tight_layout()

        # Save
        if self.metrics.save_path:
            output_path = os.path.join(
                subdirectory if subdirectory else self.metrics.save_path,
                f"{prefix}.png"
            )
            plt.savefig(output_path, dpi=self.plot_config.dpi, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
        plt.close()