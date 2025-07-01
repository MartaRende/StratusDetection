import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional, Union, Dict, List, Tuple
from functools import lru_cache
import logging
from dataclasses import dataclass
from collections import defaultdict
from PIL import Image

@dataclass
class PlotConfig:
    """Centralized configuration for plotting parameters"""
    style: str = 'seaborn'
    figsize: Tuple[int, int] = (24, 8)
    dpi: int = 300
    fontsize: Dict[str, int] = None
    colors: Dict[str, str] = None
    marker_size: int = 6
    line_width: int = 2
    
    def __post_init__(self):
        self.fontsize = self.fontsize or {'title': 14, 'labels': 12, 'ticks': 10}
        self.colors = self.colors or {'delta': '#1f77b4'}

class Metrics:
    """
    Comprehensive metrics calculator for comparing predicted and expected delta values
    (difference between nyon and dole) with datetime alignment and visualization capabilities.
    """
    
    def __init__(self, 
                 expected: Union[List, np.ndarray, pd.Series],
                 predicted: Union[List, np.ndarray, pd.Series],
                 data: Dict,
                 save_path: Optional[str] = None,
                 fp_images: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 num_views: int = 1,
                 stats_for_month: bool = True,
                 tolerance: float = 20.0,
                 plot_config: Optional[PlotConfig] = None,
                 log_level: int = logging.INFO):
        """
        Initialize the DeltaMetrics calculator.
        
        Args:
            expected: Ground truth delta values (nyon - dole)
            predicted: Predicted delta values
            data: Raw data dictionary containing 'dole' key with datetime info
            save_path: Base directory to save results
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            stats_for_month: Whether to organize results by month
            tolerance: Tolerance threshold for accuracy calculation
            plot_config: Configuration for plotting parameters
            log_level: Logging level (default: INFO)
        """
        self._initialize_data(expected, predicted, data)
        self._setup_datetime_filters(start_date, end_date)
        self._setup_paths(save_path)
        self._initialize_configurations(stats_for_month, tolerance, plot_config, log_level)
        self.image_base_folder = fp_images if fp_images else ""
        self.num_views = num_views

    def _initialize_data(self, expected, predicted, data):
        """Initialize and normalize data structures"""
        self.expected = pd.DataFrame(expected, columns=["nyon", "dole"])
        self.predicted = pd.DataFrame(predicted, columns=["delta"])
        
        # Normalize the input data
        self.data = pd.json_normalize(pd.DataFrame(data["dole"])[0])
        
        # Convert to appropriate dtypes
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        self.data["gre000z0_nyon"] = pd.to_numeric(self.data["gre000z0_nyon"])
        self.data["gre000z0_dole"] = pd.to_numeric(self.data["gre000z0_dole"])
        
        # Pre-compute numpy arrays for faster operations
        self._nyon_values = self.data["gre000z0_nyon"].to_numpy()
        self._dole_values = self.data["gre000z0_dole"].to_numpy()
        self._datetime_values = self.data["datetime"].to_numpy()
      
    def get_image_for_datetime(self, dt, view=2):
        date_str = dt.strftime('%Y-%m-%d')
        time_str = dt.strftime('%H%M')
        img_filename = f"1159_{view}_{date_str}_{time_str}.jpeg"
        img_path = os.path.join(self.image_base_folder, str(view),dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'), img_filename)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img) # Normalize to [0, 1]
            return img_array
        else:
            print(f"Image not found for {dt}: {img_path}")
            return []
        
    def _setup_datetime_filters(self, start_date, end_date):
        """Initialize datetime range filters"""
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
    def _setup_paths(self, save_path):
        """Initialize directory structure for saving results"""
        self.save_path = save_path
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            metrics_folder = os.path.join(save_path, "metrics")
            os.makedirs(metrics_folder, exist_ok=True)
            self.save_path = metrics_folder
            
    def _initialize_configurations(self, stats_for_month, tolerance, plot_config, log_level):
        """Initialize various configurations"""
        self.stats_for_month = stats_for_month
        self.tolerance = tolerance
        self.plot_config = plot_config if plot_config else PlotConfig()
        self._datetime_cache = None
        
        # Configure logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
     
    @property
    def datetime_list(self):
        """Cached property for datetime list"""
        if self._datetime_cache is None:
            self._datetime_cache = self._compute_datetime_list()
        return self._datetime_cache
        
    def _compute_datetime_list(self):
        """Compute datetime list with vectorized operations"""
        datetimes = []
        
        # Convert expected to numpy for faster access
        exp_nyon = self.expected["nyon"].to_numpy()
        exp_dole = self.expected["dole"].to_numpy()
        
        for nyon, dole in zip(exp_nyon, exp_dole):
            # Vectorized comparison
            mask = (
                np.isclose(self._nyon_values, nyon, atol=1e-6) & 
                np.isclose(self._dole_values, dole, atol=1e-6)
            )
            
            # Apply date range filter if specified
            if self.start_date and self.end_date:
                date_mask = (
                    (self._datetime_values >= self.start_date) & 
                    (self._datetime_values <= self.end_date)
                )
                mask = mask & date_mask
            
            matches = self._datetime_values[mask]
            datetimes.append(matches[0] if len(matches) > 0 else None)
  
        return datetimes

    def get_correct_predictions(self, tol: Optional[float] = None) -> int:
        """
        Count predictions within tolerance of expected values.

        Args:
            tol: Tolerance threshold (uses class default if None)

        Returns:
            Number of correct predictions
        """
        tolerance = tol if tol is not None else self.tolerance
        exp = (self.expected["dole"] - self.expected["nyon"]).to_numpy().flatten()
        if len(self.predicted) != len(self.expected):
            self.logger.warning("Mismatch in lengths of predicted and expected values.")
            return 0  # Return 0 correct predictions if lengths do not match

        return (np.abs(self.predicted.to_numpy().flatten() - exp) <= tolerance).sum()

    def get_accuracy(self, tol: Optional[float] = None) -> float:
        """Calculate accuracy within given tolerance"""
        tolerance = tol if tol is not None else self.tolerance
        correct_predictions = self.get_correct_predictions(tol)
        total_predictions = len(self.expected)

        if total_predictions == 0:
            self.logger.warning("No expected values available for accuracy calculation.")
            return 0.0  # Return 0 accuracy if there are no expected values

        return correct_predictions / total_predictions
    def exp_converted(self) -> pd.Series:
        """
        Convert expected values to delta (dole - nyon) format.
        
        Returns:
            Series of expected delta values
        """
        self.expected["delta"] = self.expected["dole"] - self.expected["nyon"]
        return self.expected
    def get_mean_absolute_error(self) -> float:
        """Calculate MAE for delta values"""
        return np.abs(self.predicted['delta'] / self.exp_converted()['delta']).mean()

    def get_root_mean_squared_error(self) -> float:
        """Calculate RMSE for delta values"""
        mse = ((self.predicted['delta'] - self.exp_converted()['delta']) ** 2).mean()
        return np.sqrt(mse)

    def get_relative_error(self) -> List[float]:
        """Calculate relative error for delta values"""
        abs_error = abs(self.predicted['delta'] - self.exp_converted()['delta'])
        rel_error = abs_error / self.exp_converted()['delta'].replace(0, np.nan)
        return rel_error.fillna(0).tolist()

    def get_mean_relative_error(self) -> float:
        """Calculate mean relative error for delta values"""
        return np.mean(self.get_relative_error())

    def _create_comparison_dataframe(self) -> pd.DataFrame:
        """Create a combined dataframe with all comparison data"""
        return pd.DataFrame({
            "datetime": np.array(self.datetime_list).flatten(),
            "expected_delta": (self.expected["dole"] / self.expected["nyon"]).to_numpy().flatten(),
            "predicted_delta": self.predicted.to_numpy().flatten(),
        }).dropna(subset=["datetime"])

    def _prepare_day_metrics(self, days) -> pd.DataFrame:
        """
        Prepare a dataframe filtered by specific days with extracted date parts.
        
        Args:
            days: List of days in format 'YYYY-MM-DD'
            
        Returns:
            Filtered dataframe with date parts
        """
        days = self._normalize_days_input(days)
        df = self._create_comparison_dataframe()
        df["date_str"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df["hour"] = df["datetime"].dt.strftime("%H:%M")
        df["month"] = df["datetime"].dt.strftime("%Y-%m")
        # Filter for requested days
        return df[df["date_str"].isin(days)]

    def get_metrics_for_days(self, days) -> Dict[str, Dict[str, float]]:
        """
        Calculate all metrics (MAE, RMSE, Relative Error) for specific days.
        
        Args:
            days: List of days in format 'YYYY-MM-DD'
            
        Returns:
            Dictionary of metrics for each day
        """
        day_df = self._prepare_day_metrics(days)

        if day_df.empty:
            self.logger.warning(f"No data found for days: {days}")
            return {}

        metrics = {}
        for day, group in day_df.groupby("date_str"):
            metrics[day] = {
                "mae": abs(group["predicted_delta"] - group["expected_delta"]).mean(),
                "rmse": np.sqrt(((group["predicted_delta"] - group["expected_delta"]) ** 2).mean()),
                "relative_error": (abs(group["predicted_delta"] - group["expected_delta"]) / 
                                 group["expected_delta"].replace(0, np.nan)).fillna(0).mean()
            }
     
        return metrics 

    def get_global_metrics_for_days(self, days: List[str]) -> Dict[str, float]:
        """
        Calculate global metrics (averaged across all specified days).
        
        Args:
            days: List of days in format 'YYYY-MM-DD'
            
        Returns:
            Dictionary of global metrics
        """
        day_metrics = self.get_metrics_for_days(days)
        if not day_metrics:
            return {}

        # Initialize aggregators
        global_metrics = {
            "mae": [],
            "rmse": [],
            "relative_error": []
        }

        # Collect all values
        for metrics in day_metrics.values():
            for metric_type in global_metrics:
                global_metrics[metric_type].append(metrics[metric_type])

        # Calculate means
        return {
            metric_type: np.mean(vals) if vals else None
            for metric_type, vals in global_metrics.items()
        }

    def plot_error_metrics(self, days: List[str], metric_type: str = "rmse", 
                          prefix: str = "stratus_days", subdirectory = None) -> None:
        """
        Plot specified error metrics for given days.
        
        Args:
            days: List of days to plot
            metric_type: Type of metric to plot ('mae', 'rmse', or 'relative_error')
            prefix: Prefix for filename
        """
        valid_metrics = ["mae", "rmse", "relative_error"]
        if metric_type not in valid_metrics:
            raise ValueError(f"metric_type must be one of {valid_metrics}")

        day_metrics = self.get_metrics_for_days(days)
        if not day_metrics:
            self.logger.warning("No data available for plotting")
            return

        # Prepare data
        days_list = sorted(day_metrics.keys())
        delta_values = [day_metrics[day][metric_type] for day in days_list]

        # Create plot
        fig, ax = plt.subplots(figsize=self.plot_config.figsize)
        
        ax.plot(days_list, delta_values, 
                marker='o', linestyle='-', 
                color=self.plot_config.colors["delta"],
                markersize=self.plot_config.marker_size,
                linewidth=self.plot_config.line_width,
                label='Delta (Nyon-Dole)')

        # Format plot
        ax.set_title(f"{metric_type.upper()} for Specific Days", 
                    fontsize=self.plot_config.fontsize["title"])
        ax.set_xlabel("Date", fontsize=self.plot_config.fontsize["labels"])
        ax.set_ylabel(metric_type.upper(), fontsize=self.plot_config.fontsize["labels"])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
      
        # Save plot
        plt.tight_layout()
        if self.save_path:
            output_path = os.path.join(
                subdirectory if subdirectory else self.save_path, 
                f"{metric_type}_specific_days_{prefix}.png"
            )
            plt.savefig(output_path, dpi=self.plot_config.dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot to {output_path}")
        plt.close()

    def plot_day_curves(self, days: List[str]) -> None:
        """
        Plot comparison curves for multiple specific days with corresponding images displayed at the bottom.

        Args:
            days: List of dates in format 'YYYY-MM-DD'
        """
        if not days:
            self.logger.warning("No days provided for plotting day curves")
            return

        # Flatten and normalize days input
        days = self._normalize_days_input(days)

        for day in days:
            day_df = self._prepare_day_metrics([day])
            if day_df.empty:
                self.logger.warning(f"No data found for day {day}")
                continue

            # Ensure datetimes are unique (keep first occurrence)
            day_df = day_df.drop_duplicates(subset=["datetime"])
            month = day_df["month"].iloc[0]
            month_dir = os.path.join(self.save_path, month)
            os.makedirs(month_dir, exist_ok=True)

            # Get datetimes for this day
            day_datetimes = day_df["datetime"].tolist()
            # Plot only a subset of images (e.g., 6 evenly spaced images)
            num_images = min(6, len(day_datetimes))
            if num_images > 1:
                indices = np.linspace(0, len(day_datetimes) - 1, num_images, dtype=int)
            else:
                indices = [0]
     
            # Create figure with subplots: curves on top, images at the bottom
            fig = plt.figure(figsize=(self.plot_config.figsize[0], self.plot_config.figsize[1] * 1.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

            # Top subplot for curves
            ax1 = fig.add_subplot(gs[0])

            ax1.plot(day_df["hour"], day_df["expected_delta"],
                    'o-', color=self.plot_config.colors["delta"],
                    markersize=self.plot_config.marker_size,
                    linewidth=self.plot_config.line_width,
                    label='Expected Delta')

            ax1.plot(day_df["hour"], day_df["predicted_delta"],
                    'x--', color=self.plot_config.colors["delta"],
                    markersize=self.plot_config.marker_size,
                    linewidth=self.plot_config.line_width,
                    label='Predicted Delta')

            ax1.set_title(f"Day Curves - {day}", fontsize=self.plot_config.fontsize["title"])
            ax1.set_ylabel("Delta Radiation (W/m²)", fontsize=self.plot_config.fontsize["labels"])
            ax1.set_xlabel("Hours", fontsize=self.plot_config.fontsize["labels"])
            ax1.legend()
            ax1.grid(True)

            # Set x-axis ticks every 10 minutes
            times = pd.to_datetime(day_df["hour"], format="%H:%M").dt.time
            ax1.set_xticks([
                t.strftime("%H:%M") for t in times if pd.Timestamp(t.strftime("%H:%M")).minute % 10 == 0
            ])
            ax1.set_xticklabels([
                t.strftime("%H:%M") for t in times if pd.Timestamp(t.strftime("%H:%M")).minute % 10 == 0
            ], rotation=45)

            # Bottom subplot for images using get_image_for_datetime
            ax2 = fig.add_subplot(gs[1])
            ax2.axis('off')

            # Display images horizontally at the bottom
            if num_images > 0:
                img_width = 1.0 / num_images
                for i, idx in enumerate(indices):
                    dt = day_datetimes[idx]
                    
                    img = self.get_image_for_datetime(dt)
                    if np.all(img == 0):
                        self.logger.warning(f"Image for {dt} is completely black.")
                    else:
                        if img.max() - img.min() < 1e-3:
                            print(f"Warning: Image for {dt} has very low dynamic range.")
                            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

                    if img.ndim == 2:
                        print(f"Converting grayscale image for {dt} to RGB.")
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

            # Original difference plot (unchanged)
            fig, ax = plt.subplots(figsize=self.plot_config.figsize)
            ax.plot(day_df["hour"],
                    (day_df["expected_delta"] - day_df["predicted_delta"]),
                    'o-', color=self.plot_config.colors["delta"],
                    markersize=self.plot_config.marker_size,
                    linewidth=self.plot_config.line_width,
                    label='Delta Difference')

            ax.set_title(f"Day Curves - {day} (Difference)",
                        fontsize=self.plot_config.fontsize["title"])
            ax.set_xlabel("Hour", fontsize=self.plot_config.fontsize["labels"])
            ax.set_ylabel("Difference", fontsize=self.plot_config.fontsize["labels"])
            ax.legend()
            ax.grid(True)

            # Set x-axis ticks every 10 minutes for the difference plot
            ax.set_xticks([
                t.strftime("%H:%M") for t in times if pd.Timestamp(t.strftime("%H:%M")).minute % 10 == 0
            ])
            ax.set_xticklabels([
                t.strftime("%H:%M") for t in times if pd.Timestamp(t.strftime("%H:%M")).minute % 10 == 0
            ], rotation=45)

            plt.tight_layout()
            plt.savefig(
                os.path.join(month_dir, f"day_curve_diff_{day}.png"),
                dpi=self.plot_config.dpi,
                bbox_inches='tight'
            )
            plt.close()

    def plot_delta_absolute_error(self, days: List[str], prefix: str = "stratus_days", subdirectory = None) -> None:
        """
        Plot absolute error of delta for the given days, saving in the corresponding month directory.

        Args:
            days: List of days in format 'YYYY-MM-DD'
            prefix: Prefix for filename
        """
        df = self._prepare_day_metrics(days)
        if df.empty:
            self.logger.warning("No data found for the provided days.")
            return

        # Compute absolute error of delta
        delta_abs_error = abs(df["predicted_delta"] - df["expected_delta"])

        # Group by month for saving in month directory
        if "month" not in df.columns:
            df["month"] = df["datetime"].dt.strftime("%Y-%m")
        months = df["month"].unique()

        for month in months:
            month_df = df[df["month"] == month]
            month_delta_abs_error = delta_abs_error[month_df.index]

            # Plot each value (not mean) for the month
            fig, ax = plt.subplots(figsize=self.plot_config.figsize)
            x_vals = np.arange(len(month_df))
            dates_labels = month_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")

            ax.plot(x_vals, month_delta_abs_error[month_df.index],
                    'o-', color='red',
                    markersize=self.plot_config.marker_size,
                    linewidth=self.plot_config.line_width,
                    label='Absolute Error (Delta)')

            # Tick every N points
            step = max(1, len(x_vals) // 10)
            ax.set_xticks(x_vals[::step])
            ax.set_xticklabels(dates_labels[::step], rotation=45)

            ax.set_title(f"Absolute Error of Delta - {month}",
                         fontsize=self.plot_config.fontsize["title"])
            ax.set_xlabel("Date", fontsize=self.plot_config.fontsize["labels"])
            ax.set_ylabel("Absolute Error", fontsize=self.plot_config.fontsize["labels"])
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()

            if self.save_path:
                output_path = os.path.join(
                    subdirectory, f"delta_absolute_error_{prefix}.png"
                )
                plt.savefig(output_path, dpi=self.plot_config.dpi, bbox_inches='tight')
                self.logger.info(f"Saved delta absolute error plot to {output_path}")
            plt.close()

    def save_metrics_report(self, stratus_days: Optional[List[str]] = None, 
                        non_stratus_days: Optional[List[str]] = None) -> None:
        """
        Save comprehensive metrics report.
        """
        report_lines = [
            "=== Delta Metrics Report ===",
            f"Accuracy (tolerance={self.tolerance}): {self.get_accuracy():.4f}",
            f"Mean Absolute Error: {self.get_mean_absolute_error():.4f}",
            f"Root Mean Squared Error: {self.get_root_mean_squared_error():.4f}",
            f"Mean Relative Error: {self.get_mean_relative_error():.4f}",
        ]

        if stratus_days:
            stratus_metrics = self.get_global_metrics_for_days(stratus_days)
            
            report_lines.extend([
                "\n=== Stratus Days Metrics ===",
                f"Days: {stratus_days}",
                f"Global RMSE: {stratus_metrics.get('rmse', None)}",
                f"Global Relative Error: {stratus_metrics.get('relative_error', None)}",
                f"Global MAE: {stratus_metrics.get('mae', None)}",
            ])

        if non_stratus_days:
            non_stratus_metrics = self.get_global_metrics_for_days(non_stratus_days)
            
            report_lines.extend([
                "\n=== Non-Stratus Days Metrics ===",
                f"Days: {non_stratus_days}",
                f"Global RMSE: {non_stratus_metrics.get('rmse', None)}",
                f"Global Relative Error: {non_stratus_metrics.get('relative_error', None)}",
                f"Global MAE: {non_stratus_metrics.get('mae', None)}",
            ])

        if self.save_path:
            report_path = os.path.join(self.save_path, "delta_metrics_report.txt")
            with open(report_path, 'w') as f:
                f.write("\n".join(report_lines))
            self.logger.info(f"Saved metrics report to {report_path}")

    def compute_and_save_metrics_by_month(self, days: List[str], label: str = "stratus_days") -> None:
        """
        Compute and save metrics organized by month.
        """
        # Flatten and validate days input
        days = self._normalize_days_input(days)
        if not days:
            self.logger.warning("No days provided for monthly metrics")
            return

        # Group days by month
        month_day_map = defaultdict(list)
        for day in days:
            month = day[:7]  # "YYYY-MM"
            month_day_map[month].append(day)

        # Process each month
        for month, month_days in month_day_map.items():
            month_dir = os.path.join(self.save_path, month)
            os.makedirs(month_dir, exist_ok=True)
            
            # Compute all metrics
            metrics = self.get_global_metrics_for_days(month_days)
            
            # Save report
            report_path = os.path.join(month_dir, f"delta_metrics_{label}.txt")
            with open(report_path, 'w') as f:
                f.write(f"Metrics for {label} - {month}\n")
                f.write(f"Days: {month_days}\n")
                f.write(f"Global RMSE: {metrics.get('rmse', None)}\n")
                f.write(f"Global Relative Error: {metrics.get('relative_error', None)}\n")
                f.write(f"Global MAE: {metrics.get('mae', None)}\n")
                
            self.logger.info(f"Saved {label} metrics for {month} to {report_path}")

            # Plot metrics in the month directory
            self.plot_error_metrics(month_days, metric_type="rmse", prefix=label, subdirectory=month_dir)
            self.plot_error_metrics(month_days, metric_type="relative_error", prefix=label, subdirectory=month_dir)
            self.plot_delta_absolute_error(month_days, prefix=label, subdirectory=month_dir)

    def _normalize_days_input(self, days) -> List[str]:
        """Helper to normalize days input to a flat list of strings"""
        if isinstance(days, np.ndarray):
            if days.ndim > 1:
                days = days.flatten()
            days = days.tolist()
        elif isinstance(days, (list, tuple)):
            # Flatten nested lists/tuples
            if any(isinstance(d, (list, tuple, np.ndarray)) for d in days):
                days = [item for sublist in days 
                    for item in (sublist if isinstance(sublist, (list, tuple, np.ndarray)) 
                    else [sublist])]
        days = [str(d) for d in days] if days else []
        return days