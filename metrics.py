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
from scipy import stats
from scipy.signal import find_peaks

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
        self.colors = self.colors or {'geneva': '#1f77b4', 'dole': '#ff7f0e'}

class Metrics:
    """
    Comprehensive metrics calculator for comparing predicted and expected values
    with datetime alignment and visualization capabilities.
    
    Features:
    - Accurate datetime alignment between predictions and ground truth
    - Multiple error metrics (MAE, RMSE, Relative Error, Accuracy)
    - Flexible visualization options
    - Automatic directory structure for saving results
    - Caching for improved performance
    - Comprehensive logging
    """
    
    def __init__(self, 
                 expected: Union[List, np.ndarray, pd.DataFrame],
                 predicted: Union[List, np.ndarray, pd.DataFrame],
                 data: Dict,
                 save_path: Optional[str] = None,
                 fp_images: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 num_views: int = 1,
                 prediction_minutes: int = 10,
                 stats_for_month: bool = True,
                 tolerance: float = 20.0,
                 plot_config: Optional[PlotConfig] = None,
                 log_level: int = logging.INFO):
        """
        Initialize the Metrics calculator.
        
        Args:
            expected: Ground truth values with columns ["geneva", "dole"]
            predicted: Predicted values with same structure as expected
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
        self.test = 0
        self.prediction_minutes = prediction_minutes

    def _initialize_data(self, expected, predicted, data):
        """Initialize and normalize data structures"""
        self.expected = pd.DataFrame(expected, columns=["geneva", "dole"])
        self.predicted = pd.DataFrame(predicted, columns=["geneva", "dole"])
        
        # Normalize the input data
        self.data = pd.json_normalize(pd.DataFrame(data["dole"])[0])
        
        # Convert to appropriate dtypes
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        self.data["gre000z0_geneva"] = pd.to_numeric(self.data["gre000z0_nyon"])
        self.data["gre000z0_dole"] = pd.to_numeric(self.data["gre000z0_dole"])
       
        # Pre-compute numpy arrays for faster operations
        self._geneva_values = self.data["gre000z0_geneva"].to_numpy()
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
        exp_geneva= self.expected["geneva"].to_numpy()
        exp_dole = self.expected["dole"].to_numpy()
        
        for geneva, dole in zip(exp_geneva, exp_dole):
            # Vectorized comparison
            mask = (
                np.isclose(self._geneva_values, geneva, atol=1e-6) & 
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
        delta = (self.predicted - self.expected).abs()
        return ((delta <= tolerance).all(axis=1)).sum()

    def get_accuracy(self, tol: Optional[float] = None) -> float:
        """Calculate accuracy within given tolerance"""
        return self.get_correct_predictions(tol) / len(self.expected)

    def get_mean_absolute_error(self) -> Dict[str, float]:
        """Calculate MAE for each variable"""
        return (self.predicted - self.expected).abs().mean().to_dict()

    def get_root_mean_squared_error(self) -> Dict[str, float]:
        """Calculate RMSE for each variable"""
        mse = ((self.predicted - self.expected) ** 2).mean()
        return np.sqrt(mse).to_dict()

    def get_relative_error(self) -> Dict[str, List[float]]:
        """Calculate relative error for each variable"""
        abs_error = (self.predicted - self.expected).abs()
        rel_error = abs_error / self.expected.replace(0, np.nan)

        return rel_error.fillna(0).to_dict(orient='list')

    def get_mean_relative_error(self) -> Dict[str, float]:
        """Calculate mean relative error for each variable"""
        rel_error = self.get_relative_error()
        return {k: np.mean(v) for k, v in rel_error.items()}

   
    def get_delta_btw_geneva_dole(self) -> pd.DataFrame:
        """
        Calculate the difference between geneva and Dole values.
        
        Returns:
            DataFrame with datetime, expected, predicted values and their differences
        """
        df = self._create_comparison_dataframe()
        df["expected_delta_geneva_dole"] = df["expected_geneva"] - df["expected_dole"]
        df["predicted_delta_geneva_dole"] = df["predicted_geneva"] - df["predicted_dole"]
        return df.dropna(subset=["datetime"])
    def get_delta_stats(self) -> Dict[str, float]:
        """
        Compute MAE, RMSE, and mean relative error for delta_dole_geneva.
        Returns:
            Dictionary with keys: 'mae', 'rmse', 'mean_relative_error'
        """
        df = self.get_delta_btw_geneva_dole()
        expected_delta = df["expected_delta_geneva_dole"]
        predicted_delta = df["predicted_delta_geneva_dole"]
        abs_error = (predicted_delta - expected_delta).abs()
        mae = abs_error.mean()
        rmse = np.sqrt(((predicted_delta - expected_delta) ** 2).mean())
        # Avoid division by zero
        rel_error = abs_error / expected_delta.abs().replace(0, np.nan)
        mean_rel_error = rel_error.fillna(0).mean()
    
        return {
            "mae": mae,
            "rmse": rmse,
            "mean_relative_error": mean_rel_error
        }
        
    def _create_comparison_dataframe(self) -> pd.DataFrame:
        """Create a combined dataframe with all comparison data"""
        return pd.DataFrame({
            "datetime": self.datetime_list,
            "expected_geneva": self.expected["geneva"],
            "expected_dole": self.expected["dole"],
            "predicted_geneva": self.predicted["geneva"],
            "predicted_dole": self.predicted["dole"],
        }).dropna(subset=["datetime"])

    def _prepare_day_metrics(self, days) -> pd.DataFrame:
        """
        Prepare a dataframe filtered by specific days with extracted date parts.
        
        Args:
            days: List of days in format 'YYYY-MM-DD'
            
        Returns:
            Filtered dataframe with date parts
        """
        
        if isinstance(days, np.ndarray):
            days = days.flatten().tolist()
        elif isinstance(days, (list, tuple)):
            # Flatten nested lists/tuples
            days = [item for sublist in days for item in (sublist if isinstance(sublist, (list, tuple, np.ndarray)) else [sublist])]
            days = [str(d) for d in days]
        else:
            days = [str(days)]
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
        
        # Flatten days if not already flat
        if isinstance(days, np.ndarray):
            days = days.flatten().tolist()
        elif isinstance(days, (list, tuple)):
            # Flatten nested lists/tuples
            days = [item for sublist in days for item in (sublist if isinstance(sublist, (list, tuple, np.ndarray)) else [sublist])]
            days = [str(d) for d in days]
        else:
            days = [str(days)]
        day_df = self._prepare_day_metrics(days)

        if day_df.empty:
            self.logger.warning(f"No data found for days: {days}")
            return {}

        metrics = {}
        for day, group in day_df.groupby("date_str"):
            metrics[day] = {
                "mae": {
                    "geneva": (group["predicted_geneva"] - group["expected_geneva"]).abs().mean(),
                    "dole": (group["predicted_dole"] - group["expected_dole"]).abs().mean(),
                },
                "rmse": {
                    "geneva": np.sqrt(((group["predicted_geneva"] - group["expected_geneva"]) ** 2).mean()),
                    "dole": np.sqrt(((group["predicted_dole"] - group["expected_dole"]) ** 2).mean()),
                },
                
                "relative_error": {
                    "geneva": ((group["predicted_geneva"] - group["expected_geneva"]).abs() / 
                            group["expected_geneva"].replace(0, np.nan)).fillna(0).mean(),
                    "dole": ((group["predicted_dole"] - group["expected_dole"]).abs() / 
                            group["expected_dole"].replace(0, np.nan)).fillna(0).mean(),
                }
            }
  
        return metrics 

    def get_global_metrics_for_days(self, days: List[str]) -> Dict[str, Dict[str, float]]:
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
            "mae": {"geneva": [], "dole": []},
            "rmse": {"geneva": [], "dole": []},
            "relative_error": {"geneva": [], "dole": []}
        }

        # Collect all values
        for metrics in day_metrics.values():
            for metric_type in global_metrics:
                for var in ["geneva", "dole"]:
                    global_metrics[metric_type][var].append(metrics[metric_type][var])

        # Calculate means
        return {
            metric_type: {
                var: np.mean(vals) if vals else None
                for var, vals in values.items()
            }
            for metric_type, values in global_metrics.items()
        }

    def plot_error_metrics(self, days: List[str], metric_type: str = "rmse", 
                          prefix: str = "stratus_days",subdirectory = None) -> None:
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
        geneva_values = [day_metrics[day][metric_type]["geneva"] for day in days_list]
        dole_values = [day_metrics[day][metric_type]["dole"] for day in days_list]

        # Create plot
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
        print(f"Plotting day curves for days: {days}")
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

            ax1.set_title(f"Day Curves - {day} - Prediction in {self.prediction_minutes} minutes", fontsize=self.plot_config.fontsize["title"])
            ax1.set_ylabel("Radiation (W/m²)", fontsize=self.plot_config.fontsize["labels"])
            ax1.set_xlabel("Hours", fontsize=self.plot_config.fontsize["labels"])
            ax1.legend()
            ax1.grid(True)

            # Set x-axis ticks every 10 minutes
            # Convert "hour" column to datetime.time for proper sorting and tick placement
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
                    # Check if the image is completely black (all zeros)
                    if np.all(img == 0):
                        self.logger.warning(f"Image for {dt} is completely black.")
                    else:
                        # Optional: normalize for visibility if dynamic range is too low
                        if img.max() - img.min() < 1e-3:
                            print(f"Warning: Image for {dt} has very low dynamic range.")
                            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

                    # Ensure RGB format if needed
                    if img.ndim == 2:
                        print(f"Converting grayscale image for {dt} to RGB.")
                        img = np.stack([img] * 3, axis=-1)# Place each image in its own axes, horizontally aligned at the bottom
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

          
    def plot_delta_absolute_error(self, days: List[str], prefix: str = "stratus_days", subdirectory = None) -> None:
        """
        Plot absolute error of delta (geneva-dole) for the given days, saving in the corresponding month directory.

        Args:
            days: List of days in format 'YYYY-MM-DD'
            prefix: Prefix for filename
        """
        print("Here")
        df = self._prepare_day_metrics(days)
        if df.empty:
            self.logger.warning("No data found for the provided days.")
            return

        # Compute absolute error of delta (geneva-dole)
        delta_abs_error = ((df["predicted_geneva"] - df["predicted_dole"]) -
                           (df["expected_geneva"] - df["expected_dole"])).abs()

        # Group by month for saving in month directory
        if "month" not in df.columns:
            df["month"] = df["datetime"].dt.strftime("%Y-%m")
        months = df["month"].unique()

        for month in months:
            month_df = df[df["month"] == month]
            month_delta_abs_error = delta_abs_error[month_df.index]

            # Plot each value (not mean) for the month
            fig, ax = plt.subplots(figsize=self.plot_config.figsize)
            # Asse X = range numerico
            x_vals = np.arange(len(month_df))
            dates_labels = month_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")

            ax.plot(x_vals, month_delta_abs_error[month_df.index],
                    'o-', color='red',
                    markersize=self.plot_config.marker_size,
                    linewidth=self.plot_config.line_width,
                    label='Absolute Error (geneva- Dole)')

            # Tick ogni N punti
            step = max(1, len(x_vals) // 10)
            ax.set_xticks(x_vals[::step])
            ax.set_xticklabels(dates_labels[::step], rotation=45)

            ax.set_title(f"Absolute Error of Delta (geneva-Dole) - {month}",
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

    def get_delta_metrics_for_days(self, days: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute delta metrics (geneva-dole differences) for specific days.
        
        Args:
            days: List of days in format 'YYYY-MM-DD'
            
        Returns:
            Dictionary with delta metrics for each day and globally
        """
        day_df = self._prepare_day_metrics(days)
        if day_df.empty:
            self.logger.warning(f"No data found for days: {days}")
            return {}

        delta_metrics = {}
        global_metrics = {
            "mae": [],
            "rmse": [],
            "mean_relative_error": []
        }

        for day, group in day_df.groupby("date_str"):
            expected_delta = group["expected_geneva"] - group["expected_dole"]
            predicted_delta = group["predicted_geneva"] - group["predicted_dole"]
            
            abs_error = (predicted_delta - expected_delta).abs()
            mae = abs_error.mean()
            rmse = np.sqrt(((predicted_delta - expected_delta) ** 2).mean())
            rel_error = abs_error / expected_delta.abs().replace(0, np.nan)
            mean_rel_error = rel_error.fillna(0).mean()
            
            delta_metrics[day] = {
                "mae": mae,
                "rmse": rmse,
                "mean_relative_error": mean_rel_error
            }
            
            # Collect for global metrics
            global_metrics["mae"].append(mae)
            global_metrics["rmse"].append(rmse)
            global_metrics["mean_relative_error"].append(mean_rel_error)

        # Calculate global delta metrics
        global_delta_metrics = {
            metric: np.mean(vals) if vals else None
            for metric, vals in global_metrics.items()
        }

        return {
            "daily": delta_metrics,
            "global": global_delta_metrics
        }

    def save_metrics_report(self, stratus_days: Optional[List[str]] = None, 
                        non_stratus_days: Optional[List[str]] = None) -> None:
        """
        Save comprehensive metrics report including delta metrics.
        """
        report_lines = [
            "=== Metrics Report ===",
            f"Accuracy (tolerance={self.tolerance}): {self.get_accuracy():.4f}",
            f"Mean Absolute Error: {self.get_mean_absolute_error()}",
            f"Root Mean Squared Error: {self.get_root_mean_squared_error()}",
            f"Mean Relative Error: {self.get_mean_relative_error()}",
            f"\n=== Global Delta geneva-Dole Stats ===",
            f"{self.get_delta_stats()}",  # Uses the cached version
        ]

        if stratus_days:
            stratus_metrics = self.get_global_metrics_for_days(stratus_days)
            stratus_delta = self.get_delta_metrics_for_days(stratus_days)["global"]
            
            report_lines.extend([
                "\n=== Stratus Days Metrics ===",
                f"Days: {stratus_days}",
                f"Global RMSE: {stratus_metrics.get('rmse', {})}",
                f"Global Relative Error: {stratus_metrics.get('relative_error', {})}",
                f"Global MAE: {stratus_metrics.get('mae', {})}",
                f"Delta geneva-Dole Stats: {stratus_delta}",
            ])

        if non_stratus_days:
            non_stratus_metrics = self.get_global_metrics_for_days(non_stratus_days)
            non_stratus_delta = self.get_delta_metrics_for_days(non_stratus_days)["global"]
            
            report_lines.extend([
                "\n=== Non-Stratus Days Metrics ===",
                f"Days: {non_stratus_days}",
                f"Global RMSE: {non_stratus_metrics.get('rmse', {})}",
                f"Global Relative Error: {non_stratus_metrics.get('relative_error', {})}",
                f"Global MAE: {non_stratus_metrics.get('mae', {})}",
                f"Delta geneva-Dole Stats: {non_stratus_delta}",
            ])

        if self.save_path:
            report_path = os.path.join(self.save_path, "metrics_report.txt")
            with open(report_path, 'w') as f:
                f.write("\n".join(report_lines))
            self.logger.info(f"Saved metrics report to {report_path}")

    def compute_and_save_metrics_by_month(self, days: List[str], label: str = "stratus_days") -> None:
        """
        Compute and save metrics organized by month, including delta metrics.
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
            delta_metrics = self.get_delta_metrics_for_days(month_days)
            # Save report
            report_path = os.path.join(month_dir, f"metrics_{label}.txt")
            with open(report_path, 'w') as f:
                f.write(f"Metrics for {label} - {month}\n")
                f.write(f"Days: {month_days}\n")
                f.write(f"Global RMSE: {metrics.get('rmse', {})}\n")
                f.write(f"Global Relative Error: {metrics.get('relative_error', {})}\n")
                f.write(f"Global MAE: {metrics.get('mae', {})}\n")
                if delta_metrics and "global" in delta_metrics and delta_metrics["global"]:
                    f.write(f"Delta geneva-Dole Stats: {delta_metrics['global']}\n")
                else:
                    f.write("Delta geneva-Dole Stats: No data available\n")
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
    


    def detect_critical_transitions(
        self,
        days: List[str],
        min_slope: float = 70,
        min_peak_distance: str = "10min",
        smooth_window: str = "15min",
        plot_day: str = "2024-11-08"
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Detect critical transitions using:
        - Slope analysis (1st and 2nd derivatives)
        - Statistical change detection (z-score based)
        - Peak detection on multiple features
        """
        from scipy.signal import find_peaks, savgol_filter
        from scipy.stats import zscore
        import matplotlib.pyplot as plt

        # Load and prepare data
        df = self.get_delta_btw_geneva_dole()
        days = self._normalize_days_input(days)
        df = df.sort_values("datetime").set_index("datetime")
        df = df[df.index.strftime("%Y-%m-%d").isin(days)].sort_index()

        # Calculate deltas and prepare results
        df["expected_delta"] = df["expected_geneva"] - df["expected_dole"]
        df["predicted_delta"] = df["predicted_geneva"] - df["predicted_dole"]
        results = {}

        for prefix in ["expected", "predicted"]:
            series = df[f"{prefix}_delta"]
            
            # 1. Smoothing and derivatives
            smoothed = series.rolling(smooth_window, center=True).mean()
            first_deriv = smoothed.diff()
            second_deriv = first_deriv.diff()
            
            # 2. Statistical change detection (z-score)
            z_scores = zscore(series.fillna(0))
            change_points = np.where(np.abs(z_scores) > 2.5)[0]  # 3.5 std threshold

            # 3. Find peaks in multiple features
            min_samples = max(1, int(pd.Timedelta(min_peak_distance).total_seconds() / 
                                  (df.index[1] - df.index[0]).total_seconds()))
            
            features_to_analyze = {
                'slope': first_deriv.abs().fillna(0).values,
                'accel': second_deriv.abs().fillna(0).values,
                'delta': series.abs().values
            }
            
            all_peaks = []
            for feature_name, feature_values in features_to_analyze.items():
                try:
                    peaks, _ = find_peaks(
                        feature_values,
                        height=np.percentile(feature_values, 75),  # Dynamic height
                        distance=min_samples
                    )
                    all_peaks.extend(peaks)
                except:
                    continue

            # Combine all detection methods
            unique_peaks = list(set(list(change_points) + all_peaks))
            
            # Create results dataframe
            peaks_df = df.iloc[unique_peaks].copy() if unique_peaks else pd.DataFrame()
            if not peaks_df.empty:
                # Add detection info
                peaks_df["slope_magnitude"] = first_deriv.iloc[unique_peaks].abs()
                peaks_df["z_score"] = z_scores[unique_peaks]
                
                # Score based on multiple factors
                peaks_df["confidence"] = (
                    0.6 * peaks_df["slope_magnitude"] / peaks_df["slope_magnitude"].max() +
                    0.4 * peaks_df["z_score"].abs() / peaks_df["z_score"].abs().max()
                )
                # Filter out low-confidence transitions
                peaks_df = peaks_df[peaks_df["confidence"] >= 0.3]
                peaks_df = peaks_df.sort_values("confidence", ascending=False)

            results[f"{prefix}_transitions"] = peaks_df

            # Plotting
            if plot_day and (pd.to_datetime(plot_day).date() in df.index.date):
                day_data = df[df.index.date == pd.to_datetime(plot_day).date()]
                
                fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
                
                # Original data and delta
                axs[0].plot(day_data.index, day_data[f"{prefix}_geneva"], label="Geneva")
                axs[0].plot(day_data.index, day_data[f"{prefix}_dole"], label="Dole")
                axs[0].plot(day_data.index, day_data[f"{prefix}_delta"], 
                           label="Delta", color='green', linestyle='--')
                axs[0].set_title(f"{prefix.capitalize()} Values and Delta")
                axs[0].legend()
                axs[0].grid()
                
                # Derivatives
                axs[1].plot(day_data.index, first_deriv[day_data.index], 
                           label="1st Derivative (Slope)", color='orange')
                axs[1].plot(day_data.index, second_deriv[day_data.index], 
                           label="2nd Derivative (Accel)", color='red')
                axs[1].axhline(y=min_slope, color='gray', linestyle='--')
                axs[1].axhline(y=-min_slope, color='gray', linestyle='--')
                axs[1].set_title("Derivatives Analysis")
                axs[1].legend()
                axs[1].grid()
                
                # Z-scores and peaks
                z_scores_series = pd.Series(z_scores, index=df.index)
                axs[2].plot(day_data.index, z_scores_series.loc[day_data.index], 
                           label="Z-scores", color='purple')
                axs[2].axhline(y=2.5, color='r', linestyle='--', label='Threshold')
                axs[2].axhline(y=-2.5, color='r', linestyle='--')
                axs[2].set_title("Statistical Change Detection")
                axs[2].legend()
                axs[2].grid()
                
                # Mark detected transitions
                day_peaks = peaks_df[peaks_df.index.date == pd.to_datetime(plot_day).date()]
                for ax in axs:
                    for idx, row in day_peaks.iterrows():
                        ax.axvline(x=idx, color='r', linestyle=':', alpha=0.7)
                        if ax == axs[0]:  # Add label only once
                            ax.text(idx, ax.get_ylim()[1]*0.9, 
                                   f"Conf: {row['confidence']:.2f}", 
                                   rotation=90, va='top')

                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.save_path, f"{prefix}_transitions_{plot_day}.png"),
                    dpi=self.plot_config.dpi, bbox_inches='tight'
                )
                plt.close()
        print(results)
        return results
    def match_strongest_peaks(
        self,
        peaks_results: Dict[str, pd.DataFrame],
        time_window: str = "3H",
        min_confidence: float = 0.7,
        include_unmatched: bool = True
    ) -> pd.DataFrame:
        """
        Enhanced matching of critical transitions that:
        1. Uses confidence scores from detect_critical_transitions
        2. Matches based on both timing and confidence similarity
        3. Includes comprehensive matching metadata
        4. Handles edge cases robustly
        
        Args:
            peaks_results: Output from detect_critical_transitions
            time_window: Maximum time difference for matching
            min_confidence: Minimum confidence threshold for matches
            include_unmatched: Whether to include unmatched peaks in results
            
        Returns:
            DataFrame with matched and optionally unmatched peaks
        """
        # Prepare the dataframes
        exp_peaks = peaks_results.get("expected_transitions", pd.DataFrame()).reset_index()
        pred_peaks = peaks_results.get("predicted_transitions", pd.DataFrame()).reset_index()

        # Early return if empty inputs
        if exp_peaks.empty or pred_peaks.empty:
            return pd.DataFrame()

        # Ensure datetime column exists
        if "datetime" not in exp_peaks.columns:
            exp_peaks = exp_peaks.rename(columns={exp_peaks.columns[0]: "datetime"})
        if "datetime" not in pred_peaks.columns:
            pred_peaks = pred_peaks.rename(columns={pred_peaks.columns[0]: "datetime"})

        # Normalize confidence scores for fair comparison
        max_conf = max(exp_peaks["confidence"].max(), pred_peaks["confidence"].max())
        exp_peaks["norm_conf"] = exp_peaks["confidence"] / max_conf
        pred_peaks["norm_conf"] = pred_peaks["confidence"] / max_conf

        matches = []
        max_time_diff = pd.Timedelta(time_window)
        used_pred_indices = set()
        matched_exp_indices = set()

        # First pass - find best matches
        for _, exp_row in exp_peaks.iterrows():
            # Find candidate predicted peaks within time window
            candidates = pred_peaks[
                (pred_peaks["datetime"].between(
                    exp_row["datetime"] - max_time_diff,
                    exp_row["datetime"] + max_time_diff
                )) &
                (~pred_peaks.index.isin(used_pred_indices))
            ].copy()

            if not candidates.empty:
                # Calculate matching scores
                candidates["time_diff"] = (candidates["datetime"] - exp_row["datetime"]).abs().dt.total_seconds()
                candidates["time_score"] = 1 - (candidates["time_diff"] / max_time_diff.total_seconds())
                candidates["conf_score"] = 1 - abs(candidates["norm_conf"] - exp_row["norm_conf"])
                
                # Combined score weights time more heavily
                candidates["combined_score"] = (
                    0.2 * candidates["time_score"] + 
                    0.8 * candidates["conf_score"]
                )

                # Get best match
                best_match = candidates.nlargest(1, "combined_score").iloc[0]

                if best_match["combined_score"] >= min_confidence:
                    matches.append({
                        "expected_time": exp_row["datetime"],
                        "predicted_time": best_match["datetime"],
                        "expected_confidence": exp_row["confidence"],
                        "predicted_confidence": best_match["confidence"],
                        "time_difference_sec": (best_match["datetime"] - exp_row["datetime"]).total_seconds(),
                        "confidence_similarity": best_match["conf_score"],
                        "combined_score": best_match["combined_score"],
                        "match_status": "matched",
                        "is_early": best_match["datetime"] < exp_row["datetime"],
                        "is_late": best_match["datetime"] > exp_row["datetime"],
                        "expected_slope": exp_row.get("slope_magnitude", None),
                        "predicted_slope": best_match.get("slope_magnitude", None),
                        "expected_z_score": exp_row.get("z_score", None),
                        "predicted_z_score": best_match.get("z_score", None)
                    })
                    matched_exp_indices.add(exp_row.name)
                    used_pred_indices.add(best_match.name)

        # Second pass - include unmatched expected peaks if requested
        if include_unmatched:
            unmatched_exp = exp_peaks[
                (~exp_peaks.index.isin(matched_exp_indices)) &
                (exp_peaks["confidence"] >= min_confidence)
            ]
            for _, exp_row in unmatched_exp.iterrows():
                matches.append({
                    "expected_time": exp_row["datetime"],
                    "predicted_time": pd.NaT,
                    "expected_confidence": exp_row["confidence"],
                    "predicted_confidence": None,
                    "time_difference_sec": None,
                    "confidence_similarity": None,
                    "combined_score": None,
                    "match_status": "unmatched_expected",
                    "is_early": None,
                    "is_late": None,
                    "expected_slope": exp_row.get("slope_magnitude", None),
                    "predicted_slope": None,
                    "expected_z_score": exp_row.get("z_score", None),
                    "predicted_z_score": None
                })

        # Third pass - include high-confidence unmatched predicted peaks
        unmatched_pred = pred_peaks[
            (~pred_peaks.index.isin(used_pred_indices)) &
            (pred_peaks["confidence"] >= min_confidence)
        ]
        for _, pred_row in unmatched_pred.iterrows():
            matches.append({
                "expected_time": pd.NaT,
                "predicted_time": pred_row["datetime"],
                "expected_confidence": None,
                "predicted_confidence": pred_row["confidence"],
                "time_difference_sec": None,
                "confidence_similarity": None,
                "combined_score": None,
                "match_status": "unmatched_predicted",
                "is_early": None,
                "is_late": None,
                "expected_slope": None,
                "predicted_slope": pred_row.get("slope_magnitude", None),
                "expected_z_score": None,
                "predicted_z_score": pred_row.get("z_score", None)
            })

        # Convert to DataFrame and clean up
        result_df = pd.DataFrame(matches)
        
        # Sort by match quality and time
        result_df = result_df.sort_values(
            by=["combined_score", "expected_time"], 
            ascending=[False, True],
            na_position="last"
        )

        # Calculate additional metrics for matched peaks
        if not result_df.empty:
            matched = result_df[result_df["match_status"] == "matched"]
            if not matched.empty:
                print(f"Matched {len(matched)} peaks "
                      f"(avg time diff: {matched['time_difference_sec'].mean():.1f}s, "
                      f"avg conf similarity: {matched['confidence_similarity'].mean():.2f})")
     
        # Remove matches where time_difference_sec is negative (i.e., predicted is earlier than expected)
        result_df = result_df[
            (result_df["time_difference_sec"].isna()) | (result_df["time_difference_sec"] >= 0)
        ]
        return result_df