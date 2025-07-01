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
from matplotlib.lines import Line2D
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
        self.colors = self.colors or {'nyon': '#1f77b4', 'dole': '#ff7f0e'}

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
                 time_key: Optional[str] = None,
                 num_views: int = 1,
                 stats_for_month: bool = True,
                 tolerance: float = 20.0,
                 plot_config: Optional[PlotConfig] = None,
                 log_level: int = logging.INFO):
        """
        Initialize the Metrics calculator.
        
        Args:
            expected: Ground truth values with columns ["nyon", "dole"]
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
        self.time_key = time_key

        self._initialize_data(expected, predicted, data)
        self._setup_datetime_filters(start_date, end_date)
        self._setup_paths(save_path)
        self._initialize_configurations(stats_for_month, tolerance, plot_config, log_level)
        self.image_base_folder = fp_images if fp_images else ""
        self.num_views = num_views
        self.test = 0

    def _initialize_data(self, expected, predicted, data):
        """Initialize and normalize data structures"""
        self.expected = pd.DataFrame(expected, columns=["nyon", "dole"])
        self.predicted = pd.DataFrame(predicted, columns=["nyon", "dole"])
    
        # Normalize the input data
        self.data = pd.json_normalize(pd.DataFrame(data["dole"])[0])
        
        # Convert to appropriate dtypes
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        self.data["gre000z0_nyon"] = pd.to_numeric(self.data["gre000z0_nyon"])
        self.data["gre000z0_dole"] = pd.to_numeric(self.data["gre000z0_dole"])
        
        # Find matching indices in raw data
        self._filtered_indices = self._find_matching_indices()
        
        # Filter the raw data to only include matching points
        self._nyon_values = self.data["gre000z0_nyon"].iloc[self._filtered_indices].to_numpy()
        self._dole_values = self.data["gre000z0_dole"].iloc[self._filtered_indices].to_numpy()
        self._datetime_values = self.data["datetime"].iloc[self._filtered_indices].to_numpy()

    def _find_matching_indices(self):
        """Find indices in raw data that match expected values"""
        matched_indices = []
        exp_nyon = self.expected["nyon"].to_numpy()
        exp_dole = self.expected["dole"].to_numpy()
        
        raw_nyon = self.data["gre000z0_nyon"].to_numpy()
        raw_dole = self.data["gre000z0_dole"].to_numpy()
        
        for nyon, dole in zip(exp_nyon, exp_dole):
            # Find closest match in raw data
            distances = np.sqrt(
                (raw_nyon - nyon)**2 + 
                (raw_dole - dole)**2
            )
            min_idx = np.argmin(distances)
            
            if distances[min_idx] < 2.0:  # Only accept if within 2 W/m²
                matched_indices.append(min_idx)
            else:
                self.logger.warning(f"No close match found for nyon={nyon}, dole={dole}")
                matched_indices.append(None)  # Or handle missing values as needed
        
        # Verify we found exactly 74 matches
        if len(matched_indices) != len(self.expected):
            self.logger.error(f"Found {len(matched_indices)} matches but expected {len(self.expected)}")
     
        return [idx for idx in matched_indices if idx is not None]  # Filter out None values
        
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
            metrics_folder = os.path.join(save_path, f"metrics_{self.time_key}") if self.time_key else os.path.join(save_path, "metrics")
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
        """Simply return the filtered datetime values"""
       
        return list(self._datetime_values)  # Now guaranteed to match expected/predicted

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
        rel_error = abs_error / self.expected.abs().replace(0, np.nan)

        assert len(self.predicted) == len(self.expected), "Mismatch in predicted and expected lengths"
        return rel_error.fillna(0).to_dict(orient='list')

    def get_mean_relative_error(self) -> Dict[str, float]:
        """Calculate mean relative error for each variable"""
        rel_error = self.get_relative_error()
        return {k: np.mean(v) for k, v in rel_error.items()}

   
    def get_delta_btw_nyon_dole(self) -> pd.DataFrame:
        """
        Calculate the difference between Nyon and Dole values.
        
        Returns:
            DataFrame with datetime, expected, predicted values and their differences
        """
        df = self._create_comparison_dataframe()
        df["expected_delta_nyon_dole"] = df["expected_nyon"] - df["expected_dole"]
        df["predicted_delta_nyon_dole"] = df["predicted_nyon"] - df["predicted_dole"]
        return df.dropna(subset=["datetime"])
    def get_delta_stats(self) -> Dict[str, float]:
        """
        Compute MAE, RMSE, and mean relative error for delta_dole_nyon.
        Returns:
            Dictionary with keys: 'mae', 'rmse', 'mean_relative_error'
        """
        df = self.get_delta_btw_nyon_dole()
        expected_delta = df["expected_delta_nyon_dole"]
        predicted_delta = df["predicted_delta_nyon_dole"]
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
            "expected_nyon": self.expected["nyon"],
            "expected_dole": self.expected["dole"],
            "predicted_nyon": self.predicted["nyon"],
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
                    "nyon": (group["predicted_nyon"] - group["expected_nyon"]).abs().mean(),
                    "dole": (group["predicted_dole"] - group["expected_dole"]).abs().mean(),
                },
                "rmse": {
                    "nyon": np.sqrt(((group["predicted_nyon"] - group["expected_nyon"]) ** 2).mean()),
                    "dole": np.sqrt(((group["predicted_dole"] - group["expected_dole"]) ** 2).mean()),
                },
                "relative_error": {
                    "nyon": ((group["predicted_nyon"] - group["expected_nyon"]).abs() / 
                            group["expected_nyon"].replace(0, np.nan)).fillna(0).mean(),
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
            "mae": {"nyon": [], "dole": []},
            "rmse": {"nyon": [], "dole": []},
            "relative_error": {"nyon": [], "dole": []}
        }

        # Collect all values
        for metrics in day_metrics.values():
            for metric_type in global_metrics:
                for var in ["nyon", "dole"]:
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
        nyon_values = [day_metrics[day][metric_type]["nyon"] for day in days_list]
        dole_values = [day_metrics[day][metric_type]["dole"] for day in days_list]

        # Create plot
        fig, ax = plt.subplots(figsize=self.plot_config.figsize)
        
        ax.plot(days_list, nyon_values, 
                marker='o', linestyle='-', 
                color=self.plot_config.colors["nyon"],
                markersize=self.plot_config.marker_size,
                linewidth=self.plot_config.line_width,
                label='Nyon')
                
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

            for var in ["nyon", "dole"]:
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

            ax1.set_title(f"Day Curves - {day}", fontsize=self.plot_config.fontsize["title"])
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

            # Original difference plot (unchanged)
            fig, ax = plt.subplots(figsize=self.plot_config.figsize)
            for var in ["nyon", "dole"]:
                ax.plot(day_df["hour"],
                        (day_df[f"expected_{var}"] - day_df[f"predicted_{var}"]),
                        'o-', color=self.plot_config.colors[var],
                        markersize=self.plot_config.marker_size,
                        linewidth=self.plot_config.line_width,
                        label=f'{var.capitalize()} Difference')

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
        Plot absolute error of delta (nyon-dole) for the given days, saving in the corresponding month directory.

        Args:
            days: List of days in format 'YYYY-MM-DD'
            prefix: Prefix for filename
        """
        print("Here")
        df = self._prepare_day_metrics(days)
        if df.empty:
            self.logger.warning("No data found for the provided days.")
            return

        # Compute absolute error of delta (nyon-dole)
        delta_abs_error = ((df["predicted_nyon"] - df["predicted_dole"]) -
                           (df["expected_nyon"] - df["expected_dole"])).abs()

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
                    label='Absolute Error (Nyon - Dole)')

            # Tick ogni N punti
            step = max(1, len(x_vals) // 10)
            ax.set_xticks(x_vals[::step])
            ax.set_xticklabels(dates_labels[::step], rotation=45)

            ax.set_title(f"Absolute Error of Delta (Nyon-Dole) - {month}",
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
        Compute delta metrics (nyon-dole differences) for specific days.
        
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
            expected_delta = group["expected_nyon"] - group["expected_dole"]
            predicted_delta = group["predicted_nyon"] - group["predicted_dole"]
            
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
            f"\n=== Global Delta Nyon-Dole Stats ===",
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
                f"Delta Nyon-Dole Stats: {stratus_delta}",
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
                f"Delta Nyon-Dole Stats: {non_stratus_delta}",
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
                    f.write(f"Delta Nyon-Dole Stats: {delta_metrics['global']}\n")
                else:
                    f.write("Delta Nyon-Dole Stats: No data available\n")
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

    def create_prediction_dataframe(self, 
                              expected_values: List[List[float]], 
                              predicted_values: List[List[float]], 
                              days: List[str],
                              time_steps: List[int] = ["t_0", "t_1", "t_2", "t_3", "t_4", "t_5"]) -> pd.DataFrame:
        """
        Create a structured DataFrame containing expected values and predictions at multiple time steps,
        filtered for specific days.
        
        Args:
            expected_values: List of expected values for each time point [[nyon, dole], ...]
            predicted_values: List of predicted values for each time point [[nyon_t0, dole_t0, nyon_t1, dole_t1, ...], ...]
            days: List of dates in format 'YYYY-MM-DD' to filter by
            time_steps: List of prediction time steps (1=10min, 2=20min, etc.)
        
        Returns:
            pd.DataFrame: Structured DataFrame with expected and predicted values for all time steps
        """
        # Get the datetime values that were already matched with expected/predicted values
        datetimes = self.datetime_list
  
        
        # Create base DataFrame with datetime and expected values
        df = pd.DataFrame({
        'datetime': self.datetime_list,
        'expected_nyon': [x[0] for x in expected_values[:, -1, :]],
        'expected_dole': [x[1] for x in expected_values[:, -1, :]],
    })
        df['date_str'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df = df.sort_values('datetime').reset_index(drop=True)
        missing_days = set(days) - set(df['date_str'].unique())
        
        if missing_days:
            print(f"Warning: No data found for days: {missing_days}")
        
        df = df[df['date_str'].isin(days)]
        
        if df.empty:
            print("Warning: No data remaining after filtering for specified days")
            return df
        
        # Convert datetime to hour:minute format
        df['hour'] = df['datetime'].dt.strftime('%H:%M')
        df['month'] = df['datetime'].dt.strftime('%Y-%m')
        
        # Get indices of filtered datetimes in original data
        original_indices = [i for i, dt in enumerate(self.datetime_list) if dt in df['datetime'].values]
    
        # Add predictions for each time step
        for t in time_steps:
     
            
            # Get predictions and expected values for this time step
            preds = predicted_values[t]
            expected = expected_values[:, -1, 0:2]
            
            # Verify lengths match
            if len(preds) != len(self.datetime_list) or len(expected) != len(self.datetime_list):
                print(f"Warning: Length mismatch for time step {t}")
                continue

            
            # Filter to only include data for our selected days
            filtered_preds = [preds[i] for i in original_indices]
            filtered_expected = [expected[i] for i in original_indices]
            
            # Add to dataframe
     
            df[f'predicted_nyon_{t}'] = [x[0] for x in filtered_preds]
            df[f'predicted_dole_{t}'] = [x[1] for x in filtered_preds]
            
            # Calculate future datetimes
            step_num = int(t.split('_')[1])
            df[f'datetime_{t}'] = df['datetime'] + pd.Timedelta(minutes=10*(step_num+1))
            df[f'hour_{t}'] = df[f'datetime_{t}'].dt.strftime('%H:%M')
            

        return df

    def plot_prediction_curves(self, 
                         expected_values: List[List[float]],
                         predicted_values: List[List[float]],
                         days: List[str],
                         time_interval_min: int = 10,
                         prediction_horizons: List[int] = [10, 20, 30, 40, 50, 60]) -> None:
        """
        Plot prediction curves for multiple horizons from each observation point for specific days,
        with all datetime points shown on x-axis regardless of data availability.
        """
        # Create dataframe filtered for specific days
        day_df = self.create_prediction_dataframe(expected_values, predicted_values, days)
        
        if day_df.empty:
            self.logger.warning(f"No data found for days: {days}")
            return

        # Plot each day separately
        for day in days:
            df_day = day_df[day_df["date_str"] == day]
            if df_day.empty:
                self.logger.warning(f"No data found for day: {day}")
                continue

            # Get ALL datetime points for this day
            all_day_datetimes = sorted([pd.to_datetime(dt) for dt in self.datetime_list 
                                    if pd.to_datetime(dt).strftime('%Y-%m-%d') == day])
            all_day_hours = [dt.strftime('%H:%M') for dt in all_day_datetimes]
            
            # Create mapping from hour string to x-position
            hour_to_pos = {hour: idx for idx, hour in enumerate(all_day_hours)}
            
            # Convert hour strings to x-positions in dataframe
            df_day["x_pos"] = df_day["hour"].map(hour_to_pos)
            
            # Prepare for images
            day_datetimes = df_day["datetime"].tolist()
            num_images = min(6, len(day_datetimes))
            if num_images > 1:
                indices = np.linspace(0, len(day_datetimes) - 1, num_images, dtype=int)
            else:
                indices = [0]

            # Create figure with subplots
            fig = plt.figure(figsize=(self.plot_config.figsize[0], self.plot_config.figsize[1] * 1.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax = fig.add_subplot(gs[0])

            # Convert prediction horizons to time steps
            time_steps = [f"t_{h // time_interval_min - 1}" for h in prediction_horizons]

            # Plot actual observations using x_pos
            ax.plot(df_day["x_pos"], df_day["expected_nyon"], 
                    '-o', color='blue', markersize=8, label='Actual Nyon')
            ax.plot(df_day["x_pos"], df_day["expected_dole"], 
                    '-o', color='red', markersize=8, label='Actual Dole')

            # Plot prediction curves
            for i, row in df_day.iterrows():
                current_time = row["hour"]
                current_xpos = row["x_pos"]
                current_nyon = row["expected_nyon"]
                current_dole = row["expected_dole"]
                
                for j, t in enumerate(time_steps):
                    pred_nyon = row.get(f"predicted_nyon_{t}", None)
                    pred_dole = row.get(f"predicted_dole_{t}", None)
                    future_time = row.get(f"hour_{t}", None)
                    
                    if pred_nyon is None or pred_dole is None or future_time is None:
                        continue

                    # Get x-position for future time
                    future_xpos = hour_to_pos.get(future_time, None)
                    if future_xpos is None:
                        # If future time not in original list, add it
                        all_day_hours.append(future_time)
                        # Re-sort and update mapping
                        all_day_hours = sorted(set(all_day_hours), key=lambda x: pd.to_datetime(x, format="%H:%M"))
                        hour_to_pos = {hour: idx for idx, hour in enumerate(all_day_hours)}
                        future_xpos = hour_to_pos[future_time]
                        # Update x-ticks
                        ax.set_xticks(range(len(all_day_hours)))
                        ax.set_xticklabels(all_day_hours, rotation=45)

                    linestyle = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 10))][j % 6]
                    nyon_label = f'Nyon +{prediction_horizons[j]}min' if i == 0 else ""
                    dole_label = f'Dole +{prediction_horizons[j]}min' if i == 0 else ""

                    ax.plot([current_xpos, future_xpos], [current_nyon, pred_nyon],
                            linestyle=linestyle, color='blue', alpha=0.7,
                            label=nyon_label)
                    ax.plot([current_xpos, future_xpos], [current_dole, pred_dole],
                            linestyle=linestyle, color='red', alpha=0.7,
                            label=dole_label)

            # Set x-ticks and labels
            ax.set_xticks(range(len(all_day_hours)))
            ax.set_xticklabels(all_day_hours, rotation=45)
            
            # Formatting
            ax.set_title(f"Prediction Curves - {day}", fontsize=14)
            ax.set_ylabel("Radiation (W/m²)", fontsize=12)
            ax.set_xlabel("Time", fontsize=12)
            
            # Legend and grid
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Actual Nyon'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Actual Dole'),
                *[Line2D([0], [0], color='gray', linestyle=linestyle, 
                label=f'+{h}min') for h, linestyle in zip(prediction_horizons, 
                                                        ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 10))])
            ]]
            ax.legend(handles=legend_elements, loc='upper right')
            ax.grid(True)

            # Bottom subplot for images
            ax2 = fig.add_subplot(gs[1])
            ax2.axis('off')
            
            if num_images > 0:
                img_width = 1.0 / num_images
                for i, idx in enumerate(indices):
                    dt = day_datetimes[idx]
                    img = self.get_image_for_datetime(dt)
                    if isinstance(img, list) or img is None or len(np.shape(img)) == 0:
                        continue
                    if np.all(img == 0):
                        self.logger.warning(f"Image for {dt} is completely black.")
                    else:
                        if img.max() - img.min() < 1e-3:
                            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                    if img.ndim == 2:
                        img = np.stack([img] * 3, axis=-1)
                    left = i * img_width
                    ax_img = fig.add_axes([left, -0.1, img_width, 0.25])
                    ax_img.imshow(img)
                    ax_img.set_title(pd.to_datetime(dt).strftime("%H:%M"), fontsize=8)
                    ax_img.axis('off')

            # Save plot
            month = df_day["month"].iloc[0]
            month_dir = os.path.join(self.save_path, month)
            os.makedirs(month_dir, exist_ok=True)
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(month_dir, f"prediction_curves_{day}.png"),
                dpi=self.plot_config.dpi,
                bbox_inches='tight'
            )
            print(f"Saved prediction curves plot for {day} to {month_dir}")
            plt.close()