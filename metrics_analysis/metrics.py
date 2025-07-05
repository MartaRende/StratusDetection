import numpy as np
import pandas as pd
import logging
import os
from typing import Optional, Union, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from PIL import Image
from .transitions import TransitionAnalyzer
from .config import PlotConfig
from .plotting import Plotter

@dataclass
class Metrics:
    """
    Main metrics calculation class with core functionality.
    Delegates plotting and transition analysis to separate classes.
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
        
        self._initialize_data(expected, predicted, data)
        self._setup_datetime_filters(start_date, end_date)
        self._setup_paths(save_path)
        self._initialize_configurations(stats_for_month, tolerance, plot_config, log_level)
        self.image_base_folder = fp_images if fp_images else ""
        self.num_views = num_views
        self.prediction_minutes = prediction_minutes
        self.transition_analyzer = TransitionAnalyzer(self)
        self.plotter = Plotter(self)
        self.config = PlotConfig()
    

    def _initialize_data(self, expected, predicted, data):
        """Initialize and normalize data structures"""
        self.expected = pd.DataFrame(expected, columns=["geneva", "dole"])
        self.predicted = pd.DataFrame(predicted, columns=["geneva", "dole"])
        self.data = pd.json_normalize(pd.DataFrame(data["dole"])[0])
        
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        self.data["gre000z0_geneva"] = pd.to_numeric(self.data["gre000z0_nyon"])
        self.data["gre000z0_dole"] = pd.to_numeric(self.data["gre000z0_dole"])
       
        self._geneva_values = self.data["gre000z0_geneva"].to_numpy()
        self._dole_values = self.data["gre000z0_dole"].to_numpy()
        self._datetime_values = self.data["datetime"].to_numpy()

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
        exp_geneva = self.expected["geneva"].to_numpy()
        exp_dole = self.expected["dole"].to_numpy()
        
        for geneva, dole in zip(exp_geneva, exp_dole):
            mask = (
                np.isclose(self._geneva_values, geneva, atol=1e-6) & 
                np.isclose(self._dole_values, dole, atol=1e-6)
            )
            
            if self.start_date and self.end_date:
                date_mask = (
                    (self._datetime_values >= self.start_date) & 
                    (self._datetime_values <= self.end_date)
                )
                mask = mask & date_mask
            
            matches = self._datetime_values[mask]
            datetimes.append(matches[0] if len(matches) > 0 else None)
            
        return datetimes

    def get_image_for_datetime(self, dt, view=2):
        """Get image for specific datetime"""
        date_str = dt.strftime('%Y-%m-%d')
        time_str = dt.strftime('%H%M')
        img_filename = f"1159_{view}_{date_str}_{time_str}.jpeg"
        img_path = os.path.join(self.image_base_folder, str(view), dt.strftime('%Y'), 
                              dt.strftime('%m'), dt.strftime('%d'), img_filename)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            return img_array
        else:
            self.logger.warning(f"Image not found for {dt}: {img_path}")
            return []

    def get_correct_predictions(self, tol: Optional[float] = None) -> int:
        """Count predictions within tolerance of expected values"""
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
        """Calculate the difference between geneva and Dole values"""
        df = self._create_comparison_dataframe()
        df["expected_delta_geneva_dole"] = df["expected_geneva"] - df["expected_dole"]
        df["predicted_delta_geneva_dole"] = df["predicted_geneva"] - df["predicted_dole"]
        return df.dropna(subset=["datetime"])

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
        """Prepare a dataframe filtered by specific days"""
        days = self._normalize_days_input(days)
        df = self._create_comparison_dataframe()
        df["date_str"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df["hour"] = df["datetime"].dt.strftime("%H:%M")
        df["month"] = df["datetime"].dt.strftime("%Y-%m")
        return df[df["date_str"].isin(days)]

    def _normalize_days_input(self, days) -> List[str]:
        """Helper to normalize days input to a flat list of strings"""
        if isinstance(days, np.ndarray):
            days = days.flatten().tolist()
        elif isinstance(days, (list, tuple)):
            days = [item for sublist in days for item in 
                   (sublist if isinstance(sublist, (list, tuple, np.ndarray)) else [sublist])]
            days = [str(d) for d in days]
        else:
            days = [str(days)]
        return days
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
            self.plotter.plot_error_metrics(month_days, metric_type="rmse", prefix=label, subdirectory=month_dir)
            self.plotter.plot_error_metrics(month_days, metric_type="relative_error", prefix=label, subdirectory=month_dir)
            self.plotter.plot_delta_absolute_error(month_days, prefix=label, subdirectory=month_dir)
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
