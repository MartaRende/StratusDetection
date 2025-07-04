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