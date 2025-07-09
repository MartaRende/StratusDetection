from dataclasses import dataclass
from typing import Dict, Tuple

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