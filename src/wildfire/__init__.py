"""wildfire: PINNs for wildfire simulation."""

from .pinn import PINN
from .plot import (
	PlotConfig,
	plot_final_perimeter,
	plot_final_result,
	plot_initial_condition,
	plot_initial_perimeter,
	plot_perimeter_comparison,
	plot_phi_comparison,
)
from .lsm import LevelSetPDE, LevelSetConfig

__all__ = [
	"PINN",
	"LevelSetPDE",
	"LevelSetConfig",
	"plot_initial_condition",
	"plot_final_result",
	"plot_initial_perimeter",
	"plot_final_perimeter",
	"plot_perimeter_comparison",
	"plot_phi_comparison",
	"PlotConfig",
]
