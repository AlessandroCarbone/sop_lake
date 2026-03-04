"""sop_lake - a computational framework for dynamical quantum embedding"""

__version__ = "0.1.0"
__author__ = "Alessandro Carbone"
__email__ = "alessandro.carbone@epfl.ch"

# Import main user-facing components
from .dmft_simulation import dmft_simulation
from .dmft_config import (
    load_sim_config,
    sim_config,
    Hubbard_system_config,
    input_config,
    embedding_config,
    optimization_config,
)
from .mbAIMSOP_solver import solver

# Make submodules available
from . import (
    cost_fn,
    cost_fn_eff,
    data_io,
    dyn_poles,
    dyn_poles_utils,
    embedding_utils,
    hubbard,
    lanczos,
    mb_utils,
    SOP,
)

__all__ = [
    # Main components
    "dmft_simulation",
    "load_sim_config",
    "sim_config",
    # Solver
    "solver",
    # Configuration classes
    "Hubbard_system_config",
    "input_config",
    "embedding_config",
    "optimization_config",
    # Submodules
    "cost_fn",
    "cost_fn_eff",
    "data_io",
    "dyn_poles",
    "dyn_poles_utils",
    "embedding_utils",
    "hubbard",
    "lanczos",
    "mb_utils",
    "SOP",
]
