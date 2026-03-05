# Usage Examples

## Basic DMFT Calculation

### 1. Prepare Configuration File

Create `config.yaml`:

```yaml
# Model parameters
model: "hubbard"
size: 1000                      # Total number of lattice sites
t: 1.0                          # Hopping parameter
U: 1.0                          # On-site interaction
sizeA: 1                        # Number of impurity sites
Np: 1.0                         # Particle density
bc: 1                           # One-dimensional lattice boundary conditions: 1 = periodic (ring), 0 = open chain

embedding:
  max_iter: 300                 # Maximum number of self-consistent iterations
  num_poles: 4                  # Number of residues/poles for the SOP representation of the embedding potential
  mu_fixed: 0.5                 # If null - chemical potential updated self-consistently
  p_type: "sqrt"                # Either the residue or its square root: "std" or "sqrt"
  axis: "imaginary"             # Frequency grid axis: "erf", "shift", "imaginary"
  eta_axis: 0.                  # Distance from the real axis to be used for "erf" and "shift" cases
  num_pts: 10000                # Number of grid points
  beta_T: 1500.0                # Value of beta to build Matsubara parameters: 2 pi (n + 1) / beta
  Nw_max: 3000                  # Maximum number of n in Matsubara parameters
  w_edges: [-5, 5]              # Frequency grid edges 
  sparse_gs: true               # Diagonalization type when searching for the gs subspace auxiliary Hamiltonian
  gs_search: "std"              # Type of gs subspace, either :"std" or "subspaces"
  solver_method: "std"          # Type of solver: full diagonalization ("std") or bi-Lanczos method ("lanczos")

optimization:
  mixing_method: "linear"       # Type of mixing of the embedding potential
  alpha: 0.5                    # Mixing parameter
  opt_method: "scipy_CG"        # Types of algorithms for the minimization: "anal_SD", "scipy_CG", "custom_CG"
  opt_params: [100, 1e-5]       # [maximum num. of iterations, initial learning rate]
  complex_poles: false          # Constraint on having complex poles
  herm_residues: true           # Constraint on having Hermitian residues
  fixed_residues: false         # Constraint on keeping the residues fixed
  odd_spectrum: true            # Constraint on having poles anti-symmetric w.r.t. the origin, and residues symmetric
  paramagnetic: true            # Constraint on having up-down spin symmetry - N.B. paramagnetic = true works only with sizeA = 1
  interp_method: "sampling"     # Method of interpolation in the line search of the custom_CG method: "sampling" or "parabola"
  print_interp: false           # Printing interpolation in the line search procedure - Applies only to custom_CG
  p0_start: "always"            # Use of previous SOP parameters: "never", "always", "no_first_iter" (not in the first iterations)
  thr_diff_prev: 1e-8           # Threshold to assume converged the self-consistent cycle (integrated difference of the DOS of 2 consecutive Gimp)
  thr_stagnation: 1e-9          # Threshold to establish stagnation of the algorithm
  RMSE_thr: 0.5                 # Threshold to warn if the fitting is poor
```

### 2. Run Simulation

```python
import logging, os
from datetime           import datetime
from .data_io            import clean_folder
from .dmft_config        import load_sim_config
from .dmft_simulation    import dmft_simulation

clean_folder(fig_dir="figures")

def setup_logging():
    logging.basicConfig(
        filename="log.txt",                 # file name
        filemode="w",                       # writing mode
        level=logging.INFO,                 # minimum level: info, debug, warning
        format="%(message)s"
    )

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("DMFT")

    output_file_names = {
        "dmft": "dmft_output.json",
        "vemb": "vemb_output.json",
        "conv": "conv_output.json",
        "opt" : "opt_output.json"}
    
    config = load_sim_config("config.yaml")

    sim = dmft_simulation(config)
    start_time = datetime.now()
    logger.info("Start DMFT simulation - %s",start_time)
    sim.run(output_file_names=output_file_names)
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    logger.info("End DMFT simulation - %s",end_time)
    logger.info("Total runtime: %s",elapsed_time)
```

### 3. Analyze Results

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open("dmft_output.json", "r") as f:
    dmft_results = json.load(f)

with open("conv_output.json", "r") as f:
    convergence = json.load(f)

# Plot convergence
```

## Accessing Results Programmatically

```python
```

## Tips for Best Results

1. **Convergence:** Start with fewer iterations, increase `max_iter` gradually
2. **Frequency Resolution:** Use at least `num_pts=10000` for reasonable accuracy, and `w_edge=U` to study the range of prominent spectral features
3. **Distance from the real axis:** Start from values of `eta_axis = 2.` and slowly decrease the value to study the effect of the proximity to the real axis on the erf path
4. **Mixing:** Use mixing ratio of `alpha=0.5` in iterative updates for stability

---

For more examples, see the `examples/` directory.
