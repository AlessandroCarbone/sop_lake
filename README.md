<div align="center">
  <img src="docs/assets/soplake_logo.png" width="300"/>
  <img src="docs/assets/sop_lake_text.png" width="300"/>
</div>

# sop_lake: a computational framework for dynamical quantum embedding based on sum-over-pole representations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://github.com/AlessandroCarbone/sop_lake/workflows/tests/badge.svg)](https://github.com/AlessandroCarbone/sop_lake/actions)

A Python implementation of Dynamical Mean-Field Theory (DMFT) based on a sum-over-poles (SOP) representation of Green's functions. The framework performs the DMFT self-consistent cycle directly close to the real frequency axis, bypassing the need for analytic continuation from the Matsubara axis.

## Physical context

The code targets the single-band Hubbard model on a 1D lattice. The DMFT self-consistency maps the lattice problem onto an auxiliary impurity model (AIM) whose embedding potential is represented as a sum of poles:

$$v_\text{emb}(\omega) = \sum_{k=1}^{M} \frac{\Gamma_k}{\omega - \sigma_k}$$

At each iteration the embedding potential is fitted to this rational form, the many-body AIM Hamiltonian is built and diagonalized exactly, and the resulting impurity Green's function is fed back into the self-consistency loop via the Dyson equation.

## Features

**Core DMFT cycle:**
- Full self-consistent DMFT loop with linear mixing
- SOP fitting of the embedding potential via cost-function minimization (conjugate gradient or steepest descent)
- Exact diagonalization (ED) impurity solver with support for both dense and sparse ground-state search
- Bi-Lanczos algorithm as an alternative to full ED for the Green's function
- Support for real-axis, error-function-shifted, and Matsubara (imaginary) frequency grids

**SOP representation:**
- `SOP` class with standard (`std`) and square-root (`sqrt`) residue parametrizations
- Analytical gradient computation for efficient optimization
- Algebraic inversion routines (`reversed_AIMSOP`) to extract self-energy and local Green's function directly as SOP objects
- Particle-hole symmetric (odd-spectrum) constraint

**Physical constraints:**
- Hermitian residues, positive semi-definite residues, real poles
- Paramagnetic (spin-degenerate) approximation for single-site fragments
- Configurable convergence on DOS difference between consecutive iterations

**Output and analysis:**
- JSON output for Green's function, self-energy, embedding potential, and convergence history
- Publication-quality plots of all quantities via `data_io`

## Requirements

- Python 3.8 or higher
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- PyYAML >= 5.3
- qiskit_nature (for Hamiltonian construction and Jordan-Wigner mapping)
- sympy (for diagonalizability checks)

## Installation

### From source

```bash
git clone https://github.com/AlessandroCarbone/sop_lake.git
cd sop_lake
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Quick start

### Basic usage

```python
from src.dmft_config import load_sim_config
from src.dmft_simulation import dmft_simulation

# Load configuration from YAML
config = load_sim_config("examples/config_hubbard.yaml")

# Run simulation
sim = dmft_simulation(config)
sim.run(output_file_names={
    "dmft": "dmft_output.json",
    "vemb": "vemb_output.json",
    "conv": "conv_output.json"
})
```

### Configuration file format

The configuration file is a YAML document with four required sections. A complete example for the Hubbard model at half-filling on a 1D ring:

```yaml
system:
  size: 1000        # Total number of lattice sites (Nk = size / sizeA k-points)
  t: 1.0            # Hopping parameter
  U: 4.0            # On-site Hubbard interaction
  sizeA: 1          # Impurity fragment size (number of sites)
  Np: 1.0           # Particle density (electrons per site)
  bc: 1             # Boundary conditions: 1 = periodic ring, 0 = open chain

input:
  input_case: "non_int"   # Start from non-interacting Green's function
  # Other options: "from_file" to restart from a previous calculation

embedding:
  max_iter: 300           # Maximum number of DMFT iterations
  num_poles: 4            # Number of poles M in the SOP representation
  mu_fixed: 0.5           # Fixed chemical potential (null = updated self-consistently)
  p_type: "sqrt"          # Residue parametrization: "std" or "sqrt"
  axis: "shift"           # Frequency axis: "shift", "erf", or "imaginary"
  eta_axis: 0.5           # Imaginary shift η (used for "shift" and "erf" axes)
  num_pts: 10000          # Number of frequency grid points
  beta_T: 1500.0          # Inverse temperature β (for Matsubara frequencies)
  Nw_max: 3000            # Maximum Matsubara frequency index
  w_edges: [-10, 10]      # Frequency window [ω_min, ω_max]
  sparse_gs: true         # Use sparse solver for ground-state search
  gs_search: "std"        # Ground-state method: "std" or "subspaces"
  solver_method: "std"    # G_imp method: "std" (full ED) or "lanczos"

optimization:
  mixing_method: "linear" # Self-consistency mixing: "linear"
  alpha: 0.5              # Linear mixing parameter α ∈ (0, 1]
  opt_method: "scipy_CG"  # Minimizer: "scipy_CG", "custom_CG", or "anal_SD"
  opt_params: [100, 1e-5] # [max_iterations, tolerance]
  complex_poles: false    # Allow complex poles
  herm_residues: true     # Enforce Hermitian residues
  fixed_residues: false   # Keep residues fixed during optimization
  odd_spectrum: false     # Enforce particle-hole symmetry
  paramagnetic: true      # Assume spin-degenerate (single-site fragment only)
  p0_start: "always"      # Reuse previous SOP parameters: "always", "never", "no_first_iter"
  thr_diff_prev: 1e-8     # Convergence threshold on DOS difference
  thr_stagnation: 1e-9    # Stagnation detection threshold
  RMSE_thr: 0.5           # RMSE warning threshold for SOP fitting
```

## Project structure

```
sop_lake/
├── src/                            # Main source code
│   ├── __init__.py                 # Package exports
│   ├── dmft_main.py                # Entry-point script
│   ├── dmft_simulation.py          # Main DMFT self-consistent loop
│   ├── dmft_config.py              # Configuration dataclasses and YAML loader
│   │
│   ├── SOP.py                      # Sum-over-poles class and parameter utilities
│   ├── AIMSOP_utils.py             # AIM-SOP matrix construction and reversed inversion
│   ├── embedding_utils.py          # Self-consistency, frequency axes, mixing
│   ├── mbAIMSOP_solver.py          # Many-body exact diagonalization solver
│   │
│   ├── hubbard.py                  # 1D Hubbard Hamiltonian (via qiskit_nature)
│   ├── lanczos.py                  # Bi-Lanczos algorithm for Green's functions
│   ├── mb_utils.py                 # Many-body operators, Slater determinants, GF utilities
│   │
│   ├── dyn_poles.py                # SOP fitting via cost-function minimization
│   ├── dyn_poles_utils.py          # Bounds, peak finding, parameter utilities
│   ├── cost_fn.py                  # Cost function and gradient (matrix residues)
│   ├── cost_fn_eff.py              # Cost function and gradient (scalar/paramagnetic)
│   ├── cost_fn_nogrid.py           # Grid-free cost function via Lorentzian overlaps
│   │
│   ├── data_io.py                  # JSON I/O, result reading, and plotting
│   └── utils.py                    # Matrix checks, numerical utilities, sparse helpers
│
├── tests/                          # Unit and integration tests
│   ├── __init__.py
│   ├── test_models.py              # Tests for SOP class and Hubbard model
│   ├── test_solvers.py             # Tests for utilities, embedding, and Lanczos
│   └── test_integration.py        # Tests for configuration loading and DMFT setup
│
├── docs/                           # Documentation
│   ├── api.md                      # API reference
│   ├── theory.md                   # Theoretical background
│   ├── examples.md                 # Usage examples
│   └── architecture.md             # Code architecture overview
│
├── examples/
│   └── config_hubbard_imaginary.yaml   # Example configuration (Matsubara axis)
│
├── pyproject.toml                  # Package metadata
├── setup.py                        # Build script
├── environment.yml                 # Conda environment
├── LICENSE
└── README.md
```

## Algorithm overview

The DMFT self-consistency cycle implemented in `dmft_simulation.run()`:

```
1. Initialize frequency grid and starting embedding potential v_emb(ω)

2. For each iteration:
   a) Fit v_emb to SOP:
      minimize  J = Σ_ω ‖v_emb(ω) − Σ_k Γ_k/(ω − σ_k)‖²
      (enforcing Hermitian residues, real poles, and other constraints)

   b) Build auxiliary AIM Hamiltonian:
      H_AIM = H_A ⊗ I_bath
             + Σ_k Σ_{ij} Γ_k^{1/2}_{ij} (c_i† c_{j,k} + h.c.)
             + Σ_k σ_k Σ_i c†_{i,k} c_{i,k}

   c) Solve impurity problem (ED or Lanczos):
      → G_imp(ω) as SOP

   d) Self-consistency via Dyson equation:
      Σ_A(ω) = G_0,imp⁻¹(ω) − G_imp⁻¹(ω)
      G_loc(ω) = (1/Nk) Σ_k [(ω + μ − ε_k)I − Σ_A(ω)]⁻¹
      v_emb^new(ω) = ω + μ − h_A − Σ_A(ω) − G_loc⁻¹(ω)

   e) Linear mixing: v_emb ← α v_emb^new + (1−α) v_emb^old

3. Check convergence: DOS difference between consecutive G_imp < thr_diff_prev
```

## Key modules

### `SOP` class ([src/SOP.py](src/SOP.py))

Stores and evaluates a sum-over-poles rational function G(ω) = Σ_k Γ_k / (ω − σ_k):

```python
from src.SOP import SOP
import numpy as np

# Single-pole 2×2 example
Gamma = [np.eye(2, dtype=complex)]
sigma = [0.5]
sop = SOP(Gamma, sigma, p_type="std")

w_list = np.linspace(-5, 5, 1000).tolist()
G_vals = sop.evaluate(w_list)   # shape (1000, 2, 2)
```

Two parametrizations are available:
- `p_type="std"`: stores Γ_k directly, evaluates Σ_k Γ_k / (ω − σ_k)
- `p_type="sqrt"`: stores √Γ_k, evaluates Σ_k (√Γ_k)² / (ω − σ_k) — ensures positive-semidefinite residues

### Configuration system ([src/dmft_config.py](src/dmft_config.py))

Four composable dataclasses (`Hubbard_system_config`, `input_config`, `embedding_config`, `optimization_config`) bundled in `sim_config`. Load from YAML with:

```python
from src.dmft_config import load_sim_config
config = load_sim_config("path/to/config.yaml")
```

### Impurity solver ([src/mbAIMSOP_solver.py](src/mbAIMSOP_solver.py))

Builds the many-body AIM Hamiltonian and solves for G_imp via exact diagonalization. Supports sparse ground-state search and the bi-Lanczos continued-fraction method for large systems.

### Embedding and self-consistency ([src/embedding_utils.py](src/embedding_utils.py))

Contains:
- `frequency_axis()` — constructs real, shifted, or Matsubara frequency grids
- `self_consistency_DMFT()` — computes Σ_A and G_loc from G_imp and the embedding potential
- `get_SigmaA_SOP()`, `get_Gloc_SOP()`, `get_new_vemb_SOP()` — algebraic SOP-level operations using `reversed_AIMSOP`

## Running tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run a specific module
pytest tests/test_models.py -v
```

## Documentation

- [API Reference](docs/api.md) — full function and class documentation
- [Theory](docs/theory.md) — theoretical background and methodology
- [Examples](docs/examples.md) — worked examples
- [Architecture](docs/architecture.md) — code design overview

## Citation

If you use this code in your research, please cite:

```bibtex
@software{carbone_sop_lake_2025,
  author    = {Alessandro Carbone},
  title     = {sop\_lake: Dynamical Mean-Field Theory via Sum-Over-Poles Embedding},
  year      = {2025},
  url       = {https://github.com/AlessandroCarbone/sop_lake}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Issues and support

- Report bugs via [GitHub Issues](https://github.com/AlessandroCarbone/sop_lake/issues)
- Ask questions in [GitHub Discussions](https://github.com/AlessandroCarbone/sop_lake/discussions)

## Author

Alessandro Carbone ([alessandro.carbone@epfl.ch](mailto:alessandro.carbone@epfl.ch))
EPFL, Lausanne, Switzerland

## References

- A. Georges, G. Kotliar, W. Krauth, M. J. Rozenberg, "Dynamical mean-field theory of strongly correlated fermion systems and the limit of infinite dimensions," *Rev. Mod. Phys.* **68**, 13 (1996)
- M. Caffarel, W. Krauth, "Exact diagonalization approach to correlated fermions in infinite dimensions: Mott transition and superconductivity," *Phys. Rev. Lett.* **72**, 1545 (1994)
- R. Haydock, V. Heine, M. J. Kelly, "Electronic structure based on the local atomic environment for tight-binding bands," *J. Phys. C* **5**, 2845 (1972) [Lanczos recursion]

---

**Last updated:** March 2026
