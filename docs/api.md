# API Reference

## Core Classes

### dmft_simulation
Main simulation class for running DMFT calculations.

```python
from src.dmft_simulation import dmft_simulation
sim = dmft_simulation(config)
sim.run(output_file_names=output_names)
```

## Configuration

### load_sim_config
Load simulation configuration from YAML files.

```python
from src.dmft_config import load_sim_config
config = load_sim_config("config.yaml")
```

## Models

### Hubbard Model
Standard one-dimensional single-band Hubbard model implementation.

## Solvers

### Full diagonalization solver
Uses full exact diagonalization to diagonalize the auxiliary Hamiltonian defined by the many-body algorithmic inversion theory.

### Bi-Lanczos method
Performs the bi-Lanczos algorithm to diagonalize the auxiliary Hamiltonian defined by the many-body algorithmic inversion theory.

## Utilities

### Dynamic Poles
Calculate dynamic poles and spectral functions.

### Lanczos Decomposition
Efficient eigenvalue decomposition using Lanczos algorithm.

## Data Management

---

For full function documentation, please refer to the docstrings in the source code.
