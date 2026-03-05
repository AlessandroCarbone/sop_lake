# Code Architecture and Design

## Directory Structure

```
sop_lake/
├── src/
│   └── sop_lake/
│       ├── __init__.py                    # Package initialization
│       ├── dmft_main.py                   # CLI entry point
│       ├── dmft_simulation.py             # DMFT loop implementation
│       ├── dmft_config.py                 # Configuration loader
│       ├── mbAIMSOP_solver.py             # AIM solver implementation
│       ├── SOP.py                         # Sum-of-poles representation
│       ├── AIMSOP_utils.py                # AIM-SOP utilities
│       ├── hubbard.py                     # Hubbard model definition
│       ├── dyn_poles.py                   # Dynamic poles calculation
│       ├── dyn_poles_utils.py             # Dynamic poles utilities
│       ├── embedding_utils.py             # Embedding optimization
│       ├── lanczos.py                     # Lanczos algorithm
│       ├── cost_fn.py                     # Cost functions
│       ├── cost_fn_eff.py                 # Effective cost functions
│       ├── cost_fn_nogrid.py              # Grid-free cost functions
│       ├── mb_utils.py                    # Many-body utilities
│       ├── data_io.py                     # I/O utilities
│       └── utils.py                       # General utilities
├── tests/                            # Test suite
│   ├── test_models.py               # Model tests
│   ├── test_solvers.py              # Solver tests
│   └── test_integration.py          # integration tests
├── docs/                             # Documentation
│   ├── api.md                       # API reference
│   ├── examples.md                  # Usage examples
│   ├── theory.md                    # Theory background
│   └── architecture.md              # This file
├── examples/                         # Example configs
│   ├── config_hubbard_single_band.yaml
│   └── config_hubbard_extended.yaml
├── .github/workflows/               # CI/CD
│   ├── tests.yml
│   └── docs.yml
├── setup.py                         # Package setup
├── pyproject.toml                   # Project metadata
├── requirements.txt                 # Dependencies
├── README.md                        # Main readme
├── CONTRIBUTING.md                  # Contribution guide
├── CHANGELOG.md                     # Version history
├── LICENSE                          # MIT License
└── .gitignore                       # Git ignore rules
```

## Core Components

### 1. DMFT Loop (`dmft_simulation.py`)
- Manages self-consistent iterations
- Coordinates between different components
- Handles convergence checking
- Manages output operations

### 2. Configuration System (`dmft_config.py`)
- Loads YAML configuration files
- Validates parameters
- Provides default values
- Type conversion and checking

### 3. Impurity Solver (`mbAIMSOP_solver.py`, `SOP.py`, `AIMSOP_utils.py`)
- Solves effective (single-site or cluster) impurity problem
- Uses many-body AIM-SOP to find the auxiliary Hamiltonian for the impurity problem
- `SOP.py` implements the sum-of-poles representation of Green's functions
- `AIMSOP_utils.py` provides helper routines for the AIM-SOP procedure
- Extracts local self-energy and Green's functions

### 4. Model Definitions (`hubbard.py`)
- Defines lattice models
- Constructs Hamiltonian
- Defines interactions

### 5. Numerical Methods

#### Lanczos algorithm (`lanczos.py`)
- Efficient sparse eigensolver
- Used for diagonalization

#### Dynamics of poles (`dyn_poles.py`, `cost_fn.py`, `cost_fn_eff.py`, `cost_fn_nogrid.py`)
- Evolve the parameters of the simulation
- Cost function minimization and definition
- `cost_fn_nogrid.py` provides a grid-free variant of the cost function
- Takes care of what typically is the fitting strategy of a DMFT self-consistent cycle

### 6. I/O and Utilities
- Configuration and data I/O (`data_io.py`)
- General utilities (`utils.py`)
- Many-body tools (`mb_utils.py`)
- Additional tools related to the quantum embedding procedure (`embedding_utils.py`)


## Data Flow

```
Config File (YAML)
    ↓
dmft_config.py (parse)
    ↓
dmft_simulation.py (main loop)
    ├→ hubbard.py (model)
    ├→ dmft_loop
    │   ├→ mbAIMSOP_solver.py (solve)
    │   ├→ dyn_poles.py (extract)
    │   └→ lanczos.py (diagonalize)
    ├→ dyn_poles_utils.py (analyze)
    ├→ embedding_utils.py (optimize)
    ├→ cost_fn.py (evaluate)
    └→ data_io.py (save)
    ↓
Output Files (JSON)
```

## Design Patterns

### 1. Configuration-Driven
- Single YAML config file
- No hardcoded parameters
- Easy reproducibility

### 2. Modular Components
- Decoupled solvers
- Interchangeable implementations
- Plugin architecture possible

### 3. Error Handling
- Comprehensive logging
- Graceful degradation
- Clear error messages

### 4. Testing Strategy
- Unit tests for components
- Integration tests for workflows
- Regression tests for reproducibility

## Future Extensions

### Possible Improvements
1. Support for other impurity solvers (CT-QMC, NCA, etc.)
2. Parallelization over frequencies/k-points
3. GPU acceleration
4. More models (t-J, Kondo, etc.)
5. MPI support for distributed computing

### Plugin System
The code is designed to support plugin-based impurity solvers:
```python
class CustomSolver(BaseSolver):
    def solve(self, weiss_field):
        # Custom solution method
        pass
```

## Development Guidelines

1. **Keep components focused** - single responsibility
2. **Use configuration** - avoid magic numbers
3. **Write tests** - especially for numerical code
4. **Document interfaces** - clear user expectations
5. **Log important events** - help with debugging
