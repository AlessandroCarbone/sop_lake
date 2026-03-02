"""
Quick start guide for sop_lake
"""

# ==============================================================
# INSTALLATION
# ==============================================================

# From GitHub repository:
# git clone https://github.com/AlessandroCarbone/sop_lake.git
# cd sop_lake
# pip install -e .

# With development tools:
# pip install -e ".[dev]"

# With documentation tools:
# pip install -e ".[docs]"

# ==============================================================
# MINIMAL EXAMPLE
# ==============================================================

from src.dmft_config import load_sim_config
from src.dmft_simulation import dmft_simulation

# Load your configuration
config = load_sim_config("examples/config_hubbard_single_band.yaml")

# Create simulation object
sim = dmft_simulation(config)

# Run the simulation
sim.run(output_file_names={
    "dmft": "dmft_results.json",
    "vemb": "vemb_results.json", 
    "conv": "convergence.json"
})

# ==============================================================
# RUNNING TESTS
# ==============================================================

# All tests:
# pytest tests/ -v

# Specific test file:
# pytest tests/test_models.py -v

# With coverage:
# pytest tests/ --cov=src --cov-report=html

# ==============================================================
# CODE FORMATTING
# ==============================================================

# Format code with Black:
# black src/ tests/

# Check with Flake8:
# flake8 src/ tests/

# Type checking:
# mypy src/

# ==============================================================
# BUILDING DOCUMENTATION
# ==============================================================

# cd docs
# sphinx-build -b html . _build/html
# # Open _build/html/index.html in browser

# ==============================================================
# KEY FILES TO MODIFY FOR YOUR PROJECT
# ==============================================================

# 1. README.md - Update with your project details
# 2. setup.py - Update author and contact info
# 3. LICENSE - Replace with your preferred license
# 4. src/dmft_main.py - Entry point for CLI
# 5. docs/theory.md - Add your theoretical background
# 6. examples/*.yaml - Add your own configurations
