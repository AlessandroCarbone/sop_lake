# Contributing to sop_lake

Thank you for your interest in contributing to sop_lake! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project and everyone participating is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, check the [issue list](https://github.com/AlessandroCarbone/sop_lake/issues) as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include screenshots/output if possible**
* **Include your environment details** (Python version, OS, dependency versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and expected behavior**
* **Explain why this enhancement would be useful**

### Pull Requests

* Follow the Python style guide (PEP 8)
* Include appropriate test cases
* Update documentation as needed
* Add an entry to CHANGELOG.md

**Process:**
1. Fork the repository and create your feature branch (`git checkout -b feature/AmazingFeature`)
2. Make your changes
3. Write or update tests
4. Run tests and ensure they pass: `pytest`
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## Development Setup

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/AlessandroCarbone/sop_lake.git
cd sop_lake

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_models.py::test_function_name -v
```

### Code Style

We use [Black](https://github.com/psf/black) for code formatting and [Flake8](https://flake8.pycqa.org/) for linting.

```bash
# Format code with Black
black src/ tests/

# Check code style with Flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

## Project Structure

```
sop_lake/
├── src/                    # Main source code
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── examples/               # Example scripts and configs
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── README.md              # Project README
├── CONTRIBUTING.md        # This file
└── CHANGELOG.md           # Version history
```

## Documentation Style

* Use docstrings for all public functions and classes
* Follow NumPy docstring style
* Include type hints when possible

Example:
```python
def calculate_gf(beta: float, n_iw: int) -> np.ndarray:
    """
    Calculate the Green's function.
    
    This function computes the Green's function at Matsubara frequencies.
    
    Parameters
    ----------
    beta : float
        Inverse temperature (1/T).
    n_iw : int
        Number of Matsubara frequencies.
    
    Returns
    -------
    np.ndarray
        Green's function values at Matsubara frequencies.
    
    Notes
    -----
    Uses the convention: iω_n = (2n+1)π/β
    
    Examples
    --------
    >>> gf = calculate_gf(beta=100.0, n_iw=256)
    >>> gf.shape
    (256,)
    """
```

## Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

Example:
```
Add support for extended Hubbard model

- Implement extended interaction terms
- Add tests for the new model parameters
- Update documentation with examples

Closes #123
```

## Questions?

Feel free to:
* Create an issue with the `[Question]` tag
* Start a discussion in GitHub Discussions
* Contact the maintainer directly

## Recognition

Contributors will be recognized in the README and in releases.

---

Thank you for contributing! 🚀
