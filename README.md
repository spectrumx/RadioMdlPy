# RadioMdlPy

A Python package focused on the simulation of radio observation, e.g. for radio astronomy. Originally, it is a Julia package created by Dr. Samuel Thé (https://github.com/SJJThe). Modified version of the Julia codes are found at `Julia/` directory.

## Requirements

- **Python**: 3.9 or later
- **Dependencies**: numpy, scipy, pandas, pyarrow, numba

## Installation

### Recommended: Use a Virtual Environment

It's recommended to create a virtual environment before installing the package:

```bash
# Create a virtual environment
python -m venv radiomdlpy_env

# Activate the virtual environment
# On Windows:
radiomdlpy_env\Scripts\activate
# On macOS/Linux:
source radiomdlpy_env/bin/activate

# Install the package
pip install RadioMdlPy

# When you're done, deactivate the virtual environment
deactivate
```

### Installation from GitHub Repository

#### Option 1: Install from GitHub Releases (Recommended for End Users)

If the package has been released on GitHub, you can install it directly:

```bash
# Install from GitHub releases (if available)
pip install https://github.com/spectrumx/RadioMdlPy/releases/latest/download/radiomdlpy-1.0.0-py3-none-any.whl
```

#### Option 2: Install from GitHub Repository (Development Version)

To install the latest development version directly from GitHub:

```bash
# Install directly from GitHub repository
pip install git+https://github.com/spectrumx/RadioMdlPy.git
```

#### Option 3: Clone and Install (For Development)

To install from the source repository for development:

```bash
# Clone the repository
git clone https://github.com/spectrumx/RadioMdlPy.git
cd RadioMdlPy

# Create and activate virtual environment (recommended)
python -m venv radiomdlpy_env
# On Windows:
radiomdlpy_env\Scripts\activate
# On macOS/Linux:
source radiomdlpy_env/bin/activate

# Install in development mode (editable install)
pip install -e .
```

**Benefits of development mode:**
- Changes to source code are immediately available without reinstalling
- Perfect for development, testing, and contributing
- Installs all dependencies automatically

### Dependencies

The package automatically installs the following dependencies:
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `pandas>=1.3.0` - Data manipulation
- `pyarrow>=6.0.0` - Data serialization
- `numba>=0.56.0` - JIT compilation for performance

## Usage

### For End Users (Installed Package)

When you install RadioMdlPy from a wheel, use it like this:

```python
# Import specific modules
from RadioMdlPy import obs_mdl, radio_types, astro_mdl

# Import specific functions
from RadioMdlPy.obs_mdl import model_observed_temp
from RadioMdlPy.radio_types import Antenna, Instrument, Observation

# Use functions
result = model_observed_temp(observation, sky_model, constellation)
```

### For Development (From Source)

When working with the source code directly:

```python
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import modules directly
from radio_types import Antenna, Instrument, Observation
from obs_mdl import model_observed_temp
from astro_mdl import estim_casA_flux, power_to_temperature

# Use functions
result = model_observed_temp(observation, sky_model, constellation)
```

## Modules

- `RadioMdl`: Core constants and utilities
- `antenna_pattern`: Antenna pattern calculations
- `astro_mdl`: Astronomical modeling
- `coord_frames`: Coordinate frame transformations
- `obs_mdl`: Observation modeling
- `radio_io`: Radio I/O operations
- `radio_types`: Radio data types
- `sat_mdl`: Satellite modeling

## Development

### Setup Development Environment

1. **Create and activate a virtual environment:**

```bash
# Create a virtual environment
python -m venv radiomdlpy_dev

# Activate the virtual environment
# On Windows:
radiomdlpy_dev\Scripts\activate
# On macOS/Linux:
source radiomdlpy_dev/bin/activate
```

2. **Install in Development Mode**

To install the package in development mode (editable install):

```bash
pip install -e .
```

### Install with Development Dependencies

To install with development tools (pytest, black, flake8):

```bash
pip install -e ".[dev]"
```

## Building

### Build from Source

To build a wheel distribution, navigate to the project root directory (where `pyproject.toml` is located) and run:

```bash
# Make sure you're in the RadioMdlPy directory
cd /path/to/RadioMdlPy

# Build the package
python -m build
```

This will create both a wheel (`.whl`) and source distribution (`.tar.gz`) in the `dist/` directory.

### Prerequisites for Building

Make sure you have the build tools installed:

```bash
pip install build
```

## License

MIT License