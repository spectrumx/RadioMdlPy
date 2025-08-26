# RadioMdlPy

RadioMdlPy is a sophisticated Python framework designed for modeling radio astronomy observations, with particular emphasis on simulating the impact of satellite constellations (such as Starlink) on radio telescope observations. The framework provides end-to-end simulation capabilities from trajectory generation to power spectral density analysis, enabling researchers and astronomers to assess and mitigate interference from satellite mega-constellations.

<br />

Originally, it is a Julia package created by Dr. Samuel Thé (https://github.com/SJJThe). Modified version of the Julia codes are found at `Julia/` directory.

## Requirements

- **Python**: 3.9 or later
- **Dependencies**: numpy, scipy, pandas, pyarrow, numba

## Modules (at src/ directory)

- `RadioMdl.py`: Core constants and utilities
- `antenna_pattern.py`: Antenna pattern calculations
- `astro_mdl.py`: Astronomical modeling
- `coord_frames.py`: Coordinate frame transformations
- `obs_mdl.py`: Observation modeling
- `radio_io.py`: Radio I/O operations
- `radio_types.py`: Radio data types
- `sat_mdl.py`: Satellite modeling

## Tutorial Resources

The `tutorial/` directory contains additional learning resources and examples and serves as a reference implementation for the modeling and simulation of radio astronomy observations. It consists of Jupyter Notebooks and their CLI scripts, and also includes data creation scripts for trajectory files of a star and satellite at the `tutorial/data_creation/` directory. Details can be found at the README inside the `tutorial/` directory.

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

# When you're done, deactivate the virtual environment
deactivate
```

### Installation from GitHub Repository

#### Option 1: Clone and Install using Git (Recommended)

To install from the source repository:

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

# Install
pip install -e .
```

#### Option 2: Install from GitHub Releases (Recommended for End Users)

If the package has been released on GitHub, you can install it directly:

```bash
# Install from GitHub releases (if available)
pip install https://github.com/spectrumx/RadioMdlPy/releases/latest/download/radiomdlpy-1.2.0-py3-none-any.whl
```

#### Option 3: Install from GitHub Repository

To install the latest development version directly from GitHub:

```bash
# Install directly from GitHub repository
pip install git+https://github.com/spectrumx/RadioMdlPy.git
```

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