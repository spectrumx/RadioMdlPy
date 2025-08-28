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

### `modular_tutorials/` directory: Recommanded for learning

This directory contains a series of focused, modular tutorials for learning radio astronomy observation modeling with the RadioMdlPy framework. Each tutorial builds upon the previous ones, providing a progressive learning experience. Details can be found in the `README.md` inside the `modular_tutorials/`.

### `tutorial/` directory

This contains additional learning resources and examples and serves as a reference implementation for the modeling and simulation of radio astronomy observations. It consists of Jupyter Notebooks and their CLI scripts, and also includes data creation scripts for trajectory files of a star and satellite at the `tutorial/data_creation/` directory. Details can be found at the `README.md` inside the `tutorial/` directory.

## Installation

It's recommended to create a virtual environment before installing the package:

```bash
# Clone the repository at a directory
git clone https://github.com/spectrumx/RadioMdlPy.git

# Create and activate virtual environment (recommended)
python -m venv radiomdlpy_env
# On Windows:
radiomdlpy_env\Scripts\activate
# On macOS/Linux:
source radiomdlpy_env/bin/activate

# change directory to RadioMdlPy and install dependencies (packages)
cd RadioMdlPy
pip install -e .

# When you finish working with RadioMdlPy, deactivate the virtual environment
deactivate
```

### Dependencies

The package automatically installs the following dependencies:
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `pandas>=1.3.0` - Data manipulation
- `pyarrow>=6.0.0` - Data serialization
- `numba>=0.56.0` - JIT compilation for performance

## Usage

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

## License

MIT License