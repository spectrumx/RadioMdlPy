### Directory Structure

The Python implementation is organized as follows:
- **Binary files**: Some components requires heavy computations, thus various `.npy` files, which are bunary NumPy arrays, are pre-computed so that they can be loaded to reduce computation time instead of direct computations
- **tuto_mdl_obs_python_simplified.ipynb**: Jupyter notebook for python which produces the same results with the Julia notebook (`Julia/test/tuto_mdl_obs_modified.ipynb`)
- **test_tuto_mdl_obs_python_simplified.py**: Python script to run in a command line interface (CLI) that is equivalent to the above notebook
- ***_simplified_load.***: Thes Jupyter notebook and python script are basically the same with above codes but these will load pre-computed binary files (`.npy`) instead of direct computions for some parts, thus these will generate result much faster than above codes
- ***_250401.***: Testing Jupyter notebook and python script with newly generated Arrow files from the data creation scripts for different time span (see **Data Creation Scripts** section below)
- **Data directory**: `tutorial/data/` - Contains input data files
- **Data creation directory**: `tutorial/data_creation/` - Contains scripts to generate input data files, .arrow, which are for getting trajectories of a star and a satellite with user-specified date and time. Currently, Cas A (Cassiopeia A) and Starlink satellite are used.


### Data Files

The `src/python/data/` directory contains input data files of simulations including:
- One **.cut** file: Gain pattern of the MIT Westford antenna generated from TRICA software
- Two **.arrow** files: Trajectory files for astronomical objects (e.g., Cas A) and satellites (e.g., Starlink)


### Data Creation Scripts

The `src/python/data_creation/` directory contains Python scripts that generate Arrow input data files (trajectory files):
- **Stars**: Currently supports Cas A trajectory calculations
- **Satellites**: Currently supports Starlink trajectory calculations

The `src/python/data_creation/traj_files` directory contains two input files for python scripts to generate Arrow files and it also stores generated arrow files:
- **de421.bsp**: positions for planets and their moons for time spans, e.g., https://rhodesmill.org/skyfield/planets.html
- **hipparcos.dat**: Hipparcos catalogue


### Usage

The Python implementation can be used through:
- Direct Python scripts in CLI (e.g., `test_tuto_mdl_obs_python_simplified_load.py`)
- Jupyter notebooks (e.g., `tuto_mdl_obs_python_simplified_load.ipynb`)
