"""
Input/Output functions for radio modeling
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import warnings

def power_pattern_from_cut_file(file_path: str, free_sp_imp: float = 377,
                              verb: bool = False) -> pd.DataFrame:
    """
    Yields the radiated power pattern, in dBW, of an antenna, times the radiation
    efficiency from the `.cut` file containing co- and cross-polarization E-field.
    Headers in the file are below a line starting with `Field`. It is composed of
    the starting value of the declination angle α, the step and number of samples of
    α and the value of the azimuthal angle β.
    """

    # parse file
    patterns = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    k = 0
    while k < len(lines):
        # Find the next header line starting with "Field"
        while k < len(lines) and not lines[k].startswith("Field"):
            k += 1
        if k >= len(lines):
            break

        # The next line after "Field ..." is the parameter line
        param_line_idx = k + 1
        if param_line_idx >= len(lines):
            break  # No parameter line after header

        param_line = lines[param_line_idx].split()
        α_start = float(param_line[0])
        α_step = float(param_line[1])
        nb_α = int(float(param_line[2]))
        θ = float(param_line[3])

        # For the next nb_α lines, read the data
        for t in range(nb_α):
            data_idx = param_line_idx + 1 + t
            if data_idx >= len(lines):
                break  # Avoid out-of-bounds
            data_line = lines[data_idx].split()
            vals = np.array([float(x) for x in data_line[:4]])
            u = np.sum(vals ** 2) / (2 * 377)  # Use your free_sp_imp value if needed
            α = α_start + t * α_step
            patterns.append([α, θ, u])

        # Move to the next block
        k = param_line_idx + 1 + nb_α

    # Convert to DataFrame
    pattern = pd.DataFrame(patterns, columns=['alpha', 'beta', 'power'])

    # Rounding as in Julia
    decimal_places = max(0, -int(np.floor(np.log10(abs(α_step - round(α_step))))))
    pattern['alpha'] = pattern['alpha'].round(decimal_places)
    print(pattern.head())

    warnings.warn("This function assumes Daniel Sheen generated files")

    # Check that α ∈ [-180,180[ and β ∈ [0, 180[
    pattern = pattern[
        (pattern['alpha'] >= -180.0) & (pattern['alpha'] < 180.0) &
        (pattern['beta'] >= 0.0) & (pattern['beta'] < 180.0)
    ]

    # Move the origin of β so that, when telescope points at the horizon, the
    # first slice (for the new β = 0) is vertical with α > 0 oriented towards
    # the ground.
    pattern['beta'] = np.mod(pattern['beta'] + 90.0, 180.0)
    pattern.loc[pattern['beta'] >= 90.0, 'alpha'] *= -1.0
    pattern.loc[pattern['alpha'] == 180.0, 'alpha'] *= -1.0

    # Change evolution domains so that α ∈ [0,180] and β ∈ [0, 360]
    pattern.loc[pattern['alpha'] <= 0, 'beta'] += 180
    rng_beta = pattern[pattern['alpha'] == pattern['alpha'].max()]['beta']
    pattern.loc[pattern['alpha'] < 0, 'alpha'] *= -1
    pattern['alpha'] = np.abs(pattern['alpha'])

    # Add points for α = 0 and α = 180
    for i in rng_beta:
        pattern = pd.concat([pattern, pd.DataFrame({
            'alpha': [0.0],
            'beta': [i],
            'power': [pattern[pattern['alpha'] == 0.0]['power'].iloc[0]]
        })], ignore_index=True)
        pattern = pd.concat([pattern, pd.DataFrame({
            'alpha': [180.0],
            'beta': [i],
            'power': [pattern[pattern['alpha'] == 180.0]['power'].iloc[0]]
        })], ignore_index=True)

    pattern = pattern.sort_values(['beta', 'alpha'])

    return pattern
