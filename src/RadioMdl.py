"""
RadioMdl - A Python module for radio modeling calculations
"""

# Constants
k_boltz = 1.380649e-23  # Boltzmann's constant in J/K
rad = 3.141592653589793 / 180  # degree to radian conversion factor
speed_c = 3e8  # speed of light in m/s

# Note: We don't import other modules here to avoid circular imports
# Instead, import them directly in your scripts when needed

# Export constants and basic functions
__all__ = [
    'k_boltz',
    'rad',
    'speed_c',
    # Classes and functions should be imported directly from their modules
    # 'Antenna', 'Instrument', 'Observation', 'Constellation', 'Trajectory',
    # 'estim_temp', 'estim_casA_flux', 'power_to_temperature',
    # 'antenna_mdl_ITU', 'sat_link_budget', 'model_observed_temp'
]