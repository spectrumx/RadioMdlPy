"""
Antenna pattern functions for radio modeling
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid

# Constants (copied from RadioMdl to avoid circular import)
rad = 3.141592653589793 / 180  # degree to radian conversion factor
speed_c = 3e8  # speed of light in m/s


def map_sphere(pattern: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> tuple:
    """
    Map antenna pattern to sphere coordinates.
    Assumes alphas and betas are degrees. Output is converted in radians.
    """
    # Form 2D matrix for interpolation argument
    # Add first column as last column to loop azimuth coordinates

    reshaped = pattern.reshape(len(alphas), len(betas), order='F')
    first_col = reshaped[:, 0]
    gain_map = np.column_stack([reshaped, first_col])

    # Generate sampling coordinates
    a = alphas * rad
    b = np.append(betas, 360) * rad

    return gain_map, a, b


def radiated_power_to_gain(rad_pow: np.ndarray, alphas: np.ndarray,
                         betas: np.ndarray, eta_rad: float = 1.0) -> np.ndarray:
    """
    Yields the gain pattern of an antenna, in dB, given a radiated power pattern.
    It is assumed that the radiated power includes the radiation efficiency.
    The angles must be in degrees.
    """

    # Map the radiated power for interpolation
    rad_pow_map, a, b = map_sphere(rad_pow, alphas, betas)

    # Integrate over the sphere
    rad_pow_avg = trapezoid(trapezoid(rad_pow_map * np.sin(a)[:, np.newaxis], b), a) / (4 * np.pi)

    # Directivity
    dir = rad_pow / rad_pow_avg

    # Gain
    return eta_rad * dir


def interpolate_gain(gain: np.ndarray, alphas: np.ndarray,
                    betas: np.ndarray) -> RegularGridInterpolator:
    """
    Create gain interpolator from gain pattern and angles
    """
    # Map the gain for interpolation
    gain_map, a, b = map_sphere(gain, alphas, betas)

    # Create gain function of angles in antenna coord. system
    return RegularGridInterpolator((a, b), gain_map, method='linear')


def gain_to_effective_aperture(gain: float, frequency: float) -> float:
    """
    Convert gain to effective aperture
    """
    wavelength = speed_c / frequency
    return gain * (wavelength**2 / (4 * np.pi))
