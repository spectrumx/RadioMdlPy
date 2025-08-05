"""
Astronomical modeling functions for radio astronomy calculations
"""

import numpy as np
import pandas as pd

# Constants (copied from RadioMdl to avoid circular import)
k_boltz = 1.380649e-23  # Boltzmann's constant in J/K


def estim_temp(flux: float, effective_aperture: float) -> float:
    """
    Estimates the temperature of a point-like source from its flux and the antenna effective
    aperture. Flux must be in Jansky.
    """
    return flux * 1e-26 / (2 * k_boltz) * effective_aperture


def power_to_temperature(power: float, bandwidth: float) -> float:
    """
    Convert power (in watts) to temperature (in Kelvin) given bandwidth (in hertz)
    """
    return power / (k_boltz * bandwidth)


def temperature_to_power(temp: float, bandwidth: float) -> float:
    """
    Convert temperature (in Kelvin) to power (in watts) given bandwidth (in hertz)
    """
    return k_boltz * bandwidth * temp


def temperature_to_flux(temp: float, effective_aperture: float) -> float:
    """
    Convert temperature (K) and effective aperture (m^2) to flux in Jansky (Jy)
    Matches Julia version.
    """
    return 2 * k_boltz * temp / effective_aperture * 1e26  # in Jy


def estim_casA_flux(center_freq: float) -> float:
    """
    Estimate the flux of Cas A at a given frequency (Hz), based on Baars et al. 1977.
    Matches Julia version, including decay since 1980.
    """
    # decay in %/year since 1980
    decay = 0.97 - 0.3 * np.log10(center_freq * 1e-9)
    # 43 years since 1980 (as in Julia)
    return 10 ** (5.745 - 0.770 * np.log10(center_freq * 1e-6)) * (1 - decay * 43 / 100)


def estim_virgoA_flux(center_freq: float) -> float:
    """
    Estimate the flux of Virgo A at a given frequency (Hz), matching Julia version.
    """
    return 10 ** (5.023 - 0.856 * np.log10(center_freq * 1e-6))

def antenna_mdl_ITU(gain_max, half_beamwidth, alphas, betas):
    """
    Create ITU recommended gain profile.
    Returns a DataFrame with columns: 'alphas', 'betas', 'gains'
    """
    alphas = np.array(alphas)
    betas = np.array(betas)
    gain_profile = np.zeros_like(alphas, dtype=float)

    # Define region boundaries
    parts = [
        0,
        half_beamwidth * np.sqrt(17/3),
        10 ** ((49 - gain_max) / 25),
        48,
        80,
        120,
        180
    ]

    # Find indices for each region
    part1 = np.where((alphas >= parts[0]) & (alphas < parts[1]))[0]
    part2 = np.where((alphas >= parts[1]) & (alphas < parts[2]))[0]
    part3 = np.where((alphas >= parts[2]) & (alphas < parts[3]))[0]
    part4 = np.where((alphas >= parts[3]) & (alphas < parts[4]))[0]
    part5 = np.where((alphas >= parts[4]) & (alphas < parts[5]))[0]
    part6 = np.where((alphas >= parts[5]) & (alphas <= parts[6]))[0]

    # Calculate gain profile in dB
    gain_profile[part1] = gain_max - 3 * (alphas[part1] / half_beamwidth) ** 2
    gain_profile[part2] = gain_max - 20
    # Avoid log10(0) by masking zeros
    with np.errstate(divide='ignore'):
        gain_profile[part3] = 29 - 25 * np.log10(alphas[part3])
    gain_profile[part4] = -13
    gain_profile[part5] = -8
    gain_profile[part6] = -13

    # Convert gain profile from dB to linear scale
    gain_linear = 10 ** (gain_profile / 10)

    # Build DataFrame: all combinations of alphas and betas
    n_alpha = len(alphas)
    n_beta = len(betas)
    df = pd.DataFrame({
        'alphas': np.tile(alphas, n_beta),
        'betas': np.repeat(betas, n_alpha),
        'gains': np.tile(gain_linear, n_beta)
    })

    return df


def flux_to_temperature(flux: float, freq: float, bw: float) -> float:
    """
    Convert flux to temperature
    """
    return flux * (freq**2) / (2 * k_boltz * bw)  # in K
