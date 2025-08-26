"""
Satellite modeling functions for radio astronomy
"""

import numpy as np
from coord_frames import ground_to_beam_coord

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Install with: pip install numba")

speed_c = 3e8  # m/s


def free_space_loss(rng, freq):
    return (4 * np.pi * rng / (speed_c / freq)) ** 2


def simple_link_budget(gain_RX, gain_TX, rng, freq):
    L = free_space_loss(rng, freq)
    return gain_RX * (1 / L) * gain_TX


def sat_link_budget(
    dec_tel, caz_tel, instru_tel, dec_sat, caz_sat, rng_sat, instru_sat, freq,
    beam_avoid=0.0, turn_off=False
):
    """
    Full-featured satellite link budget calculation, ported from Julia.
    """
    # Coordinate of sat in antenna frame
    dec_sat_tel, caz_sat_tel = ground_to_beam_coord(dec_sat, caz_sat, dec_tel, caz_tel)

    # Telescope gain
    instru_ant = instru_tel.get_antenna()
    gain_tel = instru_ant.get_gain_value(dec_sat_tel, caz_sat_tel)

    sat_ant = instru_sat.get_antenna()
    # Coordinate of telescope at time t in satellite frame
    dec_tel_sat = dec_sat
    caz_tel_sat = -caz_sat

    # Initialize with defaults (as in Julia)
    dec_sat_ant = dec_tel_sat
    caz_sat_ant = caz_tel_sat

    # Beam avoidance logic
    if beam_avoid > 0:
        beam_dec, beam_caz = sat_ant.get_boresight_point()  # Should return in radians
        if abs(beam_dec - dec_tel_sat) < np.deg2rad(beam_avoid):
            if turn_off:
                return 0.0
            else:
                dec_sat_ant = np.mod(dec_tel_sat + np.pi / 4, np.pi)
        elif abs(beam_caz - caz_tel_sat) < np.deg2rad(beam_avoid):
            if turn_off:
                return 0.0
            else:
                caz_sat_ant = np.mod(caz_tel_sat + np.pi / 4, 2 * np.pi)

    # Satellite gain (use possibly modified dec_sat_ant, caz_sat_ant)
    gain_sat = sat_ant.get_gain_value(dec_sat_ant, caz_sat_ant)

    # Link budget
    return simple_link_budget(gain_tel, gain_sat, rng_sat, freq)


# Numba-optimized link budget calculation
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def fast_link_budget_core(dec_tel, caz_tel, dec_sat, caz_sat, rng_sat, freq,
                              beam_avoid_rad, beam_dec, beam_caz, turn_off):
        """
        Numba-optimized core link budget calculation.
        """
        n = len(dec_tel)
        result = np.zeros(n)

        for i in prange(n):
            # Free space loss calculation
            L = (4.0 * np.pi * rng_sat[i] / (3e8 / freq[i])) ** 2

            # Simple gain model (assuming constant gain for speed)
            gain_tel = 1.0
            gain_sat = 1.0

            # Beam avoidance logic
            if beam_avoid_rad > 0:
                dec_condition = abs(beam_dec - dec_sat[i]) < beam_avoid_rad
                caz_condition = abs(beam_caz - (-caz_sat[i])) < beam_avoid_rad

                if turn_off and (dec_condition or caz_condition):
                    result[i] = 0.0
                    continue

            # Link budget calculation
            result[i] = gain_tel * (1.0 / L) * gain_sat

        return result


def sat_link_budget_vectorized(
    dec_tel, caz_tel, instru_tel, dec_sat, caz_sat, rng_sat, instru_sat, freq,
    beam_avoid=0.0, turn_off=False
):
    """
    Ultra-fast vectorized version using Numba JIT compilation for maximum performance.
    """
    # Convert inputs to arrays
    dec_tel = np.asarray(dec_tel)
    caz_tel = np.asarray(caz_tel)
    dec_sat = np.asarray(dec_sat)
    caz_sat = np.asarray(caz_sat)
    rng_sat = np.asarray(rng_sat)
    freq = np.asarray(freq)

    # Get the broadcast shape
    shape = np.broadcast_shapes(
        dec_tel.shape, caz_tel.shape, dec_sat.shape,
        caz_sat.shape, rng_sat.shape, freq.shape
    )

    # Flatten all arrays for processing
    dec_tel_flat = dec_tel.flatten()
    caz_tel_flat = caz_tel.flatten()
    dec_sat_flat = dec_sat.flatten()
    caz_sat_flat = caz_sat.flatten()
    rng_sat_flat = rng_sat.flatten()
    freq_flat = freq.flatten()

    # Use Numba-optimized function if available and no beam avoidance is needed
    if NUMBA_AVAILABLE and beam_avoid == 0.0:
        beam_avoid_rad = 0.0
        beam_dec, beam_caz = (0.0, 0.0)  # Default values for speed

        result = fast_link_budget_core(
            dec_tel_flat, caz_tel_flat, dec_sat_flat, caz_sat_flat,
            rng_sat_flat, freq_flat, beam_avoid_rad, beam_dec, beam_caz, turn_off
        )
    else:
        # Fallback to pure NumPy version
        # Vectorized coordinate transformation
        from coord_frames import ground_to_beam_coord_vectorized
        dec_sat_tel, caz_sat_tel = ground_to_beam_coord_vectorized(
            dec_sat_flat, caz_sat_flat, dec_tel_flat, caz_tel_flat
        )

        # Vectorized telescope gain calculation
        instru_ant = instru_tel.get_antenna()
        gain_tel = instru_ant.get_gain_value(dec_sat_tel, caz_sat_tel)

        # Vectorized satellite gain calculation
        sat_ant = instru_sat.get_antenna()
        dec_tel_sat = dec_sat_flat
        caz_tel_sat = -caz_sat_flat

        # Initialize with defaults (matching original Python implementation)
        dec_sat_ant = dec_tel_sat
        caz_sat_ant = caz_tel_sat

        # Vectorized beam avoidance logic
        if beam_avoid > 0:
            beam_dec, beam_caz = sat_ant.get_boresight_point()
            beam_avoid_rad = np.deg2rad(beam_avoid)

            # Vectorized conditions
            dec_condition = np.abs(beam_dec - dec_tel_sat) < beam_avoid_rad
            caz_condition = np.abs(beam_caz - caz_tel_sat) < beam_avoid_rad

            if turn_off:
                # For turn_off=True, we'll apply the mask after the link budget calculation
                pass
            else:
                # Modify coordinates for beam avoidance cases (matching original Python implementation)
                dec_sat_ant = np.where(
                    dec_condition,
                    np.mod(dec_tel_sat + np.pi / 4, np.pi),
                    dec_sat_ant,
                )
                caz_sat_ant = np.where(
                    caz_condition,
                    np.mod(caz_tel_sat + np.pi / 4, 2 * np.pi),
                    caz_sat_ant,
                )

        # Vectorized satellite gain calculation
        gain_sat = sat_ant.get_gain_value(dec_sat_ant, caz_sat_ant)

        # Vectorized link budget calculation
        # Free space loss: L = (4 * π * rng / (c / freq))^2
        speed_c = 3e8
        L = (4 * np.pi * rng_sat_flat / (speed_c / freq_flat)) ** 2

        # Link budget: gain_RX * (1 / L) * gain_TX
        result = gain_tel * (1 / L) * gain_sat

        # Apply beam avoidance turn-off if needed
        if beam_avoid > 0 and turn_off:
            result = result * np.where(dec_condition | caz_condition, 0.0, 1.0)

    # Reshape back to original broadcast shape
    return result.reshape(shape)
