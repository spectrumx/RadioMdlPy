"""
Satellite modeling functions for radio astronomy
"""

import numpy as np
from datetime import datetime
from coord_frames import ground_to_beam_coord

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
