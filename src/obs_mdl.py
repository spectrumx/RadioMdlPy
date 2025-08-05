"""
Observation modeling functions for radio astronomy
"""

import numpy as np
from datetime import datetime
from typing import Callable, List, Tuple
from numba import njit

# Note: We'll import classes when needed to avoid circular imports

@njit
def compute_T_sys_numba(
    dec_tel, caz_tel, f_RX, max_gain, eta_rad, T_phy, T_RX_vals,
    sat_dec_arr, sat_caz_arr, sat_rng_arr, sat_temp_arr, lnk_bdgt_arr, T_sky_arr
):
    n_freq = f_RX.shape[0]
    T_sys_arr = np.empty(n_freq, dtype=np.float64)
    for f_idx in range(n_freq):
        T_sky = max_gain * T_sky_arr[f_idx]
        T_sat = 0.0
        n_sats = sat_dec_arr.shape[0]
        for s_idx in range(n_sats):
            link_val = lnk_bdgt_arr[s_idx, f_idx]
            sat_temp = sat_temp_arr[s_idx, f_idx]
            T_sat += link_val * sat_temp
        T_A = 1/(4 * np.pi) * (T_sat + T_sky)
        T_sys = T_A + (1 - eta_rad) * T_phy + T_RX_vals[f_idx]
        T_sys_arr[f_idx] = T_sys
    return T_sys_arr


def model_observed_temp(observation, sky_mdl: Callable, constellation=None) -> np.ndarray:
    """
    Hybrid Numba: Model the observed temperature during an observation, ported from Julia version.
    The outer loop is Python, the inner computation is Numba-accelerated.
    """
    from radio_types import Observation, Constellation, Instrument

    # Get observation parameters
    times = observation.get_time_stamps()
    traj = observation.get_traj()  # DataFrame with columns: times, azimuths, elevations,
    instrument = observation.get_instrument()

    # Instrument/antenna parameters (hoist all static values)
    bw_RX = instrument.get_bandwidth()
    freq_chan = instrument.get_nb_freq_chan()
    f_RX = instrument.get_center_freq_chans()
    T_RX_func = instrument.get_inst_signal()
    T_phy = instrument.get_phy_temp()
    antenna = instrument.get_antenna()
    eta_rad = antenna.get_rad_eff()
    max_gain = antenna.get_boresight_gain()

    # Pre-group pointings by time for fast lookup
    pointings_by_time = {t: df for t, df in traj.groupby('times')}

    # Prepare constellation info if present, and pre-group satellites by time
    cons_temps = []
    cons_ant = []
    cons_objs = []
    sats_by_time_list = []
    if constellation is not None:
        if not isinstance(constellation, list):
            constellation = [constellation]
        for con in constellation:
            cons_ant.append(con.get_antenna())
            sat_TX = con.get_transmitter()
            f_sat = sat_TX.get_center_freq()
            bw_sat = sat_TX.get_bandwidth()
            instru_freq = instrument.get_center_freq()
            # Check constellation is visible by receiver
            visible_band = [
                max(instru_freq - bw_RX/2, f_sat - bw_sat/2),
                min(instru_freq + bw_RX/2, f_sat + bw_sat/2)
            ]
            visible_bw = visible_band[1] - visible_band[0]
            if visible_bw < 0:
                raise ValueError(f"Constellation not seen by telescope receiver")
            cons_temps.append(sat_TX.get_inst_signal())
            cons_objs.append(con)
            # Pre-group satellites by time for this constellation
            sats_by_time_list.append({t: df for t, df in con.sats.groupby('times')})

    # Use the pre-allocated result array from the observation
    result = observation.get_result()

    # Main simulation loop (outer loop in Python)
    for t_idx, time in enumerate(times):
        # Get all pointings at this time (pre-grouped)
        pointings = pointings_by_time[time]
        for p_idx, row in enumerate(pointings.itertuples(index=False)):
            az = row.azimuths
            el = row.elevations
            dec_tel = np.radians(90 - np.array(el))
            caz_tel = np.radians(-np.array(az))

            # Prepare T_RX_vals and T_sky_arr for this pointing
            T_RX_vals = np.array([T_RX_func(time, f) for f in f_RX])
            T_sky_arr = np.array([sky_mdl(dec_tel, caz_tel, time, f) for f in f_RX])

            # Prepare satellite arrays if needed
            if cons_objs:
                for c_idx, con in enumerate(cons_objs):
                    instru_sat = con.get_transmitter()
                    lnk_bdgt = con.get_lnk_bdgt_mdl()
                    sat_TX_func = cons_temps[c_idx]
                    sats_t = sats_by_time_list[c_idx].get(time, None)
                    if sats_t is not None and len(sats_t) > 0:
                        sat_dec_arr = np.radians(90 - sats_t['elevations'].to_numpy())
                        sat_caz_arr = np.radians(-sats_t['azimuths'].to_numpy())
                        sat_rng_arr = sats_t['distances'].to_numpy()
                        n_sats = sat_dec_arr.shape[0]
                        # Prepare arrays for each frequency
                        lnk_bdgt_arr = np.empty((n_sats, len(f_RX)), dtype=np.float64)
                        sat_temp_arr = np.empty((n_sats, len(f_RX)), dtype=np.float64)
                        for s_idx in range(n_sats):
                            for f_idx, f_bin in enumerate(f_RX):
                                lnk_bdgt_arr[s_idx, f_idx] = lnk_bdgt(
                                    dec_tel, caz_tel, instrument,
                                    sat_dec_arr[s_idx], sat_caz_arr[s_idx], sat_rng_arr[s_idx],
                                    instru_sat, f_bin
                                )
                                sat_temp_arr[s_idx, f_idx] = sat_TX_func(time, f_bin)
                    else:
                        sat_dec_arr = np.empty(0)
                        sat_caz_arr = np.empty(0)
                        sat_rng_arr = np.empty(0)
                        lnk_bdgt_arr = np.empty((0, len(f_RX)))
                        sat_temp_arr = np.empty((0, len(f_RX)))
            else:
                sat_dec_arr = np.empty(0)
                sat_caz_arr = np.empty(0)
                sat_rng_arr = np.empty(0)
                lnk_bdgt_arr = np.empty((0, len(f_RX)))
                sat_temp_arr = np.empty((0, len(f_RX)))

            # Call the Numba-accelerated function
            T_sys_arr = compute_T_sys_numba(
                dec_tel, caz_tel, f_RX, max_gain, eta_rad, T_phy, T_RX_vals,
                sat_dec_arr, sat_caz_arr, sat_rng_arr, sat_temp_arr, lnk_bdgt_arr, T_sky_arr
            )
            result[t_idx, p_idx, :] = T_sys_arr

    return result
