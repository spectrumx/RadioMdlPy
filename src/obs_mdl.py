# """
# Observation modeling functions for radio astronomy
# """

import numpy as np
from typing import Callable
from coord_frames import ground_to_beam_coord_vectorized


def model_observed_temp(observation, sky_mdl: Callable, constellation=None, beam_avoidance=False) -> np.ndarray:
    """
    Optimized using advanced NumPy operations and vectorization.
    Vectorizes per-time across satellites and frequencies; only loops over
    pointings to evaluate telescope gain for S satellites, then reduces to F.

    Args:
        observation: Observation object containing trajectory and instrument data
        sky_mdl: Callable function for sky model
        constellation: Optional constellation object(s) for satellite interference
        beam_avoidance: If True, uses non-vectorized approach for beam avoidance calculations
    """  # noqa: E501

    # Get observation parameters and sort trajectory for correct reshaping
    times = observation.get_time_stamps()
    traj = observation.get_traj().sort_values(by='times').reset_index(drop=True)
    instrument = observation.get_instrument()
    n_times = len(times)
    n_pointings = len(traj) // n_times if n_times > 0 else 0

    # Instrument/antenna parameters
    f_RX_array = instrument.get_center_freq_chans()
    n_freq = len(f_RX_array)
    T_RX_func = instrument.get_inst_signal()
    T_phy = instrument.get_phy_temp()
    antenna = instrument.get_antenna()
    eta_rad = antenna.get_rad_eff()
    max_gain = antenna.get_boresight_gain()

    # Pre-shape all pointing data
    dec_tel_grid = np.radians(90 - traj['elevations'].values.reshape(n_times, n_pointings))
    caz_tel_grid = np.radians(-traj['azimuths'].values.reshape(n_times, n_pointings))

    # Pre-compute satellite data
    satellite_data = {}
    if constellation is not None:
        if not isinstance(constellation, list):
            constellation = [constellation]
        for c_idx, con in enumerate(constellation):
            sat_TX = con.get_transmitter()
            for time in times:
                sats_t = con.sats[con.sats['times'] == time]
                if len(sats_t) > 0:
                    sat_TX_func = sat_TX.get_inst_signal()
                    sat_temps_1f = np.array([sat_TX_func(time, f) for f in f_RX_array], dtype=np.float64)
                    satellite_data[(c_idx, time)] = {
                        'sat_dec': np.radians(90 - sats_t['elevations'].values),
                        'sat_caz': np.radians(-sats_t['azimuths'].values),
                        'sat_distances': sats_t['distances'].values,
                        'sat_temps': np.tile(sat_temps_1f, (len(sats_t), 1)),
                        'lnk_bdgt': con.get_lnk_bdgt_mdl(),
                        'instru_sat': sat_TX,
                    }

    result = observation.get_result()

    # Main simulation loop over time
    for t_idx, time in enumerate(times):
        dec_tel_t = dec_tel_grid[t_idx]
        caz_tel_t = caz_tel_grid[t_idx]

        # Vectorize sky model computation for all pointings and frequencies
        T_sky_arr = np.array([
            [sky_mdl(dec, caz, time, f) for f in f_RX_array]
            for dec, caz in zip(dec_tel_t, caz_tel_t)
        ], dtype=np.float64)
        T_RX_vals = np.array([T_RX_func(time, f) for f in f_RX_array], dtype=np.float64)

        T_sat_total = np.zeros((n_pointings, n_freq), dtype=np.float64)

        for c_idx, con in enumerate(constellation or []):
            sat_key = (c_idx, time)
            if sat_key in satellite_data:
                sd = satellite_data[sat_key]
                n_sats = len(sd['sat_dec'])

                if n_sats > 0:
                    if beam_avoidance:
                        # Use non-vectorized approach for beam avoidance
                        # Process each pointing and satellite combination individually
                        # to avoid broadcasting issues with coordinate transformations
                        for p_idx in range(n_pointings):
                            dec_tel = dec_tel_t[p_idx]
                            caz_tel = caz_tel_t[p_idx]

                            for s_idx in range(len(sd['sat_dec'])):
                                sat_dec = sd['sat_dec'][s_idx]
                                sat_caz = sd['sat_caz'][s_idx]
                                sat_dist = sd['sat_distances'][s_idx]

                                for f_idx in range(n_freq):
                                    freq = f_RX_array[f_idx]

                                    link_val = sd['lnk_bdgt'](
                                        dec_tel, caz_tel, instrument, sat_dec,
                                        sat_caz, sat_dist, sd['instru_sat'], freq
                                    )

                                    T_sat_total[p_idx, f_idx] += link_val * sd['sat_temps'][s_idx, f_idx]
                    else:
                        # Use optimized vectorized approach for normal case
                        # Precompute per-satellite, per-frequency kernel independent of pointing
                        # Kernel[S, F] = gain_sat[S] * FSPL_inv[S, F] * T_sat[S, F]
                        sat_ant = sd['instru_sat'].get_antenna()
                        dec_tel_sat = sd['sat_dec']
                        caz_tel_sat = -sd['sat_caz']
                        gain_sat = sat_ant.get_gain_values(dec_tel_sat, caz_tel_sat)  # (S,)

                        r_S = sd['sat_distances'].astype(np.float64)
                        f_F = f_RX_array.astype(np.float64)
                        c = 3.0e8
                        fspl_inv = ((c / f_F)[np.newaxis, :] / (4.0 * np.pi * r_S[:, np.newaxis])) ** 2  # (S, F)

                        kernel = gain_sat[:, np.newaxis] * fspl_inv * sd['sat_temps']  # (S, F)

                        # For each pointing, compute telescope gain over S sats and reduce
                        tel_ant = instrument.get_antenna()
                        for p_idx in range(n_pointings):
                            dec_tel = dec_tel_t[p_idx]
                            caz_tel = caz_tel_t[p_idx]
                            dec_sat_tel, caz_sat_tel = ground_to_beam_coord_vectorized(
                                sd['sat_dec'], sd['sat_caz'], dec_tel, caz_tel
                            )  # (S,)
                            gain_tel = tel_ant.get_gain_values(dec_sat_tel, caz_sat_tel)  # (S,)
                            T_sat_total[p_idx, :] += (gain_tel[:, np.newaxis] * kernel).sum(axis=0)

        # Final combination
        T_A = (1 / (4 * np.pi)) * (T_sat_total + max_gain * T_sky_arr)
        T_sys = T_A + (1 - eta_rad) * T_phy + T_RX_vals[np.newaxis, :]
        result[t_idx, :, :] = T_sys

    return result
