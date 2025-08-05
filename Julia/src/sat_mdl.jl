
"""
"""
function free_space_loss(rng::T,
    freq::T) where T

    return ( 4*pi*rng / (speed_c / freq) )^2
end



"""
"""
function simple_link_budget(gain_RX::T,
    gain_TX::T,
    rng::T,
    freq::T,) where T

    L = free_space_loss(rng, freq)

    return gain_RX * 1/L * gain_TX
end



"""
"""
function sat_link_budget(dec_tel::T,
    caz_tel::T,
    instru_tel::Instrument{T},
    dec_sat::T,
    caz_sat::T,
    rng_sat::T,
    instru_sat::Instrument{T},
    freq::T;
    beam_avoid::T = 0.0,
    turn_off::Bool = false) where T

    # coordinate of sat in antenna frame
    dec_sat_tel, caz_sat_tel = ground_to_beam_coord(dec_sat, caz_sat,
                                                    dec_tel, caz_tel)

    # telescope gain
    instru_ant = get_antenna(instru_tel)
    gain_tel = get_gain_value(instru_ant, dec_sat_tel, caz_sat_tel)

    sat_ant = get_antenna(instru_sat)
    # coordinate of telescope at time t in satellite frame
    # (declination, counter-azimuth)
    dec_tel_sat = dec_sat
    caz_tel_sat = -caz_sat
    # FIXED:initialize with defaults
    dec_sat_ant, caz_sat_ant = dec_tel_sat, caz_tel_sat

    if beam_avoid > 0
        # get boresight pointing of satellite antenna
        beam_dec, beam_caz = deg2rad.(get_boresight_point(sat_ant))
        if abs.(beam_dec - dec_tel_sat) < deg2rad(beam_avoid)
            if turn_off
                return zero(T)
            else
                dec_tel_sat = mod(dec_tel_sat + pi/4, pi)
            end
        elseif abs.(beam_caz - caz_tel_sat) < deg2rad(beam_avoid)
            if turn_off
                return zero(T)
            else
                caz_sat_ant = mod(caz_sat_ant + pi/4, 2*pi)
            end
        end
    end
    # satellite gain
    gain_sat = get_gain_value(sat_ant, dec_tel_sat, caz_sat_ant)

    # link budget
    return simple_link_budget(gain_tel, gain_sat, rng_sat, freq)
end