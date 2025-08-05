import CSV

"""
beam_avoid in degrees
sky_mdl in K and depends on dec, caz, time and freq
"""
function model_observed_temp!(obs::Observation,
    sky_mdl::Function = (a,e,t -> 0.),
    constellation::Union{Nothing,Constellation,AbstractVector{Constellation}} = nothing)

    @assert !hasmethod(sky_mdl, (Real, Real, DateTime, Real))

    @warn "There may be an issue in terms of absolute temperature (bandwith scaling?)"

    # get time samples of observation
    times = get_time_stamps(obs)
    # pointing position of antenna during observation
    points = get_traj(obs)

    # get instrument used for observation
    instru = get_instrument(obs)
    # receiver bandwidth
    bw_RX = get_bandwidth(instru)
    # number of frequency channels
    freq_chan = get_nb_freq_chan(instru)
    # receiver frequencies (center freq of each freq channels)
    instru_freq = get_center_freq(instru)
    f_RX = get_center_freq_chans(instru)
    # receiver temperature
    T_RX  = get_inst_signal(instru)
    # antenna physical temnperature
    T_phy = get_phy_temp(instru)
    # antenna of instrument
    instru_ant = get_antenna(instru)
    # radiation efficiency of antenna
    eta_rad = get_rad_eff(instru_ant)
    # max gain of antenna
    max_gain = get_boresight_gain(instru_ant)

    if !isnothing(constellation)
        # temperature of satellites for each constellations
        cons_temps = Function[]
        # antenna of satellites for each constellations
        cons_ant = Antenna[]
        if typeof(constellation) <: Constellation
            constellation = [constellation]
        end
        for c in eachindex(constellation)
            con = constellation[c]
            # get antenna of satellites
            push!(cons_ant, get_antenna(con))
            # get transmitter of satellites
            sat_TX = get_transmitter(con)
            # satellite frequency
            f_sat = get_center_freq(sat_TX)
            # satellite bandwidth
            bw_sat = get_bandwidth(sat_TX)
            # check constellation is visible by receiver
            visible_band = [max(instru_freq - bw_RX/2, f_sat - bw_sat/2),
                            min(instru_freq + bw_RX/2, f_sat + bw_sat/2)] #FIXME:
            visible_bw = visible_band[2] - visible_band[1]

            visible_bw < 0 && error("Constellation #$(c) not seen by telescope receiver")

            # get satellite temperature (transmition)
            push!(cons_temps, get_inst_signal(sat_TX)) # in K
        end
    end

    # get result storage shaped as times x pointing positions
    result = get_result(obs)
    # simulate for each time sample
    for t in axes(result, 1)
        samp = times[t]
        # pointing position
        for p in axes(result, 2)
            # antenna pointing at time t in telescope frame
            # (declination, counter-azimuth)
            dec_tel = pi/2 - deg2rad(points[points.times .== samp,:elevations][1][p])
            caz_tel = -deg2rad(points[points.times .== samp,:azimuths][1][p])
            for f in axes(result, 3)
                # frequency channel
                f_bin = f_RX[f]
                # sky temperature
                #FIXME: source and sky needs to be separated to integrate sky
                #over full field of view (not a point-like source) Or sky_mdl
                #needs to be a custon structure that integrate gain over full
                #sky when pointing at position p
                T_sky = max_gain * sky_mdl(dec_tel, caz_tel, samp, f_bin)

                # satellite temperatures
                T_sat = 0.
                if !isnothing(constellation)
                    for c in eachindex(constellation)
                        # c-th constellation
                        con = constellation[c]
                        # satellites transmission in freq bin
                        sat_tmt = cons_temps[c](samp, f_bin)
                        # satellite transmitter
                        instru_sat = get_transmitter(con)
                        # link budget model
                        lnk_bdgt = get_lnk_bdgt_mdl(con)
                        # satellites antenna gain pattern
                        # sat_ant = cons_ant[c]

                        # satellite(s) up at time t
                        sats_t = subset(con.sats, :times => ts -> ts .== samp; view=true)
                        for s in axes(sats_t, 1)
                            # coordinates of sat at time t in telescope frame
                            # (declination, counter-azimuth, distance)
                            dec_sat = pi/2 - deg2rad(sats_t[s,:elevations])
                            caz_sat = -deg2rad(sats_t[s,:azimuths])
                            rng_sat = sats_t[s,:distances]

                            # link budget
                            link_val = lnk_bdgt(dec_tel, caz_tel, instru,
                                              dec_sat, caz_sat, rng_sat, instru_sat,
                                              f_bin)
                            T_sat += link_val * sat_tmt
                        end
                    end
                end

                # antenna tenperature
                T_A = 1/(4π) * (T_sat + T_sky)

                # system temperature
                T_sys = T_A + (1 - eta_rad)*T_phy + T_RX(samp, f_bin)

                result[t,p,f] = T_sys #* delta_freq #FIXME:

            end
        end
    end

    return result
end

