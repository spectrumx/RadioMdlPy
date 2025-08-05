import CSV
using DataFrames

"""
alphas are angles from z-axis towards x-y-plane, betas are angles from x-axis
towards y-axis, all in degree
the gain interpolator coords are in radians
"""
struct Antenna{T<:AbstractFloat,I<:Interpolations.GriddedInterpolation}
    gain_pat::AbstractDataFrame # gain pattern
    gain_func::I # gain interpolator
    rad_eff::T # radiation efficiency
    valid_freqs::Tuple{T,T} # min and max valid frequencies for the gain model

    function Antenna(gain_pat::AbstractDataFrame,
        gain_func::I,
        rad_eff::T,
        valid_freqs::Tuple{T,T} = (zero(T),zero(T))) where {T<:AbstractFloat,I<:Interpolations.GriddedInterpolation}

        for n in propertynames(gain_pat)
            @assert n in [:alphas, :betas, :gains]
        end
        @assert valid_freqs[1] < valid_freqs[2]

        return new{T,I}(gain_pat, gain_func, rad_eff, valid_freqs)
    end
end

function Antenna(gain_ant::AbstractDataFrame,
    rad_eff::T,
    valid_freqs = (zero(T),zero(T))) where T

    # create the gain interpolator
    alphas = subset(gain_ant, :betas => b -> b .== gain_ant[1,:betas];
                    view=true)[!,:alphas]
    betas = subset(gain_ant, :alphas => a -> a .== gain_ant[1,:alphas];
                   view=true)[!,:betas]
    gain_func = interpolate_gain(gain_ant[!,:gains], alphas, betas)

    return Antenna(gain_ant, gain_func, rad_eff, valid_freqs)
end

function Antenna(file_pattern_path::String,
    rad_eff::T,
    valid_freqs = (zero(T),zero(T));
    power_tag::Symbol = :gains,
    declination_tag::Symbol = :alphas,
    azimuth_tag::Symbol = :betas) where T

    # load the antenna power pattern
    @assert occursin(".cut", file_pattern_path) "the power pattern file must be a .cut file"
    gain_ant = power_pattern_from_cut_file(file_pattern_path)

    # rename angles columns
    rename!(gain_ant, declination_tag => :alphas)
    rename!(gain_ant, azimuth_tag => :betas)

    # convert into gain (do not use unique here for if different alphas for two betas)
    alphas = subset(gain_ant, :betas => b -> b .== gain_ant[1,:betas];
                    view=true)[!,:alphas]
    betas = subset(gain_ant, :alphas => a -> a .== gain_ant[1,:alphas];
                   view=true)[!,:betas]

    gain_ant[:,power_tag] = radiated_power_to_gain(gain_ant[!,power_tag], alphas, betas;
                                                   eta_rad=rad_eff)
    rename!(gain_ant, power_tag => :gains)

    return Antenna(gain_ant, rad_eff, valid_freqs)
end

get_gain_pattern(a::Antenna) = a.gain_pat
get_gain_value(a::Antenna, alpha::Real, beta::Real) = a.gain_func(alpha, beta)
function get_def_angles(a::Antenna)

    return unique(get_gain_pattern(a)[!,:alphas]), unique(get_gain_pattern(a)[!,:betas])
end
get_boresight_gain(a::Antenna) = maximum(get_gain_pattern(a)[:,:gains])
function get_boresight_point(a::Antenna)

    gain = get_gain_pattern(a)
    i = findmax(gain[:,:gains])[2]

    return gain[i,:alphas], gain[i,:betas]
end
function get_slice_gain(a::Antenna,
    beta::Real)

    gain_pat = get_gain_pattern(a)
    g_pos = subset(gain_pat, :betas => b -> b .== beta, view=true)
    g_neg = subset(gain_pat, :betas => b -> b .== beta + 180, view=true)

    alphas = [-reverse(g_neg[1:end-1,:alphas]); g_pos[2:end,:alphas]]
    gains = [reverse(g_neg[1:end-1,:gains]); g_pos[2:end,:gains]]

    return alphas, gains
end
get_rad_eff(a::Antenna) = a.rad_eff
get_valid_freqs(a::Antenna) = a.valid_freqs



"""
suppose the frame of antenna is oriented North-West-Up. Assumes the output of
signal_func is expressed in Kelvin (temperature).
signal_funcs should depend on the time and the frequency.
freq_chan is number of frequency channels for spectrograph model instead of full
bw integration of power.
"""
struct Instrument{T<:AbstractFloat}
    antenna::Antenna # antenna
    phy_temp::T # antenna physical temperature
    cent_freq::T # center frequency
    bw::T # bandwidth
    signal_func::Function # signal from instrument (noise, transmission, etc.)
    freq_chan::Int # number of frequency channels
    coords::AbstractVector{T} # coordinates

    function Instrument(ant::Antenna,
        phy_temp::T,
        cent_freq::T,
        bw::T,
        signal_func::Function,
        freq_chan::Int = 1,
        coords::AbstractVector{T} = T[]) where {T<:AbstractFloat}

        # check antenna model is suited for center frequency of instrument
        ant_fmin, ant_fmax = get_valid_freqs(ant)
        @assert ant_fmin <= cent_freq - bw/2 && cent_freq + bw/2 <= ant_fmax
        # check signal_func signature is correct
        @assert hasmethod(signal_func, (DateTime, T))

        return new{T}(ant, phy_temp, cent_freq, bw, signal_func, freq_chan, coords)
    end
end

function Instrument(ant::Antenna,
    phy_temp::T,
    center_freq::T,
    bandwidth::T,
    signal::T,
    freq_chan::Int = 1,
    coords::AbstractVector{T} = T[]) where {T<:AbstractFloat}

    # transform scalar in function
    signal_func(args...) = signal
    return Instrument(ant, phy_temp, center_freq, bandwidth, signal_func, freq_chan, coords)
end

get_coords(i::Instrument) = i.coords
get_antenna(i::Instrument) = i.antenna
get_phy_temp(i::Instrument) = i.phy_temp
get_center_freq(i::Instrument) = i.cent_freq
get_bandwidth(i::Instrument) = i.bw
get_nb_freq_chan(i::Instrument) = i.freq_chan
function get_center_freq_chans(i::Instrument)

    freq_chan = get_nb_freq_chan(i)
    bw_RX = get_bandwidth(i)
    delta_freq = bw_RX/freq_chan
    rng_freq = range(-bw_RX/2 + delta_freq/2, bw_RX/2 - delta_freq/2, length=freq_chan)
    return get_center_freq(i) .+ rng_freq
end
get_inst_signal(i::Instrument) = i.signal_func



"""
traj is a DataFrame where each row is indexed by a datatime and has an elevation
and azimuth angle(s). It is possible to have vectors of els and azs for a same
time (e.g. for sky mapping). All measurements are assumed to be given in SI
units (in degrees for angles, meters for distances).
"""
struct Trajectory
    traj::AbstractDataFrame # azimuth, elevation and distance info for each sampled time

    function Trajectory(traj::AbstractDataFrame)

        for n in propertynames(traj)
            @assert n in [:times, :azimuths, :elevations, :distances]
        end
        @assert length(unique(traj.times)) == length(traj.times)
        @assert typeof(traj.times[1]) == DateTime
        @assert minimum(length.(traj.azimuths) .== length(traj.azimuths[1]))
        @assert minimum(length.(traj.elevations) .== length(traj.elevations[1]))
        @assert minimum(length.(traj.distances) .== length(traj.distances[1]))
        @assert length(traj.azimuths[1]) == length(traj.elevations[1])
        @assert length(traj.azimuths[1]) == length(traj.distances[1])

        sort!(traj, :times)

        return new(traj)
    end
end

function Trajectory(file_path::String;
    time_tag::Symbol = :times,
    elevation_tag::Symbol = :altitudes,
    azimuth_tag::Symbol = :azimuths,
    distance_tag::Symbol = :distances)

    # load the trajectory as a DataFrame
    #FIXME: read also csv files without dates for real antenna positions
    if occursin(".arrow", file_path)
        traj = DataFrame(Arrow.Table(file_path))
    elseif occursin(".csv", file_path)
        traj = DataFrame(CSV.File(file_path))
    else
        error("the trajectory points are not in Arrow or CSV format")
    end

    # rename columns
    rename!(traj, time_tag => :times)
    rename!(traj, azimuth_tag => :azimuths)
    rename!(traj, elevation_tag => :elevations)
    rename!(traj, distance_tag => :distances)

    # convert time stamps into DateTime type
    @. traj[!,:times] = Dates.DateTime(traj[!,:times])

    return Trajectory(traj[!, [:times, :azimuths, :elevations, :distances]])
end

get_traj(t::Trajectory) = t.traj
function get_traj(t::Trajectory, t0::DateTime, t1::DateTime;
    skipmissing::Bool = true,
    view::Bool = true)

    return subset(get_traj(t), :times => t -> t0 .<= t .<= t1;
                  skipmissing=skipmissing, view=view)
end
get_time_bounds(t::Trajectory) = (t.traj[1,:times], t.traj[end,:times])
get_time_stamps(t::Trajectory) = t.traj[:,:times]
get_azimuths(t::Trajectory) = t.traj[:,:azimuths]
get_elevations(t::Trajectory) = t.traj[:,:elevations]
get_distances(t::Trajectory) = t.traj[:,:distances]



"""
pts can contain different positions for a same time, e.g. for a sky map.
"""
struct Observation{T<:AbstractFloat}
    pts::Trajectory # trajectory of the observation
    inst::Instrument{T} # instrument used for observation
    result::AbstractArray{T} # store the results of the modeling of the observation#FIXME:DataFrame?

    function Observation(pts::Trajectory,
        inst::Instrument{T},
        result::AbstractArray{T}) where T

        @assert length(get_time_stamps(pts)) == size(result, 1)
        @assert length(get_azimuths(pts)[1]) == size(result, 2)
        @assert get_nb_freq_chan(inst) == size(result, 3)

        return new{T}(pts, inst, result)
    end
end

function Observation(start_date::DateTime,
    stop_date::DateTime,
    trajectory::Trajectory,
    instrument::Instrument{T};
    filt_funcs::NTuple{N,Pair} = ()) where {N,T}

    # filter date and other from trajectory
    traj = get_traj(trajectory, start_date, stop_date; view=false)
    for filt in filt_funcs
        traj = subset(traj, filt; skipmissing=true, view=true)
    end
    isempty(traj) && error("No pointing positions found for the given time window and \
                            custom filters.")
    sort!(traj, :times)

    pts = Trajectory(traj)

    # create result storage
    len_time = length(get_time_stamps(pts))
    len_pos = length(get_azimuths(pts)[1])
    len_freq = get_nb_freq_chan(instrument)
    result = fill!(zeros(T, len_time, len_pos, len_freq), NaN)

    return Observation(pts, instrument, result)
end

get_traj(obs::Observation) = get_traj(obs.pts)
get_time_bounds(obs::Observation) = get_time_bounds(obs.pts)
get_time_stamps(obs::Observation; kwds...) = get_time_stamps(obs.pts; kwds...)
get_azimuths(obs::Observation) = get_azimuths(obs.pts)
get_elevations(obs::Observation) = get_elevations(obs.pts)
get_distances(obs::Observation) = get_distances(obs.pts)
get_instrument(obs::Observation) = obs.inst
get_result(obs::Observation) = obs.result

function estim_temp(flux::Real,
    obs::Observation)

    instru = get_instrument(obs)
    frequency = get_center_freq(instru)
    ant = get_antenna(instru)
    max_gain = get_boresight_gain(ant)
    A_eff_max = gain_to_effective_aperture(max_gain, frequency)

    return estim_temp(flux, A_eff_max)
end



"""
assumes the positions of sats are time-synced with the time samples of
observation.
Suppose the frame of satellite antenna is oriented North-East-Nadir. The antenna
pointing can be any direction from Nadir, defined in the map gain.
"""
struct Constellation{T<:AbstractFloat}
    sats::AbstractDataFrame
    tmt::Instrument{T}
    lnk_bdgt_mdl::Function

    function Constellation(sats::AbstractDataFrame,
        tmt::Instrument{T},
        lnk_bdgt_mdl::Function) where T

        # check lnk_bdgt_mdl signature is correct
        @assert hasmethod(lnk_bdgt_mdl, (T, T, Instrument{T}, T, T, T, Instrument{T}, T))

        return new{T}(sats, tmt, lnk_bdgt_mdl)
    end
end

function Constellation(sats::AbstractDataFrame,
    observation::Observation,
    sat_tmt::Instrument{T},
    lnk_bdgt_mdl::Function = sat_link_budget;
    filt_funcs::NTuple{N,Pair} = ()) where {N,T}

    # observation window
    start_date , stop_date = get_time_bounds(observation)

    # apply the custom filters
    sats = subset(sats, :times => t -> start_date .<= t .<= stop_date;
                  skipmissing=true, view=true)
    for filt in filt_funcs
        sats = subset(sats, filt; skipmissing=true, view=true)
    end

    # check = minimum(get_time_stamps(observation) .== unique(sats[!,:times]))
    #=@assert check=# @warn "Observation time stamps and Constellation time stamps needs\
                             to be aligned."

    return Constellation(sats, sat_tmt, lnk_bdgt_mdl)
end

function Constellation(file_path::String,
    observation::Observation,
    sat_tmt::Instrument{T},
    lnk_bdgt_mdl::Function = sat_link_budget;
    name_tag::Symbol = :sat,
    time_tag::Symbol = :time_stamps,
    elevation_tag::Symbol = :altitudes,
    azimuth_tag::Symbol = :azimuths,
    distance_tag::Symbol = :distances,
    filt_funcs::NTuple{N,Pair} = ()) where {N,T}

    # load the trajectory as a DataFrame
    sats = DataFrame(Arrow.Table(file_path))

    # rename columns
    rename!(sats, time_tag => :times)
    rename!(sats, name_tag => :sat)
    rename!(sats, azimuth_tag => :azimuths)
    rename!(sats, elevation_tag => :elevations)
    rename!(sats, distance_tag => :distances)

    @. sats[!,:times] = Dates.DateTime(sats[!,:times])

    sort!(sats, :times)

    return Constellation(sats, observation, sat_tmt, lnk_bdgt_mdl;
                         filt_funcs=filt_funcs)
end

get_antenna(c::Constellation) = get_antenna(c.tmt)
get_transmitter(c::Constellation) = c.tmt
get_sats_name(c::Constellation) = unique(c.sats[!,:sat])
get_lnk_bdgt_mdl(c::Constellation) = c.lnk_bdgt_mdl

function get_sat_traj(c::Constellation,
    s::String)

    return subset(c.sats, :sat => n -> n .== s; view=true)
end

function get_sats_names_at_time(c::Constellation,
    t::DateTime)

    return subset(c.sats, :times => ts -> ts .== t; view=true)[!,:sat]
end
