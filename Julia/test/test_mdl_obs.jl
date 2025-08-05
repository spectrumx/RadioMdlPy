
using DataFrames
using Dates
using PyPlot
const plt = PyPlot

using Revise
using RadioMdl


### INSTRUMENT ###

## Antenna
# radiation efficiency of telescope antenna
eta_rad = .45

# valid frequency band of gain pattern model
freq_band = (10e9, 12e9) # in Hz

# load telescope antenna
file_pattern_path = "/home/samthe/Documents/Data/Radio/Westford_Antenna/\
                     Gain_pattern_Ku_band/single_cut_res.cut"
tel_ant = Antenna(file_pattern_path, eta_rad, freq_band;
                            power_tag=:power,
                            declination_tag=:alpha,
                            azimuth_tag=:beta)

#=
nb_curv = 5 # number of slices to plot
alphas, betas = get_def_angles(tel_ant) # angles where gain is defined by the file
step_beta_ind = div(length(betas), 2*nb_curv)
fig, axs = plt.subplots(subplot_kw=Dict("projection"=>"polar"))
for i in 1:step_beta_ind:div(length(betas), 2)
    alphas, gains = get_slice_gain(tel_ant, betas[i])
    axs.plot(deg2rad.(alphas), 10 .*log10.(gains), label="β = $(betas[i])deg")
end
axs.legend()
fig.tight_layout()
=#

# telescope antenna physical temperature
T_phy = 300.0 # in K


## Receiver
# frequency of observation
cent_freq = 10.820e9 # in Hz

# bandwidth of telescope receiver
bw = 1e3 # in Hz

# number of frequency channels to devide the bandwidth
freq_chan = 1

# telescope receiver temperature
T_RX(time::DateTime, freq::Real) = 80.0 # in K


## Telescope
# coordinates of telescope
coords = [42.6129479883915, -71.49379366344017, 86.7689687917009]

# create instrument
westford = Instrument(tel_ant, T_phy, cent_freq, bw, T_RX, freq_chan, coords)



### OBSERVATION PLAN ###

## Source trajectory over observation window
# observation window
start_window = "2025-03-07T14:00:00.000"
stop_window = "2025-03-07T17:00:00.000"

# source position over time
# to get the trajectory of the source over Westford, launch the Python script
# 'compute_obj_overflights_full_traj.py'#TODO: implement in Julia
file_traj_obj_path = "supp/traj_files/casA_trajectory_Westford_$(start_window)_\
                      $(stop_window).arrow"
traj_src = Trajectory(file_traj_obj_path;
                      time_tag = :time_stamps,
                      elevation_tag = :altitudes,
                      azimuth_tag = :azimuths,
                      distance_tag = :distances)


## Observation Parameters
# start-end of observation
dateformat = "yyyy-mm-dd\\THH:MM:SS.sss"
start_obs = DateTime("2025-03-07T14:36:30.000", dateformat)
stop_obs = DateTime("2025-03-07T15:06:30.000", dateformat)

# offset from source at the beginning of the observation
offset_angles = (-20, 0.) # (az,el) in degrees

# time of OFF-ON transition
time_off_src = start_obs
time_on_src = time_off_src + Minute(20)

# copy trajectory
traj_obj = Trajectory(copy(traj_src.traj))

# apply offset
traj_off_ind = findall(time_off_src .<= traj_obj.traj[!,:times] .<= time_on_src)
traj_obj.traj[traj_off_ind,:azimuths] .+= offset_angles[1]
traj_obj.traj[traj_off_ind,:elevations] .+= offset_angles[2]

# filter points below 5deg elevation
filt_el = (:elevations => e -> e .> 5.)

# create observation
observ = Observation(start_obs, stop_obs, traj_obj, westford;
                     filt_funcs = (filt_el,))

#=
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, polar=true)
src_traj = get_traj(traj_src)
obs_traj = get_traj(observ)
ax.plot(deg2rad.(src_traj[!,:azimuths]), [90 .- s4 for s4 in src_traj[!,:elevations]], 
        label="source")
ax.plot(deg2rad.(obs_traj[!,:azimuths]), [90 .- s4 for s4 in obs_traj[!,:elevations]], 
        label="pointing")
ax.set_yticks(0:10:90, string.(Vector(90:-10:0)))
ax.legend()
ax.set_theta_zero_location("N")
=#



### SKY COMPONENTS ###

## Source
#source flux
flux_src = estim_casA_flux(cent_freq) # in Jy

# source temperature in K #FIXME: should account for antenna position
function T_src(t::DateTime)
    if t <= time_on_src
        return 0.
    else
        return estim_temp(flux_src, observ)
    end
end


## RFI temperature
# ground temperature in K
T_gnd = 0 # no constant RFI

# various RFI
T_var = 0 # in K (no RFI)

# total RFI temperature
T_rfi = T_gnd + T_var


## Atmosphere temperature
# atmospheric temperature at zenith
T_atm_zenith = 150 # in K

# opacity of atmosphere at zenith
tau = .05 

# atmospheric temperature model
T_atm(dec::Real) = T_atm_zenith * (1 - exp(-tau/cos(dec))) # in K


## Background temperature
# CMB temperature
T_CMB = 2.73 # in K

# galaxy temperature
T_gal(freq::Real) = 1e-1 * (freq/1.41e9)^(-2.7) # in K

# background
T_bkg(freq::Real) = T_CMB + T_gal(freq)


## Total sky model
function sky_mdl(dec::T,
    caz::T,
    time::DateTime,
    freq::T) where {T<:Real}
    
    return T_src(time) + T_atm(dec) + T_rfi + T_bkg(freq)
end

#=
azimuth_grid = collect(0.:5.:360.)
elevation_grid = collect(0.:1.:90.)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, polar=true)
pc = pcolormesh(deg2rad.(azimuth_grid), 90 .- elevation_grid, sky_mdl.(deg2rad.(90 .-elevation_grid), 
                                                                       -deg2rad.(azimuth_grid)',
                                                                       start_obs, cent_freq), cmap="plasma")
cbar = plt.colorbar(pc)
cbar.set_label("Temperature [K]")
ax.set_yticks(0:10:90, string.(Vector(90:-10:0)))
ax.set_theta_zero_location("N")
fig.tight_layout()
=#



### SATELLITES CONSTELLATION ###

## Satellite Antenna
# radiation efficiency of telescope antenna
sat_eta_rad = .5#FIXME:check value

#
# maximum gain of satellite antenna
sat_gain_max = 39.3 # in dBi#FIXME:check value in dBi

# create ITU recommended gain profile
# satelitte boresight half beamwidth
half_beamwidth = 3. # in deg#FIXME:check value
# declination angles alpha
alphas = 0.:1.:180.
# azimuth angles beta
betas = 0.:10.:350.
# create gain dataframe
gain_pat = antenna_mdl_ITU(sat_gain_max, half_beamwidth, alphas, betas)

# create satellite antenna
sat_ant = Antenna(gain_pat, sat_eta_rad, freq_band)

#=
nb_curv = 5 # number of slices to plot
alphas, betas = get_def_angles(sat_ant) # angles where gain is defined by the file
step_beta_ind = div(length(betas), 2*nb_curv)
fig, axs = plt.subplots(subplot_kw=Dict("projection"=>"polar"))
for i in 1:step_beta_ind:div(length(betas), 2)
    alphas, gains = get_slice_gain(sat_ant, betas[i])
    axs.plot(deg2rad.(alphas), 10 .*log10.(gains), label="β = $(betas[i])deg")
end
axs.legend()
fig.tight_layout()
=#

# telescope antenna physical temperature
sat_T_phy = 0. # in K


## Satellites Transmitter
# frequency of transmition
sat_freq = cent_freq # in Hz

# satellite transmition bandwidth
sat_bw = 250e6 # in Hz

# satellite effective isotropically radiated power
transmit_pow = -15+10*log10(300) # in dBW#FIXME:check value
function transmit_temp(time::DateTime,
    freq::Real)
    
    return power_to_temperature(10^(transmit_pow/10), 1.)#sat_bw) # in K
end

# create transmitter instrument
sat_transmit = Instrument(sat_ant, sat_T_phy, sat_freq, sat_bw, transmit_temp)


## Constellation of satellites
# satellites trajectories during the observation
# filter the satellites
filt_name = (:sat => s -> .!contains.(s, "[DTC]"))
filt_el = (:elevations => e -> e .> 20)

# satellite link budget estimator
lnk_bdgt(args...) = sat_link_budget(args...; beam_avoid = 0., turn_off = false)

# To get Starlink sats trajectories over Westford launch the Python script
# 'compute_Starlink_overflights_full_traj.py'#TODO: implement in Julia
file_traj_sats_path = "supp/traj_files/Starlink_trajectory_Westford_$(start_window)_\
                      $(stop_window).arrow"
starlink_constellation = Constellation(file_traj_sats_path, observ, sat_transmit,
                                       lnk_bdgt; 
                                       name_tag = :sat,
                                       time_tag = :timestamp,
                                       elevation_tag = :elevations,
                                       azimuth_tag = :azimuths,
                                       distance_tag = :ranges_westford,
                                       filt_funcs = (filt_name, filt_el))

#=
list_sats = get_sats_name(starlink_constellation)
sel_sats = 1:length(list_sats)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, polar=true)
for s in list_sats[sel_sats]
    sat = get_sat_traj(starlink_constellation, s)
    ax.plot(deg2rad.(sat[:,:azimuths]), 90 .- sat[:,:elevations])
end
ax.set_yticks(0:10:90, string.(Vector(90:-10:0)))
ax.set_theta_zero_location("N")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for s in list_sats[sel_sats]
    sat = AM.get_sat_traj(starlink_constellation, s)
    ax.scatter(sat[:,:azimuths], sat[:,:elevations])
end
ax.set_xticks(0:40:360, string.(Vector(0:40:360)))
ax.set_yticks(0:10:90, string.(Vector(0:10:90)))
ax.set_xlabel("Azimuth [deg]")
ax.set_ylabel("Elevation [deg]")
=#



### TEMPERATURE MODEL DURIMG OBSERVATION ###
model_observed_temp!(observ, sky_mdl, starlink_constellation)

fig, axs = plt.subplots()
time_samples = get_time_stamps(observ)
plot_result = temperature_to_power.(get_result(observ), bw)[:,1,1]
axs.plot(time_samples, 10 .*log10.(plot_result), label="without beam avoidance")
axs.set_xlabel("time")
axs.set_ylabel("Power [dBW]")
axs.grid(true)
axs.legend()
fig.tight_layout()


