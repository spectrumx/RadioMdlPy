module RadioMdl

export Antenna,
       antenna_mdl_ITU,
       Constellation,
       estim_casA_flux,
       estim_temp,
       estim_virgoA_flux,
       free_space_loss,
       get_center_freq_chans,
       get_def_angles,
       get_result,
       get_sats_name,
       get_sats_names_at_time,
       get_sat_traj,
       get_slice_gain,
       get_time_stamps,
       get_traj,
       Instrument,
       map_sphere,
       model_observed_temp!,
       Observation,
       power_to_temperature,
       sat_link_budget,
       simple_link_budget,
       Trajectory,
       temperature_to_flux,
       temperature_to_power

using Arrow
using DataFrames
using Dates
using DelimitedFiles
using Interpolations
using Trapz
using .Threads

""" Boltzman's constant in J/K"""
const k_boltz = 1.380649e-23

""" degree to radian conversion factor """
const rad = π/180

""" speed of light in m/s """
const speed_c = 3e8

include("astro_mdl.jl")
include("types.jl")
include("io.jl")
include("antenna_pattern.jl")
include("coord_frames.jl")
include("sat_mdl.jl")
include("obs_mdl.jl")

end # module RadioMdl
