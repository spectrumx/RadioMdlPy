import CSV

"""
assumes alphas and betas are degrees. Output is converted in radians.
"""
function map_sphere(pattern::AbstractArray{T},
    alphas::AbstractArray{T},
    betas::AbstractArray{T}) where T

    # form 2D matrix for interpolation argument
    # add first column as last column to loop azimuth coordinates
    gain_map = hcat(reshape(pattern, length(alphas), length(betas)),
                    pattern[1:length(alphas)])

    # generate sampling coordinates
    a = alphas .* rad
    b = [betas; T(360)] .* rad

    return gain_map, a, b
end


"""
yields the gain pattern of an antenna, in dB, given a radiated power pattern. It
is assumed that the radiated power includes the radiation efficiency.
the angles must be in degrees

"""
function radiated_power_to_gain(rad_pow::AbstractVector{T},
    alphas::AbstractVector{T},
    betas::AbstractVector{T};
    eta_rad::Real = 1.0) where T

    # map the radiated power for interpolation
    rad_pow_map, a, b = map_sphere(rad_pow, alphas, betas)

    # integrate over the sphere
    rad_pow_avg = trapz((a, b), rad_pow_map .* sin.(a)) / (4π)

    # rad_pow_avg
    println("rad_pow_avg: ", rad_pow_avg)

    # directivity
    dir = rad_pow ./ rad_pow_avg

    # gain
    return eta_rad .* dir
end



"""
"""
function interpolate_gain(gain::AbstractArray{T},
    alphas::AbstractArray{T},
    betas::AbstractArray{T}) where T

    # map the gain for interpolation
    gain_map, a, b = map_sphere(gain, alphas, betas)

    # gain function of angles in antenna coord. system
    return interpolate((a, b), gain_map, Gridded(Linear()))
end



"""
"""
function gain_to_effective_aperture(gain::Real,
    frequency::Real)

    wavelength = speed_c / frequency
    return gain * (wavelength^2/(4π))
end


