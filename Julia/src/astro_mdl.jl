
"""
    estim_temp(flux::T,
               effective_apperture::T) where T

estimates the temperature of a point-like source from its flux and the antenna effective
aperture. flux must be in Jansky

"""
function estim_temp(flux::T,
    effective_apperture::T) where T

    return flux*1e-26 / (2*k_boltz) * effective_apperture
end



"""
power in watt, bandwidth in hertz
"""
function power_to_temperature(power::T,
    bandwidth::T) where T

    return power / (k_boltz*bandwidth)
end



"""
"""
function temperature_to_power(temp::T,
    bandwidth::T) where T

    return k_boltz*bandwidth * temp
end



"""
in Jansky
"""
function temperature_to_flux(temp::T,
    effective_apperture::T) where T

    return (2*k_boltz) * temp / effective_apperture * 1e26
end



"""
    estim_casA_flux(center_freq::T) where T

estimates the flux of Cas A, given a frequency. Based on Baars et al. 1977

"""
function estim_casA_flux(center_freq::T) where T

    decay = 0.97 - 0.3*log10(center_freq*1e-9) # in %/year since 1980

    return 10^(5.745 - 0.770*log10(center_freq*1e-6))*(1 - decay*43/100) # in Jy
end



"""
"""
function estim_virgoA_flux(center_freq::T) where T
    return 10^(5.023 - 0.856*log10(center_freq*1e-6))
end



"""
create ITU recommended gain profile
"""
function antenna_mdl_ITU(gain_max::T,
    half_beamwidth::T,
    alphas::AbstractVector{T},
    betas::AbstractVector{T}) where T

    # gain profile container
    gain_profile = zeros(length(alphas))

    # select different parts of the gain profile
    parts = [0, half_beamwidth*sqrt(17/3), 10^((49-gain_max)/25), 48, 80, 120, 180]
    part1 = findall(i -> parts[1] <= i < parts[2], alphas)
    part2 = findall(i -> parts[2] <= i < parts[3], alphas)
    part3 = findall(i -> parts[3] <= i < parts[4], alphas)
    part4 = findall(i -> parts[4] <= i < parts[5], alphas)
    part5 = findall(i -> parts[5] <= i < parts[6], alphas)
    part6 = findall(i -> parts[6] <= i <= parts[7], alphas)

    # calculate gain profile
    gain_profile[part1] .= gain_max .- 3*(alphas[part1]./half_beamwidth).^2
    gain_profile[part2] .= gain_max - 20
    gain_profile[part3] .= 29 .- 25 .*log10.(alphas[part3])
    gain_profile[part4] .= -13
    gain_profile[part5] .= -8
    gain_profile[part6] .= -13

    # create gain dataframe
    gain_pat = DataFrame(alphas=zeros(length(alphas)*length(betas)),
                         betas=zeros(length(alphas)*length(betas)),
                         gains=zeros(length(alphas)*length(betas)))
    for b in eachindex(betas)
        gain_pat[((b-1)*length(alphas)+1):b*length(alphas), :alphas] .= alphas
        gain_pat[((b-1)*length(alphas)+1):b*length(alphas), :betas] .= betas[b]
        gain_pat[((b-1)*length(alphas)+1):b*length(alphas), :gains] .= 10 .^(gain_profile./10)
    end

    return gain_pat
end

