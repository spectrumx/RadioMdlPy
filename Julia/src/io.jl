
"""
    power_pattern_from_cut_file(file_path::String;
                                free_sp_imp::Real = 377,
                                verb::Bool = false)

yields the radiated power pattern, in dBW, of an antenna, times the radiation
efficiency from the `.cut` file containing co- and cross-polarization E-field.
Headers in the file are below a line starting with `Field`. It is composed of
the starting value of the declination angle α, the step and number of samples of
α and the value of the azimuthal angle β.

"""
function power_pattern_from_cut_file(file_path::String;
    free_sp_imp::Real = 377,
    verb::Bool = false)

    # parse file
    Es = readdlm(file_path)
    pattern = DataFrame(alpha=Float64[], beta=Float64[], power=Float64[])
    k = 1
    α_step = 0.0
    while k <= size(Es,1)
        header_line = k + findfirst(x -> x == "Field", Es[k:end,:])[1]
        header = Es[header_line,:]
        verb && println(header)
        α_start = header[1]
        α_step = header[2]
        nb_α = header[3]
        for t in 1:nb_α
            α = α_start + (t-1)*α_step
            θ = header[4]
            # power pattern, given in dBW, is the sum of the magnitude (squared
            # modulus) of the co- and cross-polarization complex electric field,
            # devided by twice the free-space impedance
            @inbounds u = sum(Es[header_line+t,1:4].^2)/(2*free_sp_imp)
            push!(pattern, [α, θ, u])
        end
        k = header_line+nb_α+1
    end
    decimal_places = max(0, -floor(Int, log10(abs(α_step - round(α_step)))))
    pattern[!,:alpha] .= round.(pattern[!,:alpha]; digits=decimal_places)

    # !!!!!!!!!!!!!!! THIS IS ONLY THE CASE WITH DANIEL'S FORMAT !!!!!!!!!!!!!!!

    @warn "This function assumes Daniel Sheen generated files"

    # check that α ∈ [-180,180[ and β ∈ [0, 180[
    subset!(pattern, :alpha => a -> -180.0 .<= a .< 180.0,
    :beta => b -> 0.0 .<= b .< 180.0)
    # pattern[1,:alpha] == -180.0 ? nothing : replace!(pattern, :alpha => a -> -a)

    # at this point, when the telescope is pointed at the horizon, β = 0 gives
    # an horizontal slice, with α > 0 oriented towards counter-clockwise azimuth
    # angles..

    # move the origin of β so that, when telescope points at the horizon, the
    # first slice (for the new β = 0) is vertical with α > 0 oriented towards
    # the ground.
    pattern[:,:beta] = mod.(pattern[!,:beta] .+ 90.0, 180.0)
    pattern[pattern.beta .>= 90.0,:alpha] .*= -1.0
    pattern[pattern.alpha .== 180.0,:alpha] .*= -1.0

    # change evolution domains so that α ∈ [0,180] and β ∈ [0, 360]
    pattern[pattern.alpha .<= 0,:beta] .+= 180
    rng_beta = pattern[pattern.alpha .== maximum(pattern.alpha),:beta]
    pattern[pattern.alpha .< 0,:alpha] .*= -1
    pattern[:,:alpha] = abs.(pattern[:,:alpha])
    append!(pattern, [(alpha = zero(eltype(pattern.alpha)), beta = i,
    power = pattern[pattern.alpha .== 0.0,:power][1])
    for i in rng_beta])
    append!(pattern, [(alpha = eltype(pattern.alpha)(180), beta = i,
    power = pattern[pattern.alpha .== 180.0,:power][1])
    for i in rng_beta])

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    sort!(pattern, [:beta, :alpha])

    return pattern
end

