
"""
    ground_to_beam_coord(dec_obj::Real,
                         caz_obj::Real,
                         dec_tel::Real,
                         caz_tel::Real)

yields the coordinates (θ,ϕ) of an object in the telescope's dish coord. frame
(p-g-b), given its declination and counter-azimuth angles (β,α) and the declination and
coutner-azimuth angles of the telescope (Ψ,Γ) in the telescope coord frame (N-W-Z).

        Z                           b
        |    b                      |   obj
        | Ψ x     obj               | θ /'
        |--/'    x                  |--/ '
        | / '                       | /  '
        |/__'_______W               |/___'______g
       / `  '                      / `   '
      /---` '                     /---`  '
     /  Γ  `'                    /  ϕ  ` '
    N                           /       `'
                               p

"""
function ground_to_beam_coord(dec_obj::Real,
    caz_obj::Real,
    dec_tel::Real,
    caz_tel::Real)

    # rotation matrix
    R = rot_mat(dec_tel, caz_tel)
    
    # cartesian coord of object in (NEZ)
    obj_NEZ = spher_to_cart_coord(dec_obj, caz_obj)
    
    # cartesian coord of object in (pgb)
    obj_pgb = R' * obj_NEZ

    # spherical coord of object in (pgb)
    obj_pgb_sph = cart_to_sphe_coord(obj_pgb[1], obj_pgb[2], obj_pgb[3])
    
    return obj_pgb_sph[1], mod(isnan(obj_pgb_sph[2]) ? 0.0 : obj_pgb_sph[2], 2π)
end



"""
    rot_mat(dec_tel::Real,
            caz_tel::Real)

yields the 3D rotation matrix given two spherical coordinates.

"""
function rot_mat(dec_tel::Real,
    caz_tel::Real)

    # rotation of caz_tel around Z
    R_z_gamma = [ cos(caz_tel) -sin(caz_tel) 0
                  sin(caz_tel) cos(caz_tel)  0
                        0          0       1 ]
    # rotation of dec_tel around W
    R_w_psi = [ cos(dec_tel)  0  sin(dec_tel)
                      0       1       0      
                -sin(dec_tel) 0  cos(dec_tel) ]
                      
    return R_z_gamma * R_w_psi
end



"""
    spher_to_cart_coord(theta::Real, 
                        phi::Real,
                        r::Real = 1.0)

convert spherical coordinates into cartesian coordinates.

"""
function spher_to_cart_coord(theta::Real, 
    phi::Real,
    r::Real = 1.0)

    return [r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)]
end



"""
    cart_to_sphe_coord(x::Real, 
                       y::Real,
                       z::Real)

convert cartesian coordinates into spherical coordinates.

"""
function cart_to_sphe_coord(x::Real, 
    y::Real,
    z::Real)

    return [acos(z / sqrt(x^2+y^2+z^2)), sign(y) * acos(x / sqrt(x^2+y^2)), 
            sqrt(x^2+y^2+z^2)]
end
