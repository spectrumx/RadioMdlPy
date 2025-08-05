"""
Coordinate frame transformation functions for radio modeling
"""

import numpy as np


def ground_to_beam_coord(dec_obj: float, caz_obj: float,
                        dec_tel: float, caz_tel: float) -> tuple:
    """
    Yields the coordinates (θ,ϕ) of an object in the telescope's dish coord. frame
    (p-g-b), given its declination and counter-azimuth angles (β,α) and the declination and
    counter-azimuth angles of the telescope (Ψ,Γ) in the telescope coord frame (N-W-Z).

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
    # Rotation matrix
    R = rot_mat(dec_tel, caz_tel)

    # Cartesian coord of object in (NEZ)
    obj_NEZ = spher_to_cart_coord(dec_obj, caz_obj)

    # Cartesian coord of object in (pgb)
    obj_pgb = R.T @ obj_NEZ

    # Spherical coord of object in (pgb)
    obj_pgb_sph = cart_to_sphe_coord(obj_pgb[0], obj_pgb[1], obj_pgb[2])

    return obj_pgb_sph[0], np.mod(np.nan_to_num(obj_pgb_sph[1], nan=0.0, copy=True), 2*np.pi)


def rot_mat(dec_tel: float, caz_tel: float) -> np.ndarray:
    """
    Yields the 3D rotation matrix given two spherical coordinates.
    """
    # Rotation of caz_tel around Z
    R_z_gamma = np.array([
        [np.cos(caz_tel), -np.sin(caz_tel), 0],
        [np.sin(caz_tel), np.cos(caz_tel), 0],
        [0, 0, 1]
    ])

    # Rotation of dec_tel around W
    R_w_psi = np.array([
        [np.cos(dec_tel), 0, np.sin(dec_tel)],
        [0, 1, 0],
        [-np.sin(dec_tel), 0, np.cos(dec_tel)]
    ])

    return R_z_gamma @ R_w_psi


def spher_to_cart_coord(theta: float, phi: float, r: float = 1.0) -> np.ndarray:
    """
    Convert spherical coordinates into cartesian coordinates.
    """
    return np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])


def cart_to_sphe_coord(x: float, y: float, z: float) -> np.ndarray:
    """
    Convert cartesian coordinates into spherical coordinates.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))
    return np.array([theta, phi, r])
