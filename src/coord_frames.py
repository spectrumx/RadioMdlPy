"""
Coordinate frame transformation functions for radio modeling
"""

import numpy as np


def ground_to_beam_coord(
        dec_obj: float, caz_obj: float,
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


def ground_to_beam_coord_vectorized(dec_obj, caz_obj, dec_tel, caz_tel):
    """
    Vectorized version of ground_to_beam_coord that can handle broadcasting.

    Args:
        dec_obj, caz_obj: Object coordinates (can be arrays)
        dec_tel, caz_tel: Telescope coordinates (can be arrays)

    Returns:
        tuple: (dec_obj_tel, caz_obj_tel) - Object coordinates in telescope frame
    """
    # Handle broadcasting by ensuring all inputs are arrays
    dec_obj = np.asarray(dec_obj)
    caz_obj = np.asarray(caz_obj)
    dec_tel = np.asarray(dec_tel)
    caz_tel = np.asarray(caz_tel)

    # Get the broadcast shape
    shape = np.broadcast_shapes(dec_obj.shape, caz_obj.shape, dec_tel.shape, caz_tel.shape)

    # Broadcast all arrays to the common shape before flattening
    dec_obj_bcast = np.broadcast_to(dec_obj, shape)
    caz_obj_bcast = np.broadcast_to(caz_obj, shape)
    dec_tel_bcast = np.broadcast_to(dec_tel, shape)
    caz_tel_bcast = np.broadcast_to(caz_tel, shape)

    # Flatten all arrays for processing
    dec_obj_flat = dec_obj_bcast.flatten()
    caz_obj_flat = caz_obj_bcast.flatten()
    dec_tel_flat = dec_tel_bcast.flatten()
    caz_tel_flat = caz_tel_bcast.flatten()

    # Initialize output arrays
    dec_obj_tel_flat = np.zeros_like(dec_obj_flat)
    caz_obj_tel_flat = np.zeros_like(caz_obj_flat)

    # Process each element
    for i in range(len(dec_obj_flat)):
        dec_obj_tel_flat[i], caz_obj_tel_flat[i] = ground_to_beam_coord(
            dec_obj_flat[i], caz_obj_flat[i],
            dec_tel_flat[i], caz_tel_flat[i]
        )

    # Reshape back to original broadcast shape
    dec_obj_tel = dec_obj_tel_flat.reshape(shape)
    caz_obj_tel = caz_obj_tel_flat.reshape(shape)

    return dec_obj_tel, caz_obj_tel


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


def rot_mat_vectorized(dec_tel, caz_tel):
    """
    Vectorized version of rot_mat that can handle broadcasting.

    Args:
        dec_tel, caz_tel: Telescope coordinates (can be arrays)

    Returns:
        np.ndarray: Rotation matrices with shape (..., 3, 3)
    """
    dec_tel = np.asarray(dec_tel)
    caz_tel = np.asarray(caz_tel)

    # Get the broadcast shape
    shape = np.broadcast_shapes(dec_tel.shape, caz_tel.shape)

    # Flatten for processing
    dec_tel_flat = dec_tel.flatten()
    caz_tel_flat = caz_tel.flatten()

    # Initialize output array
    result = np.zeros((len(dec_tel_flat), 3, 3))

    # Process each element
    for i in range(len(dec_tel_flat)):
        result[i] = rot_mat(dec_tel_flat[i], caz_tel_flat[i])

    # Reshape back to original broadcast shape + (3, 3)
    final_shape = shape + (3, 3)
    return result.reshape(final_shape)


def spher_to_cart_coord(theta: float, phi: float, r: float = 1.0) -> np.ndarray:
    """
    Convert spherical coordinates into cartesian coordinates.
    """
    return np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])


def spher_to_cart_coord_vectorized(theta, phi, r=1.0):
    """
    Vectorized version of spher_to_cart_coord that can handle broadcasting.

    Args:
        theta, phi: Spherical coordinates (can be arrays)
        r: Radius (can be array, defaults to 1.0)

    Returns:
        np.ndarray: Cartesian coordinates with shape (..., 3)
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    r = np.asarray(r)

    # Compute cartesian coordinates using broadcasting
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Stack along the last axis
    result = np.stack([x, y, z], axis=-1)

    return result


def cart_to_sphe_coord(x: float, y: float, z: float) -> np.ndarray:
    """
    Convert cartesian coordinates into spherical coordinates.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))
    return np.array([theta, phi, r])


def cart_to_sphe_coord_vectorized(x, y, z):
    """
    Vectorized version of cart_to_sphe_coord that can handle broadcasting.

    Args:
        x, y, z: Cartesian coordinates (can be arrays)

    Returns:
        np.ndarray: Spherical coordinates with shape (..., 3)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Compute spherical coordinates using broadcasting
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)

    # Handle the phi calculation carefully to avoid division by zero
    xy_sq = x**2 + y**2
    phi = np.where(xy_sq > 0,
                   np.sign(y) * np.arccos(x / np.sqrt(xy_sq)),
                   0.0)

    # Stack along the last axis
    result = np.stack([theta, phi, r], axis=-1)

    return result
