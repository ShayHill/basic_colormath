"""Vectorized delta_e cielab 2000 distance calculation.

Vectorized versions of the delta_e cielab 2000 distance calculation, squared
Euclidean, and Euclidean distances.

Proximity and cross-proximity matrices for delta_e cielab 2000, squared Euclidean,
and Euclidean distances.

# Proximity Matrix

for [a, b, c] in colors:

f(a, a), f(a, b), f(a, c)
f(b, a), f(b, b), f(b, c)
f(c, a), f(c, b), f(c, c)

# Cross-Proximity Matrix

for [a, b, c, d] in colors_a:
for [x, y, z] in colors_b:

f(a, x), f(a, y), f(a, z)
f(b, x), f(b, y), f(b, z)
f(c, x), f(c, y), f(c, z)
f(d, x), f(d, y), f(d, z)

:author: Shay Hill
:created: 2024-08-22
"""

import math
from collections.abc import Callable
from typing import cast

import numpy as np
from numpy import typing as npt

from basic_colormath.vec_conversion import hexs_to_rgb

_FloatArray = npt.NDArray[np.float64]

_RGB_TO_XYZ = [
    [0.412424, 0.357579, 0.180464],
    [0.212656, 0.715158, 0.0721856],
    [0.0193324, 0.119193, 0.950444],
]

# constants to "linearize" rgb values
_XYZ_NORMALIZATION_THRESHOLD = 10.31475
_XYZ_SML_VAL_DENOMINATOR = 3294.6
_XYZ_LRG_VAL_OFFSET = 14.025
_XYZ_LRG_VAL_DENOMINATOR = 269.025
_XYZ_LRG_VAL_EXPONENT = 2.4


def _rgbs_to_xyz(rgbs: _FloatArray) -> _FloatArray:
    """RGB to XYZ conversion.

    :param rgbs: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :return: an array (..., 3) of X, Y, and Z values
        [0.0, 1.0286], [0.0, 1.0822], [0.0, 1.178]

    The standard rgb to xyz conversion scaled for [0, 255] color values.
    """
    linear_channels = np.copy(rgbs).astype(np.float64)
    lo_mask = linear_channels <= _XYZ_NORMALIZATION_THRESHOLD
    lo_idxs = np.where(lo_mask)
    hi_idxs = np.where(~lo_mask)
    linear_channels[lo_idxs] /= _XYZ_SML_VAL_DENOMINATOR
    linear_channels[hi_idxs] = (
        (linear_channels[hi_idxs] + _XYZ_LRG_VAL_OFFSET) / _XYZ_LRG_VAL_DENOMINATOR
    ) ** _XYZ_LRG_VAL_EXPONENT
    return cast(
        "_FloatArray",
        np.tensordot(  # pyright: ignore[reportUnknownMemberType]
            linear_channels, _RGB_TO_XYZ, axes=([-1], [1])
        ),
    )


_XYZ_TO_RGB = [
    [3.24070846, -1.53725917, -0.49857039],
    [-0.96925735, 1.87599516, 0.04155555],
    [0.05563507, -0.2039958, 1.05706957],
]


def _xyzs_to_rgb(xyzs: _FloatArray) -> _FloatArray:
    """
    XYZ to RGB conversion.
    :param xyzs: an array (..., 3) of X, Y, and Z values
    :return: array (..., 3) of red, green, blue values in [0, 255]
    """
    linear_channels = cast(
        "_FloatArray",
        np.tensordot(  # pyright: ignore[reportUnknownMemberType]
            xyzs, _XYZ_TO_RGB, axes=([-1], [1])
        ),
    )
    threshold_linear = _XYZ_NORMALIZATION_THRESHOLD / _XYZ_SML_VAL_DENOMINATOR
    lo_mask = linear_channels <= threshold_linear
    hi_mask = ~lo_mask

    rgb = np.copy(linear_channels)
    rgb[lo_mask] = rgb[lo_mask] * _XYZ_SML_VAL_DENOMINATOR
    rgb[hi_mask] = (
        _XYZ_LRG_VAL_DENOMINATOR * (rgb[hi_mask] ** (1 / _XYZ_LRG_VAL_EXPONENT))
        - _XYZ_LRG_VAL_OFFSET
    )
    return rgb


_CIE_E = 216 / 24389
_1_3RD = 1 / 3
_16_116THS = 16 / 116

# this will always be the illuminant when rgb is converted to xyz without an
# illuminant argument
_XYZ_ILLUM = (0.95047, 1.0, 1.08883)


def _xyzs_to_lab(xyzs: _FloatArray) -> _FloatArray:
    """Convert XYZ to Lab.

    :param xyzs: an array (...,3) of XYZ values
        [0.0, 1.0286], [0.0, 1.0822], [0.0, 1.178]
    :return: an array (...,3) of Lab values
        [0.0, 100], [-86.183, 98.235], [-107.865, 94.477]
    """
    scaled_xyz = xyzs / _XYZ_ILLUM
    lo_mask = scaled_xyz <= _CIE_E
    lo_idxs = np.where(lo_mask)
    hi_idxs = np.where(~lo_mask)
    scaled_xyz[lo_idxs] = (7.787 * scaled_xyz[lo_idxs]) + _16_116THS
    scaled_xyz[hi_idxs] **= _1_3RD

    lab = np.empty_like(scaled_xyz)
    lab[..., 0] = (116 * scaled_xyz[..., 1]) - 16.0
    lab[..., 1] = 500 * (scaled_xyz[..., 0] - scaled_xyz[..., 1])
    lab[..., 2] = 200 * (scaled_xyz[..., 1] - scaled_xyz[..., 2])
    return lab


def _labs_to_xyz(labs: _FloatArray) -> _FloatArray:
    """
    Lab to XYZ conversion.
    :param labs: an array (..., 3) of Lab values
    :return: array (..., 3) of XYZ values
    """
    lab_l = labs[..., 0]
    lab_a = labs[..., 1]
    lab_b = labs[..., 2]

    fy = (lab_l + 16.0) / 116
    fx = fy + (lab_a / 500)
    fz = fy - (lab_b / 200)

    # Inverse f function (piecewise)
    def inv_f(t: _FloatArray) -> _FloatArray:
        # shape will be broadcasted for any array t
        t_cubed = t**3
        mask = t_cubed > _CIE_E
        result = np.empty_like(t)
        result[mask] = t_cubed[mask]
        result[~mask] = (t[~mask] - _16_116THS) / 7.787
        return result

    x = inv_f(fx) * _XYZ_ILLUM[0]
    y = inv_f(fy) * _XYZ_ILLUM[1]
    z = inv_f(fz) * _XYZ_ILLUM[2]
    xyzs = np.stack([x, y, z], axis=-1)
    return xyzs


def rgbs_to_lab(rgbs: npt.ArrayLike) -> _FloatArray:
    """Convert RGB to Lab.

    :param rgbs: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :return: an array (...,3) of Lab values
        [0.0, 100], [-86.183, 98.235], [-107.865, 94.477]
    """
    xyzs = _rgbs_to_xyz(np.asarray(rgbs, dtype=np.float64))
    return _xyzs_to_lab(xyzs)


def labs_to_rgb(labs: npt.ArrayLike) -> _FloatArray:
    """Convert Lab to RGB.

    :param labs: an array (..., 3) of Lab values
    :return: an array (..., 3) of red, green, and blue values in [0, 255]
    """
    xyzs = _labs_to_xyz(np.asarray(labs, dtype=np.float64))
    return _xyzs_to_rgb(xyzs)


def hexs_to_lab(hexs: npt.ArrayLike) -> _FloatArray:
    """Convert an array of hex colors to Lab.

    :param hexs: an array (...) of hex colors
    :return: an array (...,3) of Lab values
        [0.0, 100], [-86.183, 98.235], [-107.865, 94.477]
    """
    rgbs = hexs_to_rgb(hexs)
    return rgbs_to_lab(rgbs)


# ===============================================================================
#   Euclidean Distance
# ===============================================================================


def get_sqeuclideans(rgbs_a: npt.ArrayLike, rgbs_b: npt.ArrayLike) -> _FloatArray:
    """Calculate the squared Euclidean distances between two rgb arrays.

    :param rgbs_a: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :param rgbs_a: an array the same shape as rgbs_a
    :return: an array (...) squared Euclidean distances between the two RGB colors.
    """
    rgbs_a = np.asarray(rgbs_a, dtype=np.float64)
    rgbs_b = np.asarray(rgbs_b, dtype=np.float64)
    return np.sum((rgbs_a - rgbs_b) ** 2, axis=-1)


def get_sqeuclideans_hex(hexs_a: npt.ArrayLike, hexs_b: npt.ArrayLike) -> _FloatArray:
    """Calculate the squared Euclidean distances between two hex arrays.

    :param hexs_a: an array (...) of hex colors
    :param hexs_b: an array the same shape as hexs_a
    :return: an array (...) squared Euclidean distances between the two hex colors.
    """
    rgbs_a = hexs_to_rgb(hexs_a)
    rgbs_b = hexs_to_rgb(hexs_b)
    return get_sqeuclideans(rgbs_a, rgbs_b)


def get_euclideans(rgbs_a: npt.ArrayLike, rgbs_b: npt.ArrayLike) -> _FloatArray:
    """Calculate the Euclidean distance between two RGB colors.

    :param rgb_a: The first RGB color.
    :param rgb_b: The second RGB color.
    :return: The Euclidean distance between the two RGB colors.
    """
    return get_sqeuclideans(rgbs_a, rgbs_b) ** 0.5


def get_euclideans_hex(hex_a: npt.ArrayLike, hex_b: npt.ArrayLike) -> _FloatArray:
    """Calculate the Euclidean distance between two HEX colors.

    :param hex_a: The first HEX color.
    :param hex_b: The second HEX color.
    :return: The Euclidean distance between the two HEX colors.
    """
    hexs_a = hexs_to_rgb(hex_a)
    hexs_b = hexs_to_rgb(hex_b)
    return get_euclideans(hexs_a, hexs_b)


# ===============================================================================
#   Delta E CIE2000
# ===============================================================================

_RAD_6 = math.radians(6)
_RAD_25 = math.radians(25)
_RAD_30 = math.radians(30)
_RAD_63 = math.radians(63)
_RAD_180 = math.radians(180)
_RAD_275 = math.radians(275)
_RAD_360 = math.radians(360)
_RAD_720 = math.radians(720)
_V25_E7 = 25**7


def _compute_C(lab: _FloatArray) -> _FloatArray:
    """Run subroutine of get_deltas_e_lab."""
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


def _compute_G(C1: _FloatArray, C2: _FloatArray) -> _FloatArray:
    """Run subroutine of get_deltas_e_lab."""
    avg_c_e7 = ((C1 + C2) / 2.0) ** 7
    return 0.5 * (1 - np.sqrt(avg_c_e7 / (avg_c_e7 + _V25_E7))) + 1


def _adjust_a_prime(lab: _FloatArray, G: _FloatArray) -> _FloatArray:
    """Run subroutine of get_deltas_e_lab."""
    return lab[..., 1] * G


def _compute_C_prime(a_p: _FloatArray, b_squared: _FloatArray) -> _FloatArray:
    """Run subroutine of get_deltas_e_lab."""
    return np.sqrt(a_p**2 + b_squared)


def _compute_H_prime(a_p: _FloatArray, b: _FloatArray) -> _FloatArray:
    """Run subroutine of get_deltas_e_lab."""
    h_p = np.arctan2(b, a_p)
    h_p[h_p < 0] += _RAD_360
    return h_p


def _compute_T(Hp: _FloatArray) -> _FloatArray:
    """Run subroutine of get_deltas_e_lab."""
    return (
        1
        - 0.17 * np.cos(Hp - _RAD_30)
        + 0.24 * np.cos(2 * Hp)
        + 0.32 * np.cos(3 * Hp + _RAD_6)
        - 0.2 * np.cos(4 * Hp - _RAD_63)
    )


def _adjust_delta_hp(h1p: _FloatArray, h2p: _FloatArray) -> _FloatArray:
    """Run subroutine of get_deltas_e_lab."""
    delta_hp = h2p - h1p
    mask_dhp = np.abs(delta_hp) > _RAD_180
    mask_increase = mask_dhp & (h2p > h1p)
    mask_decrease = mask_dhp & (h2p <= h1p)
    delta_hp[mask_increase] -= _RAD_360
    delta_hp[mask_decrease] += _RAD_360
    return delta_hp


def get_deltas_e_lab(lab_a: npt.ArrayLike, lab_b: npt.ArrayLike) -> _FloatArray:
    """Calculate the Delta E (CIE2000) of two Lab colors."""
    lab_a = np.asarray(lab_a, dtype=np.float64)
    lab_b = np.asarray(lab_b, dtype=np.float64)
    lab_a_bsq = (lab_a[..., 2] ** 2).astype(np.float64)
    lab_b_bsq = (lab_b[..., 2] ** 2).astype(np.float64)

    Lp = (lab_a[..., 0] + lab_b[..., 0]) / 2.0

    C1 = _compute_C(lab_a)
    C2 = _compute_C(lab_b)

    G = _compute_G(C1, C2)
    a1p = _adjust_a_prime(lab_a, G)
    a2p = _adjust_a_prime(lab_b, G)

    C1p = _compute_C_prime(a1p, lab_a_bsq)
    C2p = _compute_C_prime(a2p, lab_b_bsq)
    Cp = (C1p + C2p) / 2.0

    h1p = _compute_H_prime(a1p, lab_a[..., 2])
    h2p = _compute_H_prime(a2p, lab_b[..., 2])

    Hp = (h1p + h2p) / 2.0
    Hp[np.abs(h1p - h2p) > _RAD_180] += _RAD_180

    T = _compute_T(Hp)

    delta_hp = _adjust_delta_hp(h1p, h2p)
    delta_Lp = lab_b[..., 0] - lab_a[..., 0]
    delta_Cp = C2p - C1p
    delta_Hp = 2 * np.sqrt(C2p * C1p) * np.sin(delta_hp / 2.0)

    lp_minus_50_sq = (Lp - 50) ** 2
    S_L = 1 + (0.015 * lp_minus_50_sq) / np.sqrt(20 + lp_minus_50_sq)
    S_C = 1 + 0.045 * Cp
    S_H = 1 + 0.015 * Cp * T

    delta_ro = _RAD_30 * np.exp(-(((Hp - _RAD_275) / _RAD_25) ** 2))

    avg_cp_e7 = Cp**7
    R_C = np.sqrt(avg_cp_e7 / (avg_cp_e7 + _V25_E7))
    R_T = -2 * R_C * np.sin(2 * delta_ro)

    e_terms = [
        (delta_Lp / S_L) ** 2,
        (delta_Cp / S_C) ** 2,
        (delta_Hp / S_H) ** 2,
        R_T * (delta_Cp / S_C) * (delta_Hp / S_H),
    ]
    return sum(e_terms) ** 0.5


def get_deltas_e(rgbs_a: npt.ArrayLike, rgbs_b: npt.ArrayLike) -> _FloatArray:
    """Calculate the Delta E (CIE2000) of two arrays of RGB colors.

    :param rgbs_a: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :param rgbs_b: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :return: an array (...) of Delta E (CIE2000) distances between a and b.
        [0, 235.1568]
    """
    labs_a = rgbs_to_lab(rgbs_a)
    labs_b = rgbs_to_lab(rgbs_b)
    return get_deltas_e_lab(labs_a, labs_b)


def get_deltas_e_hex(hexs_a: npt.ArrayLike, hexs_b: npt.ArrayLike) -> _FloatArray:
    """Calculate the Delta E (CIE2000) of two arrays of HEX colors.

    :param hexs_a: an array (...) of hex colors, e.g. '#ff0000'
    :param hexs_b: an array (...) of hex colors
    :return: an array (...) of Delta E (CIE2000) distances between a and b.
        [0, 235.1568]
    """
    labs_a = hexs_to_lab(hexs_a)
    labs_b = hexs_to_lab(hexs_b)
    return get_deltas_e_lab(labs_a, labs_b)


# ===============================================================================
#   Proximity and Cross-Proximity Matrices
# ===============================================================================


def _build_proximity_matrix(
    colors: npt.ArrayLike, func: Callable[[npt.ArrayLike, npt.ArrayLike], _FloatArray]
) -> _FloatArray:
    """Build a proximity matrix from a list of colors.

    :param colors: an array (n, 3) of Lab or rgb colors
    :param func: a commutative function that calculates the proximity of two Lab
        colors. It is assumed that identical colors have a proximity of 0.
    :return: an array (n, n) of proximity values between every pair of Lab colors

    The proximity matrix is symmetric.
    """
    colors = np.asarray(colors)
    n = len(colors)
    rows = np.repeat(colors[:, np.newaxis, :], n, axis=1)
    cols = np.repeat(colors[np.newaxis, :, :], n, axis=0)
    proximity_matrix = np.zeros((n, n))
    ut = np.triu_indices(n, k=1)
    lt = (ut[1], ut[0])
    proximity_matrix[ut] = func(cols[ut], rows[ut])
    proximity_matrix[lt] = proximity_matrix[ut]
    return proximity_matrix


def _build_cross_proximity_matrix(
    colors_a: npt.ArrayLike,
    colors_b: npt.ArrayLike,
    func: Callable[[npt.ArrayLike, npt.ArrayLike], _FloatArray],
) -> _FloatArray:
    """Build a cross-proximity matrix from two lists of colors.

    :param colors_a: an array (n, 3) of Lab or rgb colors
    :param colors_b: an array (n, 3) of Lab or rgb colors
    :param func: a function that calculates the proximity of two Lab colors
    :return: an array (n, m) of proximity values between every pair (n_i, m_j) of Lab
        colors.
    """
    colors_a = np.asarray(colors_a)
    colors_b = np.asarray(colors_b)
    rows = np.repeat(colors_a[:, np.newaxis, :], len(colors_b), axis=1)
    cols = np.repeat(colors_b[np.newaxis, :, :], len(colors_a), axis=0)
    return func(rows, cols)


def get_delta_e_matrix_lab(
    labs_a: npt.ArrayLike, labs_b: npt.ArrayLike | None = None
) -> _FloatArray:
    """Build a Delta E (CIE2000) matrix from a list of Lab colors.

    :param labs_a: an array (n, 3) of Lab colors
    :param labs_b: an optional array (m, 3) of Lab colors
    :return: an array (n, n) or (n, m) of Delta E (CIE2000) distances between every
        pairs of Lab colors.

    If colors_b is None, the matrix is every distance between the colors in colors_a.

    If colors_b is not None, the matrix is the distance between every pair of colors
    (n_i, m_j) in colors_a and colors_b.
    """
    if labs_b is None:
        return _build_proximity_matrix(labs_a, get_deltas_e_lab)
    return _build_cross_proximity_matrix(labs_a, labs_b, get_deltas_e_lab)


def get_delta_e_matrix(
    rgbs_a: npt.ArrayLike, rgbs_b: npt.ArrayLike | None = None
) -> _FloatArray:
    """Build a Delta E (CIE2000) matrix from a list of RGB colors.

    :param rgbs_a: an array (n, 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :param rgbs_b: an optional array (m, 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :return: an array (n, n) or (n, m) of Delta E (CIE2000) distances between pairs.
    """
    labs_a = rgbs_to_lab(rgbs_a)
    labs_b = None if rgbs_b is None else rgbs_to_lab(rgbs_b)
    return get_delta_e_matrix_lab(labs_a, labs_b)


def get_delta_e_matrix_hex(
    hexs_a: npt.ArrayLike, hexs_b: npt.ArrayLike | None = None
) -> _FloatArray:
    """Build a Delta E (CIE2000) matrix from a list of HEX colors.

    :param hexs_a: an array (n,) of hex colors, e.g. '#ff0000'
    :param hexs_b: an optional array (m,) of hex colors
    :return: an array (n, n) or (n, m) of Delta E (CIE2000) distances between pairs.
    """
    labs_a = hexs_to_lab(hexs_a)
    labs_b = None if hexs_b is None else hexs_to_lab(hexs_b)
    return get_delta_e_matrix_lab(labs_a, labs_b)


def get_sqeuclidean_matrix(
    rgbs_a: npt.ArrayLike, rgbs_b: npt.ArrayLike | None = None
) -> _FloatArray:
    """Build a squared Euclidean distance matrix from a list of RGB colors.

    :param rgbs_a: an array (n, 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :param rgbs_b: an optional array (m, 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :return: an array (n, n) or (n, m) of squared Euclidean distances between pairs.
    """
    if rgbs_b is None:
        return _build_proximity_matrix(rgbs_a, get_sqeuclideans)
    return _build_cross_proximity_matrix(rgbs_a, rgbs_b, get_sqeuclideans)


def get_sqeuclidean_matrix_hex(
    hexs_a: npt.ArrayLike, hexs_b: npt.ArrayLike | None = None
) -> _FloatArray:
    """Build a squared Euclidean distance matrix from a list of HEX colors.

    :param hexs_a: an array (n,) of hex colors, e.g. '#ff0000'
    :param hexs_b: an optional array (m,) of hex colors
    :return: an array (n, n) or (n, m) of squared Euclidean distances between pairs.
    """
    rgbs_a = hexs_to_rgb(hexs_a)
    rgbs_b = None if hexs_b is None else hexs_to_rgb(hexs_b)
    return get_sqeuclidean_matrix(rgbs_a, rgbs_b)


def get_euclidean_matrix(
    rgbs_a: npt.ArrayLike, rgbs_b: npt.ArrayLike | None = None
) -> _FloatArray:
    """Build a Euclidean distance matrix from a list of RGB colors.

    :param rgbs_a: an array (n, 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :param rgbs_b: an optional array (m, 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :return: an array (n, n) or (n, m) of squared Euclidean distances between pairs.
    """
    if rgbs_b is None:
        return _build_proximity_matrix(rgbs_a, get_euclideans)
    return _build_cross_proximity_matrix(rgbs_a, rgbs_b, get_euclideans)


def get_euclidean_matrix_hex(
    hexs_a: npt.ArrayLike, hexs_b: npt.ArrayLike | None = None
) -> _FloatArray:
    """Build a Euclidean distance matrix from a list of HEX colors.

    :param hexs_a: an array (n,) of hex colors, e.g. '#ff0000'
    :param hexs_b: an optional array (m,) of hex colors
    :return: an array (n, n) or (n, m) of squared Euclidean distances between pairs.
    """
    rgbs_a = hexs_to_rgb(hexs_a)
    rgbs_b = None if hexs_b is None else hexs_to_rgb(hexs_b)
    return get_euclidean_matrix(rgbs_a, rgbs_b)
