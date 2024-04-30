"""Get the delta_e cielab 2000 distance between two color tuples.

Three DeltaE CIE 2000 distance functions are provided:

get_delta_e  ->  get DeltaE CIE 2000 distance between two rgb tuples [0-255]
get_delta_e_hex  ->  get DeltaE CIE 2000 distance between two hex strings
get_delta_e_lab  ->  get DeltaE CIE 2000 distance between cached `_to_lab` calls

I've also provided squared Euclidean and Euclidean distance between rgb tuples or hex
strings.

get_sqeuclidean
get_sqeuclidean_hex
get_euclidean
get_euclidean_hex


`get_delta_e` is the function you'll usually want, but `get_delta_e_lab` will be a
bit faster if you want to cache the results of `rgb_to_lab` or `hex_to_lab` yourself.

Most of the math can be found here:
http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE2000.html

I've tried to preserve Bruce Lindbloom's naming conventions so you can follow along.

Intermediate color formats are 3-tuples with the following ranges:

    * RGB [0..255], [0..255], [0.255]
    * XYZ [0.0, 1.0286], [0.0, 1.0822], [0.0, 1.178]
    * Lab [0.0, 100], [-86.183, 98.235], [-107.865, 94.477]

:author: Shay Hill
:created: 2023-04-29
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

from basic_colormath.conversion import hex_to_rgb

if TYPE_CHECKING:
    from basic_colormath.type_hints import RGB, Hex, Lab

_Triple = Tuple[float, float, float]

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


def _rgb_to_xyz(rgb: RGB) -> _Triple:
    """RGB to XYZ conversion. Expects RGB values between 0 and 255.

    :param rgb: RGB values between 0 and 255 inclusive.
    :return: XYZ values between 0 and 1 inclusive.

    The standard rgb to xyz conversion scaled for [0, 255] color values.
    """
    linear_channels: list[float] = []
    for channel in rgb:
        if channel <= _XYZ_NORMALIZATION_THRESHOLD:
            linear_channel = channel / _XYZ_SML_VAL_DENOMINATOR
        else:
            linear_channel = channel + _XYZ_LRG_VAL_OFFSET
            linear_channel /= _XYZ_LRG_VAL_DENOMINATOR
            linear_channel = pow(linear_channel, _XYZ_LRG_VAL_EXPONENT)
        linear_channels.append(linear_channel)

    result_matrix = [
        sum(x * y for x, y in zip(row, linear_channels)) for row in _RGB_TO_XYZ
    ]
    rgb_r, rgb_g, rgb_b = (max(c, 0) for c in result_matrix)
    return rgb_r, rgb_g, rgb_b


_CIE_E = 216 / 24389
_1_3RD = 1 / 3
_16_116THS = 16 / 116

# this will always be the illuminant when rgb is converted to xyz from an without
# an illuminant argument
_XYZ_ILLUM = (0.95047, 1.0, 1.08883)


def _xyz_to_lab(xyz: _Triple) -> Lab:
    """Convert XYZ to Lab.

    :param xyz: XYZ color tuple
    :return: Lab color tuple
    """
    # this will always be the illuminant when rgb is converted to xyz from an without
    # an illuminant argument
    scaled_xyz = [c / y for c, y in zip(xyz, _XYZ_ILLUM)]
    for i, channel in enumerate(scaled_xyz):
        if channel > _CIE_E:
            scaled_xyz[i] = channel**_1_3RD
        else:
            scaled_xyz[i] = (7.787 * channel) + _16_116THS

    x, y, z = scaled_xyz
    lab_l = (116 * y) - 16.0
    lab_a = 500 * (x - y)
    lab_b = 200 * (y - z)
    return lab_l, lab_a, lab_b


_RAD_6 = math.radians(6)
_RAD_25 = math.radians(25)
_RAD_30 = math.radians(30)
_RAD_63 = math.radians(63)
_RAD_180 = math.radians(180)
_RAD_275 = math.radians(275)
_RAD_360 = math.radians(360)
_RAD_720 = math.radians(720)
_V25_E7 = 25**7


def rgb_to_lab(rgb: RGB) -> Lab:
    """Convert RGB to Lab.

    :param rgb: The RGB color to convert.
    :return: The Lab color.
    """
    xyz = _rgb_to_xyz(rgb)
    return _xyz_to_lab(xyz)


def hex_to_lab(hex_: Hex) -> Lab:
    """Convert hex color to Lab.

    :param hex_: The hex color to convert.
    :return: The Lab color.
    """
    rgb = hex_to_rgb(hex_)
    return rgb_to_lab(rgb)


def get_delta_e_lab(lab_a: Lab, lab_b: Lab) -> float:
    """Calculate the Delta E (CIE2000) of two Lab colors.

    :param lab_a: The first Lab color.
    :param lab_b: The second Lab color.
    :return: The Delta E (CIE2000) of the two colors.
    """
    lab_a_bsq = lab_a[2] ** 2
    lab_b_bsq = lab_b[2] ** 2

    Lp = (lab_a[0] + lab_b[0]) / 2.0

    C1 = (lab_a[1] ** 2 + lab_a_bsq) ** 0.5
    C2 = (lab_b[1] ** 2 + lab_b_bsq) ** 0.5
    avg_c_e7 = ((C1 + C2) / 2.0) ** 7
    G = 0.5 * (1 - (avg_c_e7 / (avg_c_e7 + _V25_E7)) ** 0.5) + 1

    a1p, a2p = (lab[1] * G for lab in (lab_a, lab_b))

    C1p = (a1p**2 + lab_a_bsq) ** 0.5
    C2p = (a2p**2 + lab_b_bsq) ** 0.5
    Cp = (C1p + C2p) / 2.0

    h1p = math.atan2(lab_a[2], a1p)
    h1p = h1p if h1p >= 0 else h1p + _RAD_360
    h2p = math.atan2(lab_b[2], a2p)
    h2p = h2p if h2p >= 0 else h2p + _RAD_360
    Hp = (h1p + h2p) / 2
    Hp = Hp if abs(h1p - h2p) <= _RAD_180 else Hp + _RAD_180

    T = (
        1
        - 0.17 * math.cos(Hp - _RAD_30)
        + 0.24 * math.cos(2 * Hp)
        + 0.32 * math.cos(3 * Hp + _RAD_6)
        - 0.2 * math.cos(4 * Hp - _RAD_63)
    )

    delta_hp = h2p - h1p
    if abs(delta_hp) > _RAD_180:
        if h2p > h1p:
            delta_hp -= _RAD_360
        else:
            delta_hp += _RAD_360

    delta_Lp = lab_b[0] - lab_a[0]
    delta_Cp = C2p - C1p
    delta_Hp = 2 * (C2p * C1p) ** 0.5 * math.sin(delta_hp / 2)

    lp_minus_50_sq = (Lp - 50) ** 2
    S_L = 1 + (0.015 * lp_minus_50_sq) / (20 + lp_minus_50_sq) ** 0.5
    S_C = 1 + 0.045 * Cp
    S_H = 1 + 0.015 * Cp * T

    delta_ro = _RAD_30 * math.exp(-(((Hp - _RAD_275) / _RAD_25) ** 2))

    avg_cp_e7 = Cp**7
    R_C = (avg_cp_e7 / (avg_cp_e7 + _V25_E7)) ** 0.5
    R_T = -2 * R_C * math.sin(2 * delta_ro)

    return (
        (delta_Lp / S_L) ** 2
        + (delta_Cp / S_C) ** 2
        + (delta_Hp / S_H) ** 2
        + R_T * delta_Cp / S_C * delta_Hp / S_H
    ) ** 0.5


def get_sqeuclidean(rgb_a: RGB, rgb_b: RGB) -> float:
    """Calculate the squared Euclidean distance between two RGB colors.

    :param rgb_a: The first RGB color.
    :param rgb_b: The second RGB color.
    :return: The squared Euclidean distance between the two RGB colors.
    """
    return sum((a - b) ** 2 for a, b in zip(rgb_a, rgb_b))


def get_sqeuclidean_hex(hex_a: Hex, hex_b: Hex) -> float:
    """Calculate the squared Euclidean distance between two HEX colors.

    :param hex_a: The first HEX color.
    :param hex_b: The second HEX color.
    :return: The squared Euclidean distance between the two HEX colors.
    """
    return get_sqeuclidean(hex_to_rgb(hex_a), hex_to_rgb(hex_b))


def get_euclidean(rgb_a: RGB, rgb_b: RGB) -> float:
    """Calculate the Euclidean distance between two RGB colors.

    :param rgb_a: The first RGB color.
    :param rgb_b: The second RGB color.
    :return: The Euclidean distance between the two RGB colors.
    """
    return get_sqeuclidean(rgb_a, rgb_b) ** 0.5


def get_euclidean_hex(hex_a: Hex, hex_b: Hex) -> float:
    """Calculate the Euclidean distance between two HEX colors.

    :param hex_a: The first HEX color.
    :param hex_b: The second HEX color.
    :return: The Euclidean distance between the two HEX colors.
    """
    return get_euclidean(hex_to_rgb(hex_a), hex_to_rgb(hex_b))


def get_delta_e(rgb_a: RGB, rgb_b: RGB) -> float:
    """Calculate the Delta E (CIE 2000) between two RGB colors.

    :param rgb_a: The first RGB color.
    :param rgb_b: The second RGB color.
    :return: The Delta E (CIE 2000) between the two RGB colors.
    """
    return get_delta_e_lab(rgb_to_lab(rgb_a), rgb_to_lab(rgb_b))


def get_delta_e_hex(hex_a: Hex, hex_b: Hex) -> float:
    """Calculate the Delta E (CIE 2000) between two hex colors.

    :param hex_a: The first hex color.
    :param hex_b: The second hex color.
    :return: The Delta E (CIE 2000) between the two hex colors.
    """
    return get_delta_e_lab(hex_to_lab(hex_a), hex_to_lab(hex_b))
