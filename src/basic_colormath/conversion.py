"""Convert between (upscaled) RGB, HSL, HSV, and HEX.

These are the straightforward conversions. This library also converts to Lab
(niavely, there are parameters that are not taken into account), but only for finding
CIEDE2000 color differences.

That conversion is in the distance.py module.

:author: Shay Hill
:created: 2023-04-30
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_MAX_8BIT = 255

if TYPE_CHECKING:
    from basic_colormath.type_hints import HSL, HSV, RGB, Hex


def _get_hue_from_rgb(rgb: RGB, min_: float, max_: float) -> float:
    """Get the hue value in degrees from an rgb tuple.

    :param rgb: A tuple of red, green, and blue values, ([0, 255], [0, 255], [0, 255]).
    :param min_: The minimum value in the rgb tuple.
    :param max_: The maximum value in the rgb tuple.
    :return: The hue value in degrees, [0, 360).
    """
    red, grn, blu = rgb
    if max_ == min_:
        return 0
    if max_ == red:
        return (60 * ((grn - blu) / (max_ - min_)) + 360) % 360
    if max_ == grn:
        return 60 * ((blu - red) / (max_ - min_)) + 120
    return 60 * ((red - grn) / (max_ - min_)) + 240


def _sort_channels_given_hue(hue: float, min_mid_max: RGB) -> RGB:
    """Sort the channels of an rgb tuple given a hue value.

    :param hue: The hue value in degrees, [0, 360).
    :param min_mid_max: A tuple of the minimum, middle, and maximum values in the rgb
    :return: an arrangement of min_mid_max that corresponds to the hue.

    Regardless of saturation, lightness, value, etc., the order of minimum, middle,
    and maximum values in an rgb tuple will be consistent. For instance, given a hue
    of 0, the red channel will always be the highest value.
    """
    hue %= 360
    sextant2order = {
        0: (2, 1, 0),
        1: (1, 2, 0),
        2: (0, 2, 1),
        3: (0, 1, 2),
        4: (1, 0, 2),
        5: (2, 0, 1),
    }
    red, grn, blu = (min_mid_max[i] for i in sextant2order[int(hue // 60)])
    return red, grn, blu


def rgb_to_hsv(rgb: RGB) -> HSV:
    """Convert from rgb to hsv.

    :param rgb: a tuple of red, green, and blue values ([0, 255], [0, 255], [0, 255]).
    :return: a tuple of hue, saturation, and value ([0, 360), [0, 100], [0, 100]).
    """
    min_ = min(rgb)
    max_ = max(rgb)
    hue = _get_hue_from_rgb(rgb, min_, max_)
    sat = 0 if max_ == 0 else (1 - (min_ / max_)) * 100
    val = max_ / 2.55
    return hue, sat, val


def hsv_to_rgb(hsv: HSV) -> RGB:
    """Convert from hsv to rgb.

    :param hsv: a tuple of hue, saturation, and value ([0, 360], [0, 100], [0, 100]).
    :return: a tuple of red, green, and blue values ([0, 255], [0, 255], [0, 255]).
    """
    hue, sat, val = hsv
    scaled_val = val * 2.55
    if sat == 0 or val == 0:
        return scaled_val, scaled_val, scaled_val
    max_ = scaled_val
    min_ = max_ - (sat / 100 * max_)
    hue_mid_ratio = 1 - abs((hue / 60) % 2 - 1)
    mid = hue_mid_ratio * (max_ - min_) + min_
    return _sort_channels_given_hue(hue, (min_, mid, max_))


def rgb_to_hsl(rgb: RGB) -> HSL:
    """Convert rgb to hsl.

    :param rgb: a tuple of red, green, and blue values ([0, 255], [0, 255], [0, 255]).
    :return: a tuple of hue, saturation, and lightness ([0, 360), [0, 100], [0, 100]).
    """
    var_min = min(rgb)
    var_max = max(rgb)
    hue = _get_hue_from_rgb(rgb, var_min, var_max)
    lig = var_max + var_min

    if var_max == var_min:
        sat = 0.0
    elif lig <= _MAX_8BIT:
        sat = (var_max - var_min) / (0.01 * lig)
    else:
        sat = (var_max - var_min) / (5.1 - (0.01 * lig))
    return hue, sat, lig / 5.1


def hsl_to_rgb(hsl: HSL) -> RGB:
    """Convert hsl to rgb.

    :param hsl: a tuple of hue, sat, and lightness ([0, 360], [0, 100], [0, 100]).
    :return: a tuple of red, green, and blue values ([0, 255], [0, 255], [0, 255]).
    """
    hue, sat, lig = hsl
    max_p = (100 - abs(2 * lig - 100)) * sat * 0.0255
    mid_p = max_p * (1 - abs((hue / 60) % 2 - 1))
    min_ = (lig - max_p / 5.1) * 2.55
    return _sort_channels_given_hue(hue, (min_, mid_p + min_, max_p + min_))


# A large integer that won't break most systems. Used for float_to_8bit_int.
_BIG_INT = 2**32 - 1


def float_to_8bit_int(float_: float) -> int:
    """Convert a float between 0 and 255 to an int between 0 and 255.

    :param float_: a float in the closed interval [0 .. 255]
    :return: an int in the closed interval [0 .. 255]
    :raise ValueError: if float_ is not in the closed interval [0 .. 255]

    Convert color floats [0 .. 255] to ints [0 .. 255] without rounding, which "short
    changes" 0 and 255.
    """
    if not 0 <= float_ <= _MAX_8BIT:
        msg = f"float argument must be in [0 .. 255], not `{float_}`"
        raise ValueError(msg)
    if float_ % 1:
        high_int = int(float_ / _MAX_8BIT * _BIG_INT)
        return high_int >> 24
    return int(float_)


def float_tuple_to_8bit_int_tuple(rgb: RGB) -> tuple[int, int, int]:
    """Convert an rgb float tuple to an rgb int tuple.

    :param rgb: a tuple of floats in the closed interval [0 .. 255]
    :return: a tuple of ints in the clossdffed interval [0 .. 255]
    """
    red, grn, blu = (float_to_8bit_int(c) for c in rgb)
    return red, grn, blu


def rgb_to_hex(rgb: RGB) -> Hex:
    """Convert rgb to hex.

    :param rgb: a tuple of red, green, and blue values ([0, 255], [0, 255], [0, 255]).
    :return: a hex string representation of the rgb value.
    """
    red, grn, blu = (float_to_8bit_int(c) for c in rgb)
    return f"#{red:02x}{grn:02x}{blu:02x}"


def hex_to_rgb(hex_: Hex) -> tuple[int, int, int]:
    """Convert hex to rgb.

    :param hex_: a hex string representation of an rgb value. With or w/o leading #.
    :return: a tuple of red, green, and blue ints ([0, 255], [0, 255], [0, 255]).
    """
    hex_ = hex_.lstrip("#")
    red, grn, blu = (int(hex_[i : i + 2], 16) for i in (0, 2, 4))
    return (red, grn, blu)
