"""Convert between (upscaled) RGB and HSV or HSL color arrays.

:author: Shay Hill
:created: 2024-08-22
"""

from typing import Any, TypeVar, cast

import numpy as np
from numpy import typing as npt

from basic_colormath.conversion import hex_to_rgb, rgb_to_hex

_FloatArray = npt.NDArray[np.float64]
_Uint8Array = npt.NDArray[np.uint8]
_TArray = TypeVar("_TArray", bound=npt.NDArray[Any])

_MAX_8BIT = 255


def _get_hues_from_rgbs(
    rgbs: npt.ArrayLike, mins: _FloatArray, maxs: _FloatArray
) -> _FloatArray:
    """Get the hue values in degrees from an array of rgb tuples.

    :param rgbs: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :param mins: (..., 1) pre-calculated minimum values in the rgb tuples.
    :param maxs: (..., 1) pre-calculated maximum values in the rgb tuples.
    :return: (..., 1) The hue values in degrees, [0, 360).
    """
    reds, grns, blus = (np.asarray(rgbs, dtype=np.float64)[..., i] for i in range(3))
    hues = np.zeros_like(reds).astype(float)
    deltas = maxs - mins
    delta_mask = maxs != mins
    rmask = (maxs == reds) & delta_mask
    gmask = (maxs == grns) & ~rmask & delta_mask
    bmask = ~rmask & ~gmask & delta_mask
    hues[rmask] = 60 * ((grns[rmask] - blus[rmask]) / (deltas[rmask])) + 360
    hues[rmask] %= 360
    hues[gmask] = 60 * ((blus[gmask] - reds[gmask]) / (deltas[gmask])) + 120
    hues[bmask] = 60 * ((reds[bmask] - grns[bmask]) / (deltas[bmask])) + 240
    return hues


def _sort_channels_given_hues(hues: _FloatArray, min_mid_max: _TArray) -> _TArray:
    """Sort the channels of an rgb tuple given a hue value.

    :param hue: The hue value in degrees, [0, 360).
    :param min_mid_max: A tuple of the minimum, middle, and maximum values in the rgb
    :return: an arrangement of min_mid_max that corresponds to the hue.

    Regardless of saturation, lightness, value, etc., the order of minimum, middle,
    and maximum values in an rgb tuple will be consistent. For instance, given a hue
    of 0, the red channel will always be the highest value.
    """
    rgbs: _TArray = cast("_TArray", np.copy(min_mid_max))
    hue_sextants = (hues // 60).astype(int)
    sextant2order = {
        0: (2, 1, 0),
        1: (1, 2, 0),
        2: (0, 2, 1),
        3: (0, 1, 2),
        4: (1, 0, 2),
        5: (2, 0, 1),
    }
    for sextant, order in sextant2order.items():
        mask = hue_sextants == sextant
        for i, j in enumerate(order):
            if i == j:
                continue
            rgbs[mask, i] = min_mid_max[mask, j]
    return rgbs


def rgbs_to_hsv(rgbs: npt.ArrayLike) -> _FloatArray:
    """Convert from rgb to hsv.

    :param rgbs: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :return: an array (...,3) of hue, saturation, and value
        [0, 360), [0, 100], [0, 100].
    """
    mins = np.min(rgbs, axis=-1)
    maxs = np.max(rgbs, axis=-1)
    hsvs = np.zeros_like(rgbs).astype(float)
    hsvs[..., 0] = _get_hues_from_rgbs(rgbs, mins, maxs)
    maxzero = maxs == 0
    hsvs[maxzero, 1] = 0
    puts = (1 - (mins[~maxzero] / maxs[~maxzero])) * 100
    hsvs[~maxzero, 1] = puts
    hsvs[..., 2] = maxs / 2.55
    return hsvs


def hsvs_to_rgb(hsvs: npt.ArrayLike) -> _FloatArray:
    """Convert from hsv to rgb.

    :param hsv: an array (...,3) of hue, sat, and val values
        [0, 360), [0, 100], [0, 100]
    :return: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    """
    hues, sats, vals = (np.asarray(hsvs, dtype=np.float64)[..., i] for i in range(3))
    maxs = vals * 2.55
    mins = maxs - (sats / 100 * maxs)
    hue_mid_ratio = 1 - abs((hues / 60) % 2 - 1)
    mids = hue_mid_ratio * (maxs - mins) + mins
    mins_mids_maxs = np.stack((mins, mids, maxs), axis=-1)
    return _sort_channels_given_hues(hues, mins_mids_maxs)


def rgbs_to_hsl(rgbs: npt.ArrayLike) -> _FloatArray:
    """Convert rgb to hsl.

    :param rgbs: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :return: an array (...,3) of hue, sat, and lightness values
        [0, 360), [0, 100], [0, 100].
    """
    mins = np.min(rgbs, axis=-1).astype(float)
    maxs = np.max(rgbs, axis=-1).astype(float)
    deltas = maxs - mins
    delta_mask = deltas != 0
    hsls = np.zeros_like(rgbs).astype(float)
    # hue
    hsls[..., 0] = _get_hues_from_rgbs(rgbs, mins, maxs)
    # lightness
    hsls[..., 2] = maxs + mins
    # saturation
    lo_mask = (hsls[..., 2] <= _MAX_8BIT) & delta_mask
    hi_mask = ~lo_mask & delta_mask
    hsls[lo_mask, 1] = deltas[lo_mask] / (0.01 * hsls[lo_mask, 2])
    hsls[hi_mask, 1] = deltas[hi_mask] / (5.1 - (0.01 * hsls[hi_mask, 2]))
    hsls[..., 2] /= 5.1
    return hsls


def hsls_to_rgb(hsls: npt.ArrayLike) -> _FloatArray:
    """Convert hsl to rgb.

    :param hsls: an array (...,3) of hue, sat, and lightness values
        [0, 360), [0, 100], [0, 100]
    :return: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    """
    hues, sats, ligs = (np.asarray(hsls, dtype=np.float64)[..., i] for i in range(3))
    max_ps = (100 - abs(2 * ligs - 100)) * sats * 0.0255
    mid_ps = max_ps * (1 - abs((hues / 60) % 2 - 1))
    mins = (ligs - max_ps / 5.1) * 2.55
    mins_mids_maxs = np.stack((mins, mins + mid_ps, mins + max_ps), axis=-1)
    return _sort_channels_given_hues(hues, mins_mids_maxs)


# A large integer that won't break most systems. Used for float_to_8bit_int.
_BIG_INT: int = 2**32 - 1


def floats_to_uint8(rgbs: npt.ArrayLike) -> _Uint8Array:
    """Convert a float between 0 and 255 to an int between 0 and 255.

    :param rgbs: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :return: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255] distributed (almost) evenly over 0 to 255.

    This function exists because
    `np.array([254.999]).astype(np.uint8)` -> `array([254], dtype=uint8)`

    Numpy uses floor for int conversion. Round would be better, but would still
    "short change" 0 and 255. This function gives a better distribution.
    """
    rgbs = np.asarray(rgbs)
    if rgbs.dtype is np.uint8:
        return rgbs.astype(np.uint8)
    big_ints = (np.clip(rgbs, 0, 255) / 255 * _BIG_INT).astype(np.uint32)
    return (big_ints >> 24).astype(np.uint8)


def rgbs_to_hex(rgbs: npt.ArrayLike) -> npt.NDArray[np.str_]:
    """Convert rgb to hex.

    :param rgbs: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    :return: an array (..., 1) of hex strings. e.g. '#ff0000'
    """
    rgbs = floats_to_uint8(rgbs)
    return np.apply_along_axis(rgb_to_hex, -1, rgbs)


def _hex_to_rgb(hex_: npt.ArrayLike) -> _Uint8Array:
    """Convert a hex string to an rgb tuple.

    :param hex_: A hex string. e.g. '#ff0000'
    :return: A tuple of red, green, and blue values [0, 255].
    """
    hex_ = np.asarray(hex_)
    return np.array(hex_to_rgb(hex_[0]), dtype=np.uint8)


def hexs_to_rgb(hexs: npt.ArrayLike) -> _Uint8Array:
    """Convert hex to rgb.

    :param hexs: an array (..., 1) of hex strings. e.g. '#ff0000'
    :return: an array (..., 3) of red, green, and blue values
        [0, 255], [0, 255], [0, 255]
    """
    hexs = np.asarray(hexs)
    return np.apply_along_axis(_hex_to_rgb, -1, hexs[..., np.newaxis])
