"""Mix rgb or hex color values.

:author: Shay Hill
:created: 2023-04-30
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from basic_colormath.conversion import hex_to_rgb, rgb_to_hex

if TYPE_CHECKING:
    from basic_colormath.type_hints import RGB, Hex

_Ratio = Union[float, "tuple[float, ...]", None]


def _split_float(f: float, num: int) -> tuple[float, ...]:
    """Divide a float into num parts.

    :param f: float to divide
    :param num: number of parts
    :return: tuple of floats
    """
    return (f / num,) * num


def _infer_ps(ratio: float | tuple[float, ...] | None, num: int) -> tuple[float, ...]:
    """Infer p values from a single float or tuple of floats.

    :param ratios: float or tuple of floats
    :param num: number of ratios to return (len of rgb_args)
    :return: tuple of floats summing to 1
    :raise ValueError: if ratios cannot be distributed across values and sum to 1

    Three cases:
    1. ratio is None: return (1/num, ...)
    2. ratio is a float: return (ratios, ((1-ratios) / (num-1), ...)
    3. ratio is a tuple: fill in missing ratios with (1-sum(ratios)) / missing
    """
    # preserve ratio arg for error messages
    ratio_arg = ratio
    if ratio is None:
        return _split_float(1, num)
    if isinstance(ratio, (float, int)):
        ratio = (ratio,)

    if any(r < 0 for r in ratio):
        msg = f"ratios must be >= 0, not {ratio_arg}"
        raise ValueError(msg)
    if len(ratio) > num:
        msg = f"ratios has {len(ratio)} elements, but only <= {num} are allowed"
        raise ValueError(msg)
    sum_ratios = sum(ratio)
    missing = num - len(ratio)
    if sum_ratios == 0 and missing == 0:
        msg = f"ratios must sum to > 0, not {sum_ratios}"
        raise ValueError(msg)
    if sum_ratios >= 1:
        return tuple(r / sum_ratios for r in ratio) + _split_float(0, missing)
    return ratio + _split_float(1 - sum_ratios, missing)


def scale_rgb(rgb: RGB, scalar: float) -> RGB:
    """Scale an rgb tuple by a scalar.

    :param rgb: rgb tuple to scale ([0, 255], [0, 255], [0, 255])
    :param scalar: scalar to multiply each element by
    :return: scaled rgb tuple
    """
    red, grn, blu = (scalar * i for i in rgb)
    return red, grn, blu


def mix_rgb(*rgb_args: RGB, ratio: _Ratio = None) -> RGB:
    """Mix any number of rgb tuples.

    :param rgb_args: rgb tuples ([0, 255], [0, 255], [0, 255])
    :param ratio: 0.0 to 1.0 for the weight of the first rgb_arg or a tuple of floats
        to distribute across rgb_args or None for equal ratios. Ratios will be
        normalized and (if fewer ratios than colors are provided) the remaining
        ratios will be equal.
    :return: rgb tuple ([0, 255], [0, 255], [0, 255])
    """
    ps = _infer_ps(ratio, len(rgb_args))
    scaled_rgbs = [scale_rgb(rgb, p) for rgb, p in zip(rgb_args, ps)]
    red, grn, blu = (sum(i) for i in zip(*scaled_rgbs))
    return (red, grn, blu)


def scale_hex(hex_: Hex, scalar: float) -> Hex:
    """Scale a hex color by a scalar.

    :param hex_: hex color with or without leading #
    :param scalar: scalar to multiply each element by
    :return: scaled hex color with leading #
    """
    return rgb_to_hex(scale_rgb(hex_to_rgb(hex_), scalar))


def mix_hex(*hex_args: Hex, ratio: _Ratio = None) -> Hex:
    """Mix any number of hex colors.

    :param hex_args: hex colors with or without leading #
    :param ratio: 0.0 to 1.0 for the weight of the first rgb_arg or a tuple of floats
        to distribute across rgb_args or None for equal ratios. Ratios will be
        normalized and (if fewer ratios than colors are provided) the remaining
        ratios will be equal.
    :return: hex string with leading #
    """
    return rgb_to_hex(mix_rgb(*(hex_to_rgb(i) for i in hex_args), ratio=ratio))
