"""Mix rgb or hex color values.

:author: Shay Hill
:created: 2023-04-30
"""

from typing import Union

from basic_colormath.conversion import hex_to_rgb, rgb_to_hex
from basic_colormath.helpers import infer_ps
from basic_colormath.type_hints import Hex, Rgb, RgbLike

Ratio = Union[float, "tuple[float, ...]", None]


def scale_rgb(rgb: RgbLike, scalar: float) -> Rgb:
    """Scale an rgb tuple by a scalar.

    :param rgb: rgb tuple to scale ([0, 255], [0, 255], [0, 255])
    :param scalar: scalar to multiply each element by
    :return: scaled rgb tuple
    """
    red, grn, blu = (scalar * i for i in rgb)
    return red, grn, blu


def mix_rgb(*rgb_args: RgbLike, ratio: Ratio = None) -> Rgb:
    """Mix any number of rgb tuples.

    :param rgb_args: rgb tuples ([0, 255], [0, 255], [0, 255])
    :param ratio: 0.0 to 1.0 for the weight of the first rgb_arg or a tuple of floats
        to distribute across rgb_args or None for equal ratios. Ratios will be
        normalized and (if fewer ratios than colors are provided) the remaining
        ratios will be equal.
    :return: rgb tuple ([0, 255], [0, 255], [0, 255])
    """
    ps = infer_ps(ratio, len(rgb_args))
    scaled_rgbs = [scale_rgb(rgb, p) for rgb, p in zip(rgb_args, ps, strict=True)]
    red, grn, blu = (sum(i) for i in zip(*scaled_rgbs, strict=True))
    return (red, grn, blu)


def scale_hex(hex_: Hex, scalar: float) -> Hex:
    """Scale a hex color by a scalar.

    :param hex_: hex color with or without leading #
    :param scalar: scalar to multiply each element by
    :return: scaled hex color with leading #
    """
    return rgb_to_hex(scale_rgb(hex_to_rgb(hex_), scalar))


def mix_hex(*hex_args: Hex, ratio: Ratio = None) -> Hex:
    """Mix any number of hex colors.

    :param hex_args: hex colors with or without leading #
    :param ratio: 0.0 to 1.0 for the weight of the first rgb_arg or a tuple of floats
        to distribute across rgb_args or None for equal ratios. Ratios will be
        normalized and (if fewer ratios than colors are provided) the remaining
        ratios will be equal.
    :return: hex string with leading #
    """
    return rgb_to_hex(mix_rgb(*(hex_to_rgb(i) for i in hex_args), ratio=ratio))
