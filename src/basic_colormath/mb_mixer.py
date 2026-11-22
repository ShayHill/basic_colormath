"""If mixbox is installed in the environment, provide additional mixing functions.

:author: Shay Hill
:created: 2026-10-10
"""

from typing import Union

import mixbox  # pyright: ignore[reportMissingTypeStubs]

from basic_colormath.conversion import hex_to_rgb, rgb_to_hex
from basic_colormath.helpers import infer_ps
from basic_colormath.type_hints import Hex, Rgb, RgbLike

Ratio = Union[float, "tuple[float, ...]", None]
Latent = tuple[float, float, float, float, float, float, float]


def _scale_latent(latent: Latent, scalar: float) -> Latent:
    """Scale a latent color by a ratio.

    :param latent: latent color
    :param ratio: 0.0 to 1.0 for the weight of the latent color
    :return: scaled latent color
    """
    a, b, c, d, e, f, g = (i * scalar for i in latent)
    return a, b, c, d, e, f, g


def mixbox_latent(*latent_args: Latent, ratio: Ratio = None) -> Rgb:
    """Mix any number of latent colors.

    :param latent_args: mixbox latent colors (a, b, c, d, e, f, g)
    :param ratio: 0.0 to 1.0 for the weight of the first latent_arg or a tuple of floats
        to distribute across latent_args or None for equal ratios. Ratios will be
        normalized and (if fewer ratios than colors are provided) the remaining
        ratios will be equal.
    :return: rgb tuple ([0, 255], [0, 255], [0, 255])
    """
    ps = infer_ps(ratio, len(latent_args))
    lats = [_scale_latent(t, p) for t, p in zip(latent_args, ps, strict=True)]
    a, b, c, d, e, f, g = tuple(sum(i) for i in zip(*lats, strict=True))
    return mixbox.latent_to_rgb((a, b, c, d, e, f, g))  # pyright: ignore[reportUnknownMemberType]


def mixbox_rgb(*rgb_args: RgbLike, ratio: Ratio = None) -> Rgb:
    """Mix any number of rgb tuples.

    :param rgb_args: rgb tuples ([0, 255], [0, 255], [0, 255])
    :param ratio: 0.0 to 1.0 for the weight of the first rgb_arg or a tuple of floats
        to distribute across rgb_args or None for equal ratios. Ratios will be
        normalized and (if fewer ratios than colors are provided) the remaining
        ratios will be equal.
    :return: rgb tuple ([0, 255], [0, 255], [0, 255])
    """
    return mixbox_latent(*map(mixbox.rgb_to_latent, rgb_args), ratio=ratio)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]


def mixbox_hex(*hex_args: Hex, ratio: Ratio = None) -> Hex:
    """Mix any number of hex colors.

    :param hex_args: hex colors with or without leading #
    :param ratio: 0.0 to 1.0 for the weight of the first rgb_arg or a tuple of floats
        to distribute across rgb_args or None for equal ratios. Ratios will be
        normalized and (if fewer ratios than colors are provided) the remaining
        ratios will be equal.
    :return: hex color with leading #
    """
    return rgb_to_hex(mixbox_rgb(*(hex_to_rgb(hex_) for hex_ in hex_args), ratio=ratio))
