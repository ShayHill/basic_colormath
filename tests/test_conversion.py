"""Check simple color conversions.

:author: Shay Hill
:created: 2023-04-30
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

import random
from typing import Tuple

import numpy as np
import pytest
from colormath.color_conversions import convert_color  # type: ignore
from colormath.color_objects import HSLColor, HSVColor, sRGBColor  # type: ignore

from basic_colormath.conversion import (
    _get_hue_from_rgb,
    float_tuple_to_8bit_int_tuple,
    hex_to_rgb,
    hsl_to_rgb,
    hsv_to_rgb,
    rgb_to_hex,
    rgb_to_hsl,
    rgb_to_hsv,
)


def get_hue(rgb: Tuple[float, float, float]) -> float:
    min_ = min(rgb)
    max_ = max(rgb)
    return _get_hue_from_rgb(rgb, min_, max_)


@pytest.fixture(scope="module", params=range(100))
def saturated_rgb_tuple() -> Tuple[float, float, float]:
    """Return a random rgb tuple with full saturation and full value."""
    vals = [0.0, 255.0, random.random() * 255]
    random.shuffle(vals)
    red, grn, blu = vals
    return red, grn, blu


class TestHue:
    def test_upscaled(self) -> None:
        assert get_hue((0, 0, 0)) == 0
        assert get_hue((255, 100, 100)) == 0
        assert get_hue((100, 255, 100)) == 120
        assert get_hue((100, 100, 255)) == 240
        assert get_hue((255, 255, 255)) == 0
        assert get_hue((255, 255, 100)) == 60
        assert get_hue((100, 255, 255)) == 180
        assert get_hue((255, 100, 255)) == 300


class TestHSV:
    def test_vs_colormath(self, rgb_tuple: Tuple[float, float, float]) -> None:
        colormath_rgb = sRGBColor(*rgb_tuple, is_upscaled=True)
        cm_hsv_tuple = convert_color(colormath_rgb, HSVColor).get_value_tuple()
        cm_hsv_tuple = (cm_hsv_tuple[0], cm_hsv_tuple[1] * 100, cm_hsv_tuple[2] * 100)
        our_hsv = rgb_to_hsv(rgb_tuple)
        assert np.allclose(cm_hsv_tuple, our_hsv)

    def test_reverse(self, rgb_tuple: Tuple[float, float, float]) -> None:
        """Test that the hue is the same when the rgb tuple is reversed."""
        hsv = rgb_to_hsv(rgb_tuple)
        rgb = hsv_to_rgb(hsv)
        assert np.allclose(rgb, rgb_tuple)


class TestHSL:
    def test_vs_colormath(self, rgb_tuple: Tuple[float, float, float]) -> None:
        colormath_rgb = sRGBColor(*rgb_tuple, is_upscaled=True)
        cm_hsl_tuple = convert_color(colormath_rgb, HSLColor).get_value_tuple()
        cm_hsl_tuple = (cm_hsl_tuple[0], cm_hsl_tuple[1] * 100, cm_hsl_tuple[2] * 100)
        our_hsl = rgb_to_hsl(rgb_tuple)
        assert np.allclose(cm_hsl_tuple, our_hsl)

    def test_reverse(self, rgb_tuple: Tuple[float, float, float]) -> None:
        """Test that the hue is the same when the rgb tuple is reversed."""
        hsl = rgb_to_hsl(rgb_tuple)
        rgb = hsl_to_rgb(hsl)
        assert np.allclose(rgb, rgb_tuple)


class TestHex:
    def test_known_values(self) -> None:
        assert rgb_to_hex((0, 0, 0)) == "#000000"
        assert rgb_to_hex((255, 255, 255)) == "#ffffff"
        assert rgb_to_hex((255, 0, 0)) == "#ff0000"
        assert rgb_to_hex((0, 255, 0)) == "#00ff00"
        assert rgb_to_hex((0, 0, 255)) == "#0000ff"
        assert rgb_to_hex((255, 255, 0)) == "#ffff00"
        assert rgb_to_hex((0, 255, 255)) == "#00ffff"
        assert rgb_to_hex((255, 0, 255)) == "#ff00ff"
        assert rgb_to_hex((128, 128, 128)) == "#808080"

    def test_reverse(self, rgb_tuple: Tuple[float, float, float]) -> None:
        """Test that the hue is the same when the rgb tuple is reversed."""
        rgb = float_tuple_to_8bit_int_tuple(rgb_tuple)
        hex_ = rgb_to_hex(rgb_tuple)
        rgb = hex_to_rgb(hex_)
        assert rgb == rgb_tuple
