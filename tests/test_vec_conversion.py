"""Test vectorized conversion functions.

:author: Shay Hill
:created: 2024-08-22
"""

# pyright: reportPrivateUsage=false

import numpy as np

from basic_colormath.conversion import (
    float_tuple_to_8bit_int_tuple,
    rgb_to_hex,
    rgb_to_hsl,
    rgb_to_hsv, hex_to_rgb
)
from basic_colormath.vec_conversion import (
    _get_hues_from_rgbs,
    floats_to_uint8,
    hexs_to_rgb,
    hsls_to_rgb,
    hsvs_to_rgb,
    rgbs_to_hex,
    rgbs_to_hsl,
    rgbs_to_hsv,
)


class TestGetHuesFromRgbs:
    def test_explicit(self):
        rgbs = np.array(
            [
                [0, 0, 0],
                [255, 0, 0],
                [255, 255, 0],
                [0, 255, 0],
                [0, 255, 255],
                [0, 0, 255],
                [255, 0, 255],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )
        # set mins as the minimum value for each of the rgb channels
        mins = np.min(rgbs, axis=-1)
        maxs = np.max(rgbs, axis=-1)
        hues = _get_hues_from_rgbs(rgbs, mins, maxs)
        np.testing.assert_array_almost_equal([0, 0, 60, 120, 180, 240, 300, 0], hues)


class TestRgbsToHsvs:
    def test_match_single(self):
        """Match result of single conversion mapped over array."""
        rgbs = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        hsvs = rgbs_to_hsv(rgbs)
        for ixs in np.ndindex(rgbs.shape[:-1]):
            r, g, b = map(int, rgbs[ixs])
            hsv = rgb_to_hsv((r, g, b))
            np.testing.assert_array_almost_equal(hsvs[ixs], hsv)


class TestHscsToRgbs:
    def test_roflection(self):
        """rgb to hsv to rgb to hsv

        hsv to rgb to hsv will not always be the same, because hue information is
        lost when saturation or value are zero.
        """
        rgbs_a = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        hsvs_a = rgbs_to_hsv(rgbs_a)
        rgbs_b = hsvs_to_rgb(hsvs_a)
        hsvs_b = rgbs_to_hsv(rgbs_b)
        np.testing.assert_array_almost_equal(hsvs_a, hsvs_b)


class TestRgbsToHsls:
    def test_match_single(self):
        """Match result of single conversion mapped over array."""
        rgbs = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        hsls = rgbs_to_hsl(rgbs)
        for ixs in np.ndindex(rgbs.shape[:-1]):
            r, g, b = map(int, rgbs[ixs])
            hsl = rgb_to_hsl((r, g, b))
            np.testing.assert_array_almost_equal(hsls[ixs], hsl)


class TestHslsToRgbs:
    def test_reflection(self):
        """rgb to hsl to rgb to hsl

        hsl to rgb to hsl will not always be the same, because hue information is
        lost when saturation or lightness are zero.
        """
        rgbs_a = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        hsls_a = rgbs_to_hsl(rgbs_a)
        rgbs_b = hsls_to_rgb(hsls_a)
        hsls_b = rgbs_to_hsl(rgbs_b)
        np.testing.assert_array_almost_equal(hsls_a, hsls_b)


class TestRgbsToHexs:
    def test_match_single(self):
        """Match result of single conversion mapped over array."""
        rgbs = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        hexs = rgbs_to_hex(rgbs)
        for ixs in np.ndindex(rgbs.shape[:-1]):
            r, g, b = map(int, rgbs[ixs])
            hex_ = rgb_to_hex((r, g, b))
            assert hexs[ixs] == hex_


class TestHexsToRgbs:
    def test_match_single(self):
        """Match result of single conversion mapped over array."""
        hexs = np.array(
            [
                "#000000",
                "#ff0000",
                "#ffff00",
                "#00ff00",
                "#00ffff",
                "#0000ff",
                "#ff00ff",
                "#ffffff",
            ],
            dtype=np.str_,
        )
        rgbs = hexs_to_rgb(hexs)
        for ixs in np.ndindex(hexs.shape):
            rgb = hex_to_rgb(hexs[ixs])
            np.testing.assert_array_almost_equal(rgbs[ixs], rgb)

    def test_reflection(self):
        """rgb to hex to rgb

        hex to rgb to hex will always be the same.
        """
        rgbs_a = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        hexs_a = rgbs_to_hex(rgbs_a)
        rgbs_b = hexs_to_rgb(hexs_a)
        hexs_b = rgbs_to_hex(rgbs_b)
        np.testing.assert_array_equal(hexs_a, hexs_b)


class TestFloatsToUint8:
    def test_match_single(self):
        """Match result of single conversion mapped over array."""
        floats = np.random.rand(10, 11, 12, 3) * 255
        uint8s = floats_to_uint8(floats)
        for ixs in np.ndindex(floats.shape[:-1]):
            r, g, b = map(float, floats[ixs])
            uint8 = float_tuple_to_8bit_int_tuple((r, g, b))
            np.testing.assert_array_almost_equal(uint8s[ixs], uint8)
