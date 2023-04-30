"""Test color mixing

:author: Shay Hill
:created: 2023-04-30
"""

from typing import List, Tuple

import pytest

from basic_colormath import mix
from basic_colormath.conversion import (
    float_tuple_to_8bit_int_tuple,
    hex_to_rgb,
    rgb_to_hex,
)


class TestMixRGB:
    def test_mix(self):
        assert mix.mix_rgb((0, 0, 0), (255, 255, 255)) == (127.5, 127.5, 127.5)
        assert mix.mix_rgb((0, 0, 0), (0, 0, 0), (3, 3, 3)) == (1, 1, 1)

    def test_float_ratio(self):
        assert mix.mix_rgb((0, 0, 0), (255, 255, 255), ratio=1 / 255) == (254, 254, 254)

    def test_float_ratio_over_one(self):
        """Weight of first arg is 1, others 0"""
        assert mix.mix_rgb((255, 255, 255), (0, 0, 0), ratio=2) == (255, 255, 255)

    def test_negative_float_ratio(self):
        """Weight of first arg is 0, others 1"""
        with pytest.raises(ValueError):
            _ = mix.mix_rgb((255, 255, 255), (0, 0, 0), ratio=-1 / 255)

    def test_zero_float_ratio(self):
        """Weight of first arg is 0, others 1"""
        with pytest.raises(ValueError):
            _ = mix.mix_rgb((255, 255, 255), (0, 0, 0), ratio=-1 / 255)

    def test_negative_ratio_in_tuple(self):
        with pytest.raises(ValueError):
            _ = mix.mix_rgb((255, 255, 255), (0, 0, 0), ratio=(-1 / 255,))

    def test_too_many_ratios(self):
        with pytest.raises(ValueError):
            _ = mix.mix_rgb((255, 255, 255), (0, 0, 0), ratio=(1, 1, 1))

    def test_ratios_sum_to_zero_no_room(self):
        """Raise value error if a ratio is given for all colors and they sum to zero"""
        with pytest.raises(ValueError):
            _ = mix.mix_rgb((255, 255, 255), (0, 0, 0), ratio=(0, 0))

    def test_ratios_sum_to_zero_more_args(self):
        """Allow if ratios sum to zero, but args remain for more colors"""
        assert mix.mix_rgb((255, 255, 255), (0, 0, 0), (1, 1, 1), ratio=(0, 0)) == (
            1,
            1,
            1,
        )

    def test_ratios_sum_over_one(self):
        """Scale to sum of ratios"""
        assert mix.mix_rgb((255, 255, 255), (0, 0, 0), (1, 1, 1), ratio=(2, 2)) == (
            127.5,
            127.5,
            127.5,
        )


class TestMixHEX:
    def test_vs_rgb(self, rgb_args: List[Tuple[int, int, int]]):
        mixed_rgb = mix.mix_rgb(*rgb_args)
        mixed_hex = mix.mix_hex(*[rgb_to_hex(x) for x in rgb_args])
        mixed_rgb_ints = float_tuple_to_8bit_int_tuple(mixed_rgb)
        mixed_hex_ints = hex_to_rgb(mixed_hex)
        assert mixed_rgb_ints == mixed_hex_ints
