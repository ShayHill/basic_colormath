"""Compare results of delta_e to python-colormath.

:author: Shay Hill
:created: 2023-04-29
"""

import random
from typing import Any, Tuple, cast

import numpy as np
import numpy.typing as npt
import pytest
from colormath.color_conversions import convert_color  # type: ignore
from colormath.color_diff import delta_e_cie2000  # type: ignore
from colormath.color_objects import LabColor, XYZColor, sRGBColor  # type: ignore

from basic_colormath.conversion import hex_to_rgb, rgb_to_hex
from basic_colormath.distance import (
    get_delta_e,
    get_delta_e_hex,
    get_euclidean,
    get_euclidean_hex,
    get_sqeuclidean,
)


def _patch_asscalar(a: npt.NDArray[np.float_]) -> float:
    """Alias for np.item(). Patch np.asscalar for colormath.

    :param a: numpy array
    :return: input array as scalar


    looks like python-colormath is abandoned. The code on PyPI will not work with the
    latest numpy because asscaler has been removed from numpy. This kludges it.
    np.asscalar is not called in this module, but it is called when computing
    delta-e.
    """
    return a.item()


np.asscalar = _patch_asscalar  # type: ignore


def _colormath_delta_e(
    rgb_a: Tuple[float, float, float], rgb_b: Tuple[float, float, float]
) -> float:
    srgb_a = cast(Any, sRGBColor(*rgb_a, is_upscaled=True))
    srgb_b = cast(Any, sRGBColor(*rgb_b, is_upscaled=True))
    lab_a = cast(Any, convert_color(srgb_a, LabColor))
    lab_b = cast(Any, convert_color(srgb_b, LabColor))
    return cast(float, delta_e_cie2000(lab_a, lab_b))


@pytest.fixture(scope="module", params=range(100))
def rgb_pair() -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return a pair of random rgb tuples."""
    red1, grn1, blu1 = (random.randint(0, 255) for _ in range(3))
    red2, grn2, blu2 = (random.randint(0, 255) for _ in range(3))
    return ((red1, grn1, blu1), (red2, grn2, blu2))


class TestDeltaE:
    def test_compare_rgb_to_colormath(
        self, rgb_pair: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ):
        """Test that our delta-e is close to colormath's delta-e."""
        rgb_a, rgb_b = rgb_pair
        colormath_delta_e = _colormath_delta_e(rgb_a, rgb_b)
        our_delta_e = get_delta_e(rgb_a, rgb_b)
        assert abs(colormath_delta_e - our_delta_e) < 0.0001

    def test_compare_hex_to_colormath(
        self, rgb_pair: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ):
        """Test that our delta-e is close to colormath's delta-e."""
        hex_a, hex_b = rgb_to_hex(rgb_pair[0]), rgb_to_hex(rgb_pair[1])
        rgb_a, rgb_b = hex_to_rgb(hex_a), hex_to_rgb(hex_b)

        colormath_delta_e = _colormath_delta_e(rgb_a, rgb_b)
        our_delta_e = get_delta_e_hex(hex_a, hex_b)
        assert abs(colormath_delta_e - our_delta_e) < 0.0001


class TestEuclideanDistance:
    def test_known_sqeuclidean(self):
        """Test that our euclidean distance is correct."""
        assert get_sqeuclidean((0, 0, 0), (1, 1, 1)) == 3
        assert get_sqeuclidean((0, 0, 0), (1, 2, 1)) == 6

    def test_known_euclidean(self):
        """Test that our euclidean distance is correct."""
        assert np.isclose(get_euclidean((0, 0, 0), (1, 1, 1)), 3**0.5)
        assert np.isclose(get_euclidean((0, 0, 0), (1, 2, 1)), 6**0.5)

    def test_hext_vs_rgb(
        self, rgb_pair: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ):
        """Test that our euclidean distance is correct."""
        hex_a, hex_b = rgb_to_hex(rgb_pair[0]), rgb_to_hex(rgb_pair[1])
        rgb_a, rgb_b = hex_to_rgb(hex_a), hex_to_rgb(hex_b)
        assert get_euclidean(rgb_a, rgb_b) == get_euclidean_hex(hex_a, hex_b)


def _compare_speed():
    """Compare the speed of our delta-e to colormath's delta-e."""
    import time

    count = 10000
    rgb_as = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(count)]
    rgb_bs = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(count)]

    beg = time.time()
    for rgb_a, rgb_b in zip(rgb_as, rgb_bs):
        _ = _colormath_delta_e(rgb_a, rgb_b)
    end = time.time()
    colormath_time = end - beg
    print(f"colormath: {colormath_time}")
    beg = time.time()
    for rgb_a, rgb_b in zip(rgb_as, rgb_bs):
        _ = get_delta_e(rgb_a, rgb_b)
    end = time.time()
    our_time = end - beg
    print(f"delta_e: {our_time}")

    print(f"speedup: {colormath_time / our_time}")


if __name__ == "__main__":
    """A speed comparison for our delta-e and colormath's delta-e."""
    _compare_speed()
