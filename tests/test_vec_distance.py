"""Test vectorized Lab color distance calculations.

:author: Shay Hill
:created: 2024-08-22
"""

# pyright: reportPrivateUsage=false

import numpy as np

from basic_colormath.distance import (
    _rgb_to_xyz,
    _xyz_to_lab,
    get_delta_e_lab,
    hex_to_lab,
    rgb_to_lab,
    get_sqeuclidean,
    get_sqeuclidean_hex,
    get_euclidean,
    get_euclidean_hex,
    get_delta_e, get_delta_e_hex
)
from basic_colormath.vec_distance import (
    _rgbs_to_xyz,
    _xyzs_to_lab,
    get_deltas_e_lab,
    hexs_to_lab,
    rgbs_to_lab,
    get_sqeuclideans,
    get_sqeuclideans_hex,
    get_euclideans,
    get_euclideans_hex,
    get_deltas_e,
    get_deltas_e_hex
)
from basic_colormath.vec_conversion import hexs_to_rgb


class TestRgbsToXyz:
    def test_match_single(self):
        """Match result of single conversion mapped over array."""
        rgbs = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        xyzs = _rgbs_to_xyz(rgbs)
        for ixs in np.ndindex(rgbs.shape[:-1]):
            r, g, b = map(int, rgbs[ixs])
            xyz = _rgb_to_xyz((r, g, b))
            np.testing.assert_array_almost_equal(xyzs[ixs], xyz)


class TestXyxsToRgb:
    def test_match_single(self):
        """Match result of single conversion mapped over array."""
        xyzs = np.random.rand(10, 11, 12, 3)
        rgbs = _xyzs_to_lab(xyzs)
        for ixs in np.ndindex(xyzs.shape[:-1]):
            x, y, z = xyzs[ixs]
            lab = _xyz_to_lab((x, y, z))
            np.testing.assert_array_almost_equal(rgbs[ixs], lab)


class TestRgbsToLab:
    def test_match_single(self):
        """Match result of single conversion mapped over array."""
        rgbs = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        labs = rgbs_to_lab(rgbs)
        for ixs in np.ndindex(rgbs.shape[:-1]):
            r, g, b = map(int, rgbs[ixs])
            lab = rgb_to_lab((r, g, b))
            np.testing.assert_array_almost_equal(labs[ixs], lab)


class TestHexsToLab:
    def test_match_single(self):
        """Match result of single conversion mapped over array."""
        hexs = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], (10, 11, 12)
        )
        labs = hexs_to_lab(hexs)
        for ixs in np.ndindex(hexs.shape):
            lab = hex_to_lab(hexs[ixs])
            np.testing.assert_array_almost_equal(labs[ixs], lab)

class TestCieDistance:
    def test_match_single_lab(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        rgbs_b = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        labs_a = rgbs_to_lab(rgbs_a)
        labs_b = rgbs_to_lab(rgbs_b)
        deltas = get_deltas_e_lab(labs_a, labs_b)
        for ixs in np.ndindex(labs_a.shape[:-1]):
            delta = get_delta_e_lab(labs_a[ixs], labs_b[ixs])
            np.testing.assert_almost_equal(deltas[ixs], delta)

    def test_match_single_rgb(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        rgbs_b = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        deltas = get_deltas_e(rgbs_a, rgbs_b)
        for ixs in np.ndindex(rgbs_a.shape[:-1]):
            delta = get_delta_e(rgbs_a[ixs], rgbs_b[ixs])
            np.testing.assert_almost_equal(deltas[ixs], delta)

    def test_match_single_hex(self):
        """Match result of single conversion mapped over array."""
        hexs_a = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], (10, 11, 12)
        )
        hexs_b = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], (10, 11, 12)
        )
        deltas = get_deltas_e_hex(hexs_a, hexs_b)
        for ixs in np.ndindex(hexs_a.shape):
            delta = get_delta_e_hex(hexs_a[ixs], hexs_b[ixs])
            np.testing.assert_almost_equal(deltas[ixs], delta)



class TestSquaredEucDistance:
    def test_match_single_rgb(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        rgbs_b = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        sqeuclideans = get_sqeuclideans(rgbs_a, rgbs_b)
        for ixs in np.ndindex(rgbs_a.shape[:-1]):
            sqeuclidean = get_sqeuclidean(rgbs_a[ixs], rgbs_b[ixs])
            np.testing.assert_almost_equal(sqeuclideans[ixs], sqeuclidean)

    def test_match_single_hex(self):
        """Match result of single conversion mapped over array."""
        hexs_a = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], (10, 11, 12)
        )
        hexs_b = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], (10, 11, 12)
        )
        sqeuclideans = get_sqeuclideans_hex(hexs_a, hexs_b)
        for ixs in np.ndindex(hexs_a.shape):
            sqeuclidean = get_sqeuclidean_hex(hexs_a[ixs], hexs_b[ixs])
            np.testing.assert_almost_equal(sqeuclideans[ixs], sqeuclidean)


class TestEuclideanDistance:
    def test_match_single_rgb(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        rgbs_b = np.random.randint(0, 256, (10, 11, 12, 3), dtype=np.uint8)
        euclideans = get_euclideans(rgbs_a, rgbs_b)
        for ixs in np.ndindex(rgbs_a.shape[:-1]):
            euclidean = get_euclidean(rgbs_a[ixs], rgbs_b[ixs])
            np.testing.assert_almost_equal(euclideans[ixs], euclidean)

    def test_match_single_hex(self):
        """Match result of single conversion mapped over array."""
        hexs_a = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], (10, 11, 12)
        )
        hexs_b = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], (10, 11, 12)
        )
        euclideans = get_euclideans_hex(hexs_a, hexs_b)
        for ixs in np.ndindex(hexs_a.shape):
            euclidean = get_euclidean_hex(hexs_a[ixs], hexs_b[ixs])
            np.testing.assert_almost_equal(euclideans[ixs], euclidean)



