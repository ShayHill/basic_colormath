"""Test vectorized Lab color distance calculations.

:author: Shay Hill
:created: 2024-08-22
"""

# pyright: reportPrivateUsage=false

import numpy as np

from basic_colormath.distance import (
    _rgb_to_xyz,
    _xyz_to_lab,
    get_delta_e,
    get_delta_e_hex,
    get_delta_e_lab,
    get_euclidean,
    get_euclidean_hex,
    get_sqeuclidean,
    get_sqeuclidean_hex,
    hex_to_lab,
    rgb_to_lab,
)
from basic_colormath.vec_distance import (
    _rgbs_to_xyz,
    _xyzs_to_lab,
    get_delta_e_matrix,
    get_delta_e_matrix_hex,
    get_delta_e_matrix_lab,
    get_deltas_e,
    get_deltas_e_hex,
    get_deltas_e_lab,
    get_euclidean_matrix,
    get_euclidean_matrix_hex,
    get_euclideans,
    get_euclideans_hex,
    get_sqeuclidean_matrix,
    get_sqeuclidean_matrix_hex,
    get_sqeuclideans,
    get_sqeuclideans_hex,
    hexs_to_lab,
    rgbs_to_lab,
)


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


class TestProximityMatrices:
    def test_match_single_delta_e_lab(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (12, 3), dtype=np.uint8)
        labs_a = rgbs_to_lab(rgbs_a)
        p_mat = get_delta_e_matrix_lab(labs_a)
        for i, aaa in enumerate(labs_a):
            for j, bbb in enumerate(labs_a):
                delta = get_delta_e_lab(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], delta)

    def test_match_single_delta_e_rgb(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (12, 3), dtype=np.uint8)
        p_mat = get_delta_e_matrix(rgbs_a)
        for i, aaa in enumerate(rgbs_a):
            for j, bbb in enumerate(rgbs_a):
                delta = get_delta_e(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], delta)

    def test_match_single_delta_e_hex(self):
        """Match result of single conversion mapped over array."""
        hexs_a = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], 12
        )
        p_mat = get_delta_e_matrix_hex(hexs_a)
        for i, aaa in enumerate(hexs_a):
            for j, bbb in enumerate(hexs_a):
                delta = get_delta_e_hex(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], delta)

    def test_match_single_sqeuclidean_rgb(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (12, 3), dtype=np.uint8)
        p_mat = get_sqeuclidean_matrix(rgbs_a)
        for i, aaa in enumerate(rgbs_a):
            for j, bbb in enumerate(rgbs_a):
                sqeuclidean = get_sqeuclidean(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], sqeuclidean)

    def test_match_single_sqeuclidean_hex(self):
        """Match result of single conversion mapped over array."""
        hexs_a = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], 12
        )
        p_mat = get_sqeuclidean_matrix_hex(hexs_a)
        for i, aaa in enumerate(hexs_a):
            for j, bbb in enumerate(hexs_a):
                sqeuclidean = get_sqeuclidean_hex(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], sqeuclidean)

    def test_match_single_euclidean_rgb(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (12, 3), dtype=np.uint8)
        p_mat = get_euclidean_matrix(rgbs_a)
        for i, aaa in enumerate(rgbs_a):
            for j, bbb in enumerate(rgbs_a):
                euclidean = get_euclidean(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], euclidean)

    def test_match_single_euclidean_hex(self):
        """Match result of single conversion mapped over array."""
        hexs_a = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], 12
        )
        p_mat = get_euclidean_matrix_hex(hexs_a)
        for i, aaa in enumerate(hexs_a):
            for j, bbb in enumerate(hexs_a):
                euclidean = get_euclidean_hex(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], euclidean)


class TestCrossProximityMatrices:
    def test_match_single_delta_e_lab(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (12, 3), dtype=np.uint8)
        labs_a = rgbs_to_lab(rgbs_a)
        rgbs_b = np.random.randint(0, 256, (24, 3), dtype=np.uint8)
        labs_b = rgbs_to_lab(rgbs_b)
        p_mat = get_delta_e_matrix_lab(labs_a, labs_b)
        for i, aaa in enumerate(labs_a):
            for j, bbb in enumerate(labs_b):
                delta = get_delta_e_lab(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], delta)

    def test_match_single_delta_e_rgb(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (12, 3), dtype=np.uint8)
        rgbs_b = np.random.randint(0, 256, (24, 3), dtype=np.uint8)
        p_mat = get_delta_e_matrix(rgbs_a, rgbs_b)
        for i, aaa in enumerate(rgbs_a):
            for j, bbb in enumerate(rgbs_b):
                delta = get_delta_e(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], delta)

    def test_match_single_delta_e_hex(self):
        """Match result of single conversion mapped over array."""
        hexs_a = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], 12
        )
        hexs_b = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], 24
        )
        p_mat = get_delta_e_matrix_hex(hexs_a, hexs_b)
        for i, aaa in enumerate(hexs_a):
            for j, bbb in enumerate(hexs_b):
                delta = get_delta_e_hex(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], delta)

    def test_match_single_sqeuclidean_rgb(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (12, 3), dtype=np.uint8)
        rgbs_b = np.random.randint(0, 256, (24, 3), dtype=np.uint8)
        p_mat = get_sqeuclidean_matrix(rgbs_a, rgbs_b)
        for i, aaa in enumerate(rgbs_a):
            for j, bbb in enumerate(rgbs_b):
                sqeuclidean = get_sqeuclidean(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], sqeuclidean)

    def test_match_single_sqeuclidean_hex(self):
        """Match result of single conversion mapped over array."""
        hexs_a = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], 12
        )
        hexs_b = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], 24
        )
        p_mat = get_sqeuclidean_matrix_hex(hexs_a, hexs_b)
        for i, aaa in enumerate(hexs_a):
            for j, bbb in enumerate(hexs_b):
                sqeuclidean = get_sqeuclidean_hex(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], sqeuclidean)

    def test_match_single_euclidean_rgb(self):
        """Match result of single conversion mapped over array."""
        rgbs_a = np.random.randint(0, 256, (12, 3), dtype=np.uint8)
        rgbs_b = np.random.randint(0, 256, (24, 3), dtype=np.uint8)
        p_mat = get_euclidean_matrix(rgbs_a, rgbs_b)
        for i, aaa in enumerate(rgbs_a):
            for j, bbb in enumerate(rgbs_b):
                euclidean = get_euclidean(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], euclidean)

    def test_match_single_euclidean_hex(self):
        """Match result of single conversion mapped over array."""
        hexs_a = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], 12
        )
        hexs_b = np.random.choice(
            ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"], 24
        )
        p_mat = get_euclidean_matrix_hex(hexs_a, hexs_b)
        for i, aaa in enumerate(hexs_a):
            for j, bbb in enumerate(hexs_b):
                euclidean = get_euclidean_hex(aaa, bbb)
                np.testing.assert_almost_equal(p_mat[i, j], euclidean)
