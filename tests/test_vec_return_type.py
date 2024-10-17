"""Test that functions returning vectors from ArrayLike args are typed correctly.

The Python type system will not recognize when a function returns
`npt.NDArray[np.int64]` when hinted to return `npt.NDArray[np.float64]`. This module
checks that the functions in vec_conversion.py return the correct types.

:author: Shay Hill
:created: 2024-10-16
"""

import numpy as np

from basic_colormath import vec_conversion as vc
from basic_colormath import vec_distance as vd


class TestRgbsToHsv:
    def test_uint8_input(self):
        not_floats = np.array([[255, 0, 0], [0, 1, 0]], dtype=np.uint8)
        result = vc.rgbs_to_hsv(not_floats)
        assert result.dtype == np.float64

    def test_list_input(self):
        not_floats = [[255, 0, 0], [0, 1, 0]]
        result = vc.rgbs_to_hsv(not_floats)
        assert result.dtype == np.float64


class TestHsvsToRgb:
    def test_int64_input(self):
        not_floats = np.array([[0, 1, 1], [0, 0, 0]], dtype=np.int64)
        result = vc.hsvs_to_rgb(not_floats)
        assert result.dtype == np.float64

    def test_list_input(self):
        not_floats = [[0, 1, 1], [0, 0, 0]]
        result = vc.hsvs_to_rgb(not_floats)
        assert result.dtype == np.float64


class TestRgbsToHsl:
    def test_uint8_input(self):
        not_floats = np.array([[255, 0, 0], [0, 1, 0]], dtype=np.uint8)
        result = vc.rgbs_to_hsl(not_floats)
        assert result.dtype == np.float64

    def test_list_input(self):
        not_floats = [[255, 0, 0], [0, 1, 0]]
        result = vc.rgbs_to_hsl(not_floats)
        assert result.dtype == np.float64


class TestHslsToRgb:
    def test_int64_input(self):
        not_floats = np.array([[0, 1, 1], [0, 0, 0]], dtype=np.int64)
        result = vc.hsls_to_rgb(not_floats)
        assert result.dtype == np.float64

    def test_list_input(self):
        not_floats = [[0, 1, 1], [0, 0, 0]]
        result = vc.hsls_to_rgb(not_floats)
        assert result.dtype == np.float64


class TestFloatsToUint8:
    def test_float64_input(self):
        not_uint8 = np.array([0.0, 1.0, 0.5], dtype=np.float64)
        result = vc.floats_to_uint8(not_uint8)
        assert result.dtype == np.uint8

    def test_list_input(self):
        not_uint8 = [0.0, 1.0, 0.5]
        result = vc.floats_to_uint8(not_uint8)
        assert result.dtype == np.uint8


class TestRgbsToHex:
    def test_uint8_input(self):
        not_str = np.array([[255, 0, 0], [0, 1, 0]], dtype=np.uint8)
        result = vc.rgbs_to_hex(not_str)
        assert result.dtype == "<U7"

    def test_list_input(self):
        not_str = [[255, 0, 0], [0, 1, 0]]
        result = vc.rgbs_to_hex(not_str)
        assert result.dtype == "<U7"


class TestHexToRgb:
    def test_str_input(self):
        not_uint8 = np.array(["#ff0000", "#00ff00"], dtype="<U7")
        result = vc.hexs_to_rgb(not_uint8)
        assert result.dtype == np.uint8

    def test_list_input(self):
        not_uint8 = [["#ff0000"], ["#00ff00"]]
        result = vc.hexs_to_rgb(not_uint8)
        assert result.dtype == np.uint8


class TestRgbsToLab:
    def test_uint8_input(self):
        not_floats = np.array([[255, 0, 0], [0, 1, 0]], dtype=np.uint8)
        result = vd.rgbs_to_lab(not_floats)
        assert result.dtype == np.float64

    def test_list_input(self):
        not_floats = [[255, 0, 0], [0, 1, 0]]
        result = vd.rgbs_to_lab(not_floats)
        assert result.dtype == np.float64


class TestHexsToLab:
    def test_str_input(self):
        not_uint8 = np.array(["#ff0000", "#00ff00"], dtype="<U7")
        result = vd.hexs_to_lab(not_uint8)
        assert result.dtype == np.float64

    def test_list_input(self):
        not_uint8 = [["#ff0000"], ["#00ff00"]]
        result = vd.hexs_to_lab(not_uint8)
        assert result.dtype == np.float64


class TestGetSqeuclineans:
    def test_float64_input(self):
        not_floats_a = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        not_floats_b = np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        result = vd.get_sqeuclideans(not_floats_a, not_floats_b)
        assert result.dtype == np.float64

    def test_list_input(self):
        not_floats_a = [[0, 0, 0], [255, 255, 255]]
        not_floats_b = [[0, 255, 0], [255, 0, 255]]
        result = vd.get_sqeuclideans(not_floats_a, not_floats_b)
        assert result.dtype == np.float64

class TestGetSqeuclideansHex:
    def test_str_input(self):
        not_uint8_a = np.array(["#000000", "#ffffff"], dtype="<U7")
        not_uint8_b = np.array(["#00ff00", "#ff00ff"], dtype="<U7")
        result = vd.get_sqeuclideans_hex(not_uint8_a, not_uint8_b)
        assert result.dtype == np.float64

    def test_list_input(self):
        not_uint8_a = [["#000000"], ["#ffffff"]]
        not_uint8_b = [["#00ff00"], ["#ff00ff"]]
        result = vd.get_sqeuclideans_hex(not_uint8_a, not_uint8_b)
        assert result.dtype == np.float64

class TestGetDeltasELab:
    def test_float64_input(self):
        arg_a = np.array([[100, 0, 0], [40, 0, 20]], dtype=np.float64)
        arg_b = np.array([[0, 10, 0], [100, -19, 13]], dtype=np.float64)
        result = vd.get_deltas_e_lab(arg_a, arg_b)
        assert result.dtype == np.float64

    def test_list_input(self):
        not_floats_a = [[100, 0, 0], [40, 0, 20]]
        not_floats_b = [[0, 10, 0], [100, -19, 13]]
        result = vd.get_deltas_e_lab(not_floats_a, not_floats_b)
        assert result.dtype == np.float64

class GetDeltaEMatrixLab:
    def test_float64_input(self):
        arg_a = np.array([[100, 0, 0], [40, 0, 20]], dtype=np.float64)
        arg_b = np.array([[0, 10, 0], [100, -19, 13]], dtype=np.float64)
        result = vd.get_delta_e_matrix_lab(arg_a, arg_b)
        assert result.dtype == np.float64

    def test_list_input(self):
        not_floats_a = [[100, 0, 0], [40, 0, 20]]
        not_floats_b = [[0, 10, 0], [100, -19, 13]]
        result = vd.get_delta_e_matrix_lab(not_floats_a, not_floats_b)
        assert result.dtype == np.float64
