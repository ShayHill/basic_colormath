"""Test Mixbox-powered color mixing.

:author: Shay Hill
:created: 2026-10-10
"""

import pytest

from basic_colormath.mb_mixer import mixbox_hex, mixbox_rgb


class TestMixboxRGB:
    def test_equal_ratios(self) -> None:
        assert mixbox_rgb((255, 0, 0), (0, 0, 255)) == (113, 1, 105)

    @pytest.mark.parametrize(
        ("ratio", "expected"), [(0, (0, 0, 255)), (1, (255, 0, 0))]
    )
    def test_boundary_ratio_preserves_selected_color(
        self, ratio: float, expected: tuple[int, int, int]
    ) -> None:
        assert mixbox_rgb((255, 0, 0), (0, 0, 255), ratio=ratio) == expected

    def test_float_ratio_weights_first_color(self) -> None:
        assert mixbox_rgb((255, 0, 0), (0, 0, 255), ratio=0.75) == (175, 9, 46)

    def test_tuple_ratio_weights_multiple_colors(self) -> None:
        result = mixbox_rgb((255, 0, 0), (0, 255, 0), (0, 0, 255), ratio=(0.5, 0.25))

        assert result == (125, 61, 49)

    def test_invalid_ratio_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="ratios must be >= 0"):
            _ = mixbox_rgb((255, 0, 0), (0, 0, 255), ratio=-0.1)


class TestMixboxHex:
    def test_matches_rgb_mixing_and_accepts_hashless_hex(self) -> None:
        assert mixbox_hex("ff0000", "#0000ff", ratio=0.75) == "#af092e"
