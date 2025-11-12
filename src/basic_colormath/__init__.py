"""Raise public functions into the package namespace.

:author: Shay Hill
:created: 2023-04-30
"""

# Numpy is an optional dependency. If it's not installed, the numpy functions will
# not be available.
try:
    import numpy
except ImportError:
    numpy = None


from basic_colormath.conversion import (
    float_to_8bit_int,
    float_tuple_to_8bit_int_tuple,
    hex_to_rgb,
    hsl_to_rgb,
    hsv_to_rgb,
    rgb_to_hex,
    rgb_to_hsl,
    rgb_to_hsv,
)
from basic_colormath.distance import (
    get_delta_e,
    get_delta_e_hex,
    get_delta_e_lab,
    get_euclidean,
    get_euclidean_hex,
    get_sqeuclidean,
    get_sqeuclidean_hex,
    lab_to_rgb,
    rgb_to_lab,
)
from basic_colormath.mix import mix_hex, mix_rgb, scale_hex, scale_rgb

__all__ = [
    "float_to_8bit_int",
    "float_tuple_to_8bit_int_tuple",
    "get_delta_e",
    "get_delta_e_hex",
    "get_delta_e_lab",
    "get_euclidean",
    "get_euclidean_hex",
    "get_sqeuclidean",
    "get_sqeuclidean_hex",
    "hex_to_rgb",
    "hsl_to_rgb",
    "hsv_to_rgb",
    "lab_to_rgb",
    "mix_hex",
    "mix_rgb",
    "rgb_to_hex",
    "rgb_to_hsl",
    "rgb_to_hsv",
    "rgb_to_lab",
    "scale_hex",
    "scale_rgb",
]

if numpy:
    from basic_colormath.vec_conversion import (
        floats_to_uint8,
        hexs_to_rgb,
        hsls_to_rgb,
        hsvs_to_rgb,
        rgbs_to_hex,
        rgbs_to_hsl,
        rgbs_to_hsv,
    )
    from basic_colormath.vec_distance import (
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
        labs_to_rgb,
        rgbs_to_lab,
    )

    __all__ += [
        "floats_to_uint8",
        "get_delta_e_matrix",
        "get_delta_e_matrix_hex",
        "get_delta_e_matrix_lab",
        "get_deltas_e",
        "get_deltas_e_hex",
        "get_deltas_e_lab",
        "get_euclidean_matrix",
        "get_euclidean_matrix_hex",
        "get_euclideans",
        "get_euclideans_hex",
        "get_sqeuclidean_matrix",
        "get_sqeuclidean_matrix_hex",
        "get_sqeuclideans",
        "get_sqeuclideans_hex",
        "hexs_to_rgb",
        "hsls_to_rgb",
        "hsvs_to_rgb",
        "labs_to_rgb",
        "rgbs_to_hex",
        "rgbs_to_hsl",
        "rgbs_to_hsv",
        "rgbs_to_lab",
    ]
