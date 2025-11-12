# basic_colormath

Everything I wanted to salvage from the [python-colormath](https://github.com/gtaylor/python-colormath/tree/master) library ... with no numpy dependency and 14x speed.

* Perceptual (DeltaE CIE 2000) and Euclidean distance between colors
* Conversion between RGB, HSV, HSL, Lab, and 8-bit hex colors
* Some convenience functions for RGB tuples and 8-bit hex color strings
* Vectorized functions for numpy arrays
* Proximity and cross-proximity (rectangular) matrices for numpy arrays

Lab color format is exciting because it can cover a larger colorspace than RGB. But don't get *too* excited yet. If you convert an RGB tuple to Lab *with no additional information*, then the result will—of course—*not* contain more information than the RGB tuple you converted from. Other parameters are necessary to get anything out of these elaborate formats. I don't know how to do that, and most likely neither do you, so why not drop all of that complexity?

I've installed [python-colormath](https://github.com/gtaylor/python-colormath/tree/master) on a lot of projects. The library does many interesting things, but most of what I wanted was perceptual color distance. This requires Lab colors, which allow more parameters than R, G, and B. **Colormath didn't use those parameters**, so the result didn't require the elaborate classes and methods provided by Colormath.

The color distance I provide here is DeltaE CIE 2000. Aside from (presumably) some specialized applications, this is the best of the multiple color distances provided by [python-colormath](https://github.com/gtaylor/python-colormath/tree/master). Tuples in, floats out, and with a lot more speed. It doesn't use all of those expert parameters, **but neither did Colormath**. This is the same result you'll get from any of the online DeltaE calculators you're likely to find.

This library is more or less specialized for working with "upscaled" RGB tuples `([0, 255], [0, 255], [0, 255])`. Functions will take floats or ints in that range and return floats. If you want ints, use `float_tuple_to_8bit_int_tuple`. This is dramatically better int conversion than `int(float)` or `int(round(float))`, so use it instead of those.

## distance functions

```python
Rgb = Annotated[tuple[float, float, float], ([0, 255], [0, 255], [0, 255])]
Hsv = Annotated[tuple[float, float, float], ([0, 365), [0, 100], [0, 100])]
Hsl = Annotated[tuple[float, float, float], ([0, 365), [0, 100], [0, 100])]
Lab = Annotated[tuple[float, float, float], ([0, 100], [-128, 127], [-128, 127])]
Hex = Annotated[str, "#000fff"]

rgb_to_lab(rgb: Rgb) -> Lab:
    # Converts RGB to Lab. To optionally cache for get_delta_e_lab

hex_to_lab(hex: Hex) -> Lab:
    # Converts hex to Lab. To optionally cache for get_delta_e_lab

lab_to_rgb(lab: Lab) -> Rgb:
    # Converts Lab to RGB. Does not check for out-of-gamut colors.

get_delta_e(rgb_a: Rgb, rgb_b: Rgb) -> float:
    # Calculate the Delta E (CIE 2000) between two RGB colors.
    # This is the one you'll usually want.

get_delta_e_hex(hex_a: Hex, hex_b: Hex) -> float:
    # Calculate the Delta E (CIE 2000) between two hex colors.
    # Takes hex colorstrings.

get_delta_e_lab(lab_a: Lab, lab_b: Lab) -> float:
    # Calculate the Delta E (CIE2000) between two Lab colors.
    # To call with cached Lab values.

get_sqeuclidean(rgb_a: Rgb, rgb_b: Rgb) -> float:
    # Calculate the squared Euclidean distance between two RGB colors.

get_sqeuclidean_hex(hex_a: Hex, hex_b: Hex) -> float:
    # Calculate the squared Euclidean distance between two HEX colors.

get_euclidean(rgb_a: Rgb, rgb_b: Rgb) -> float:
    # Calculate the Euclidean distance between two RGB colors.

get_euclidean_hex(hex_a: Hex, hex_b: Hex) -> float:
    # Calculate the Euclidean distance between two HEX colors.
```

## other conversions

Converts to other simple formats.

```python
def rgb_to_hsv(rgb: Rgb) -> Hsv: ...

def hsv_to_rgb(hsv: Hsv) -> Rgb: ...

def rgb_to_hsl(rgb: Rgb) -> Hsl: ...

def hsl_to_rgb(hsl: Hsl) -> Rgb: ...

def rgb_to_hex(rgb: Rgb) -> Hex: ...

def hex_to_rgb(hex_: Hex) -> Rgb: ...
```

### basic_colormath vs python.colorsys

The `colorsys` module in the Python standard library also provides functions for rgb to hsv to hls (*not* hsl) conversion. These functions use the same math but take and return floats in different ranges. Colorsys assumes you are working with floats in the range [0, 1] for rgb, hsv, and hls values. Basic_colormath assumes you are working with floats in the range [0, 255] for rgb values and [0, 365], [0, 100], [0, 100] for hsv and hsl values. Which is better for you will depend on where you are getting your data and where you will put the results.

## convenience functions

```python
_Ratio = float | tuple[float, ...] | None

scale_rgb(rgb: Rgb, scalar: float) -> Rgb:
    # Scale an rgb tuple by a scalar.

mix_rgb(*rgb_args: Rgb, ratio: _Ratio=None) -> Rgb:
    """ Mix any number of rgb tuples.

    :param rgb_args: rgb tuples ([0, 255], [0, 255], [0, 255])
    :param ratio: 0.0 to 1.0 for the weight of the first rgb_arg or a tuple of floats
        to distribute across rgb_args or None for equal ratios. Ratios will be
        normalized and (if fewer ratios than colors are provided) the remaining
        ratios will be equal.
    :return: rgb tuple ([0, 255], [0, 255], [0, 255])
    """

scale_hex(hex_: Hex, scalar: float)-> Hex
    # Scale a hex color by a scalar.

mix_hex(*hex_args: Hex, ratio: _Ratio=None) -> Hex
    # Mix any number of hex colors.
```

## better float to int conversion

```python
def float_to_8bit_int(float_: float | int) -> int: ...

def float_tuple_to_8bit_int_tuple(rgb: Rgb) -> tuple[int, int, int]: ...
```

## vectorized functions

If you have numpy installed in your Python environment, basic_colormath will provide vectorized versions of most functions along with proximity matrices and cross-proximity matrices.

| Function                      | Vectorized Function           | (Cross-) Proximity Matrix  |
| ----------------------------- | ----------------------------- | -------------------------- |
| float_to_8bit_int             | floats_to_uint8               |                            |
| get_delta_e                   | get_deltas_e                  | get_delta_e_matrix         |
| get_delta_e_hex               | get_deltas_e_hex              | get_delta_e_matrix_hex     |
| get_delta_e_lab               | get_deltas_e_lab              | get_delta_e_matrix_lab     |
| get_euclidean                 | get_euclideans                | get_euclidean_matrix       |
| get_euclidean_hex             | get_euclideans_hex            | get_euclidean_matrix_hex   |
| get_sqeuclidean               | get_sqeuclideans              | get_squeclidean_matrix     |
| get_sqeuclidean_hex           | get_sqeuclideans_hex          | get_sqeuclinean_matrix_hex |
| hex_to_rgb                    | hexs_to_rgb                   |                            |
| hsl_to_rgb                    | hsls_to_rgb                   |                            |
| hsv_to_rgb                    | hsvs_to_rgb                   |                            |
| rgb_to_hex                    | rgbs_to_hex                   |                            |
| rgb_to_hsl                    | rgbs_to_hsl                   |                            |
| rgb_to_hsv                    | rgbs_to_hsv                   |                            |
| rgb_to_lab                    | rgbs_to_lab                   |                            |
| lab_to_rgb                    | labs_to_rgb                   |                            |
| mix_hex                       |                               |                            |
| mix_rgb                       |                               |                            |
| scale_hex                     |                               |                            |
| scale_rgb                     |                               |                            |

## proximity matrices

(Cross-)proximity matrix functions take a (-1, 3) array of color values or (-1,) array of hex color strings and return a proximity matrix. This is a square matrix with the same number of rows and columns as the number of colors provided. The value at row `i` and column `j` is the distance between the color at index `i` and the color at index `j`. The diagonal is always zero.

An optional second argument creates a *cross-proximity* matrix. This is a matrix with the same number of rows as the first argument and the same number of columns as the second argument. The value at row `i` and column `j` is the distance between the color at index `i` in the first argument and the color at index `j` in the second argument.

## If you need more

Sadly, [python-colormath](https://github.com/gtaylor/python-colormath/tree/master) has been abandoned, long enough now that a numpy function on which it relies has been not only deprecated but removed. If you still need to use [python-colormath](https://github.com/gtaylor/python-colormath/tree/master), patch `np.asscalar`:

```python
import numpy as np
import numpy.typing as npt

def _patch_asscalar(a: npt.NDArray[np.float64]) -> float:
    """Alias for np.item(). Patch np.asscalar for colormath.

    :param a: numpy array
    :return: input array as scalar
    """
    return a.item()

np.asscalar = _patch_asscalar  # type: ignore
```
