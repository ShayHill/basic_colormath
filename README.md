# basic_colormath

Everything I wanted to salvage from the [python-colormath](https://github.com/gtaylor/python-colormath/tree/master) library ... with no numpy deps and 14x speed.

* Perceptual (DeltaE CIE 2000) and Euclidean distance between colors
* Conversion between RGB, HSV, HSL, and 8-bit hex colors
* Simple, one-way conversion to Lab
* Some convenience functions for RGB tuples and 8-bit hex color strings

Lab and LCh color formats are exciting because they can cover a larger colorspace than RGB. But don't get *too* excited yet. If you convert an RGB tuple to Lab or LCh *with no additional information*, then the result will—of course—not* contain more information than the RGB tuple you converted from. Other parameters are necessary to get anything out of these elaborate formats. I don't know how to do that, and most likely neither do you, so why not drop all of that complexity?

I've installed [python-colormath](https://github.com/gtaylor/python-colormath/tree/master) on a lot of projects. The library does many interesting things, but most of what I wanted was perceptual color distance. This requires Lab colors, which have more parameters than an RGB tuple provides. **Colormath didn't use those parameters**, so the result didn't require the elaborate classes and methods provided by Colormath.

The color distance I provide here is DeltaE CIE 2000. Aside from (presumably) some specialized applications, this is the best of the multiple color distances provided by [python-colormath](https://github.com/gtaylor/python-colormath/tree/master). Tuples in, float out, and with a lot more speed. It doesn't use all of those expert parameters, **but neither did Colormath**. This is the same result you'll get from any of the online DeltaE calculators you're likely to find.

This library is more or less specialized for working with "upscaled" RGB tuples `([0, 255], [0, 255], [0, 255])`. Functions will take floats or ints in that range and return floats. If you want ints, use `float_tuple_to_8bit_int_tuple`. This is dramatically better int conversion than `int(float)` or `int(round(float))`, so use it insead of those.

## don't miss

CIE 2000 is *not* cummutative. That is, `get_delta_e(a, b)` is not the same as `get_delta_e(b, a)`. If this is important to you, you'll need to calculate both and take the min, max, or average.

## distance functions

    RGB = Annotated[tuple[float, float, float], ([0, 255], [0, 255], [0, 255])]
    HSV = Annotated[tuple[float, float, float], ([0, 365), [0, 100], [0, 100])]
    HSL = Annotated[tuple[float, float, float], ([0, 365), [0, 100], [0, 100])]
    Lab = Annotated[tuple[float, float, float], ([0, 100], [-128, 127], [-128, 127])]
    Hex = Annotated[str, "#000fff"]

    rgb_to_lab(rgb: RGB) -> Lab:
        # Converts RGB to Lab. To optionally cache for get_delta_e_lab

    hex_to_lab(hex: Hex) -> Lab:
        # Converts hex to Lab. To optionally cache for get_delta_e_lab

    get_delta_e(rgb_a: RGB, rgb_b: RGB) -> float:
        # Calculate the Delta E (CIE 2000) between two RGB colors.
        # This is the one you'll usually want.

    get_delta_e_hex(hex_a: Hex, hex_b: Hex) -> float:
        # Calculate the Delta E (CIE 2000) between two hex colors.
        # Takes and returns hex colorstrings.

    get_delta_e_lab(lab_a: Lab, lab_b: Lab) -> float:
        # Calculate the Delta E (CIE2000) of two Lab colors.
        # To call with cached Lab values.

    get_sqeuclidean(rgb_a: RGB, rgb_b: RGB) -> float:
        # Calculate the squared Euclidean distance between two RGB colors.

    get_sqeuclidean_hex(hex_a: Hex, hex_b: Hex) -> float:
        # Calculate the squared Euclidean distance between two HEX colors.

    get_euclidean(rgb_a: RGB, rgb_b: RGB) -> float:
        # Calculate the Euclidean distance between two RGB colors.

    get_euclidean_hex(hex_a: Hex, hex_b: Hex) -> float:
        # Calculate the Euclidean distance between two HEX colors.

## other conversions

Converts to other simple formats.

    rgb_to_hsv(rgb: RGB) -> HSV

    hsv_to_rgb(hsv: HSV) -> RGB

    rgb_to_hsl(rgb: RGB) -> HSL

    hsl_to_rgb(hsl: HSL) -> RGB

    rgb_to_hex(rgb: RGB) -> Hex

    hex_to_rgb(hex_: Hex) -> RGB

## convenience functions

    _Ratio = float | tuple[float, ...] | None

    scale_rgb(rgb: RGB, scalar: float) -> RGB:
        # Scale an rgb tuple by a scalar.

    mix_rgb(*rgb_args: RGB, ratio: _Ratio=None) -> RGB:
        # Mix any number of rgb tuples.

        :param rgb_args: rgb tuples ([0, 255], [0, 255], [0, 255])
        :param ratio: 0.0 to 1.0 for the weight of the first rgb_arg or a tuple of floats
            to distribute across rgb_args or None for equal ratios. Ratios will be
            normalized and (if fewer ratios than colors are provided) the remaining
            ratios will be equal.
        :return: rgb tuple ([0, 255], [0, 255], [0, 255])

    scale_hex(hex_: Hex, scalar: float)-> Hex

    mix_hex(*hex_args: Hex, ratio: _Ratio=None) -> Hex

## better float to int conversion

    float_to_8bit_int(float_: float | int) -> int:

    float_tuple_to_8bit_int_tuple(rgb: RGB) -> tuple[int, int, int]:


## If you need more

Sadly, [python-colormath](https://github.com/gtaylor/python-colormath/tree/master) has been abandoned, long enough now that a numpy function on which it relies has been not only deprecated but removed. If you still need to use [python-colormath](https://github.com/gtaylor/python-colormath/tree/master), patch `np.asscalar`:

    import numpy as np
    import numpy.typing as npt

    def _patch_asscalar(a: npt.NDArray[np.float_]) -> float:
        """Alias for np.item(). Patch np.asscalar for colormath.

        :param a: numpy array
        :return: input array as scalar
        """
        return a.item()

    np.asscalar = _patch_asscalar  # type: ignore
