vim9script

# RGB [0..255], [0..255], [0.255]
# XYZ [0.0, 1.0286], [0.0, 1.0822], [0.0, 1.178]
# Lab [0.0, 100], [-86.183, 98.235], [-107.865, 94.477]

const RGB_TO_XYZ = [
    [0.412424, 0.357579, 0.180464],
    [0.212656, 0.715158, 0.0721856],
    [0.0193324, 0.119193, 0.950444],
]

def RgbToXyz(rgb: list<number>): list<float>
    # RGB to XYZ conversion. Expects RGB values between 0 and 255.
    #
    # Inputs:
    #   rgb - RGB values between 0 and 255 inclusive.
    # Returns:
    #   - XYZ values between 0 and 1 inclusive.
    #
    # The standard rgb to xyz conversion scaled for [0, 255] color values.

    var lin = []
    for channel in rgb
        if channel <= 10.31475
            call add(lin, channel / 3294.6)
        else
            call add(lin, pow((channel + 14.025) / 269.025, 2.4))
        endif
    endfor

    var result_matrix: list<float> = []
    var m: float
    for row in RGB_TO_XYZ
        m = 0.0
        for [i, r] in items(row)
            m += r * lin[i]
        endfor
        call add(result_matrix, m)
    endfor

    return map(result_matrix, (_, v) => v > 0.0 ? v : 0.0)
enddef


const CIE_E = 216.0 / 24389
const FRAC_1_3 = 1.0 / 3
const FRAC_16_116 = 16.0 / 116

# This will always be the illuminant when rgb is converted to xyz from an
# without an illuminant argument. This (without an illuminant argument) is the
# way XYZ is converted to Lab in every library and online calculator I've
# found.
const XYZ_ILLUM = [0.95047, 1.0, 1.08883]

def XyzToLab(xyz: list<float>): list<float>
    # XYZ to Lab conversion. Expects XYZ values between 0 and 1.
    #
    # Inputs:
    #   xyz - XYZ values between 0 and 1 inclusive.
    # Returns:
    #   - Lab values. Lab values are canonically in the space [0..100] for L,
    #   [-127..127] for a and [-127..127] for b. Lab values are canonically in
    #   the space [0..1], [0..1], [0..1]. There are xyz values in that range
    #   that will produce Lab values outside of the canonical range. This
    #   should not happen with any XYZ values derived from [0..255] rgb
    #   values, but I haven't tested all 16 million of them.

    # var scaled_xyz: list<float> = []
    # var c: float
    # for [i, illum] in items(XYZ_ILLUM)
    #     c = xyz[i]
    #     add(scaled_xyz, c / illum)
    # endfor

    var scaled_xyz = map(copy(xyz), (i, v) => v / XYZ_ILLUM[i])

    for [i, channel] in items(scaled_xyz)
        if channel > CIE_E
            scaled_xyz[i] = pow(channel, FRAC_1_3)
        else
            scaled_xyz[i] = (7.787 * channel) + FRAC_16_116
        endif
    endfor

    var [x, y, z] = scaled_xyz
    var lab_l = (116 * y) - 16.0
    var lab_a = 500 * (x - y)
    var lab_b = 200 * (y - z)

    return [lab_l, lab_a, lab_b]
enddef


def RgbToLab(rgb: list<number>): list<float>
    # RGB to Lab conversion. Expects RGB values between 0 and 255.
    #
    # Inputs:
    #   rgb - RGB values between 0 and 255 inclusive.
    # Returns:
    #   - Lab values [0..100], [-127, 127], [-127, 127]
    #
    return XyzToLab(RgbToXyz(rgb))
enddef


def HexToRgb(hex_color: string): list<number>
    # convert a color in hex notation (e.g., '#ffffff') to three 8-bit
    # integers
    var red = str2nr('0x' .. strpart(hex_color, 1, 2), 16)
    var green = str2nr('0x' .. strpart(hex_color, 3, 2), 16)
    var blue = str2nr('0x' .. strpart(hex_color, 5, 2), 16)
    return [red, green, blue]
enddef

const RAD_6 = 6 * acos(-1) / 180
const RAD_25 = 25 * acos(-1) / 180
const RAD_30 = 30 * acos(-1) / 180
const RAD_63 = 63 * acos(-1) / 180
const RAD_180 = 180 * acos(-1) / 180
const RAD_275 = 275 * acos(-1) / 180
const RAD_360 = 360 * acos(-1) / 180
const RAD_720 = 720 * acos(-1) / 180
const V25_E7 = pow(25, 7)

def HexToLab(hex_color: string): list<float>
    # Convert hex color to Lab.
    rgb = HexToRgb(hex_color)
    return RgbToLab(rgb)
enddef


def DeltaELab(lab_a: list<float>, lab_b: list<float>): float
    # Calculate the Delta E (CIE2000) of two Lab colors.
    # lab_a: The first Lab color.
    # lab_b: The second Lab color.

    var lab_a_bsq = pow(lab_a[2], 2)
    var lab_b_bsq = pow(lab_b[2], 2)

    var Lp = (lab_a[0] + lab_b[0]) / 2.0

    var C1 = pow(pow(lab_a[1], 2) + lab_a_bsq, 0.5)
    var C2 = pow(pow(lab_b[1], 2) + lab_b_bsq, 0.5)
    var avg_c_e7 = pow((C1 + C2) / 2.0, 7)
    var G = 0.5 * (1 - pow(avg_c_e7 / (avg_c_e7 + V25_E7), 0.5)) + 1

    var [a1p, a2p] = map([lab_a, lab_b], (_, v) => v[1] * G)

    var C1p = pow(pow(a1p, 2) + lab_a_bsq, 0.5)
    var C2p = pow(pow(a2p, 2) + lab_b_bsq, 0.5)
    var Cp = (C1p + C2p) / 2.0

    var h1p = atan2(lab_a[2], a1p)
    h1p = h1p >= 0 ? h1p : h1p + RAD_360
    var h2p = atan2(lab_b[2], a2p)
    h2p = h2p >= 0 ? h2p : h2p + RAD_360
    var Hp = (h1p + h2p) / 2
    Hp = abs(h1p - h2p) <= RAD_180 ? Hp : Hp + RAD_180

    var T = (1 - 0.17 * cos(Hp - RAD_30)
             + 0.24 * cos(2 * Hp)
             + 0.32 * cos(3 * Hp + RAD_6)
             - 0.2 * cos(4 * Hp - RAD_63)
    )

    var delta_hp = h2p - h1p
    if abs(delta_hp) > RAD_180
        delta_hp = h2p > h1p ? delta_hp - RAD_360 : delta_hp + RAD_360
    endif

    var delta_Lp = lab_b[0] - lab_a[0]
    var delta_Cp = C2p - C1p
    var delta_Hp = 2 * pow(C2p * C1p, 0.5) * sin(delta_hp / 2)

    var lp_minus_50_sq = pow(Lp - 50, 2)
    var S_L = 1 + (0.015 * lp_minus_50_sq) / pow(20 + lp_minus_50_sq, 0.5)
    var S_C = 1 + 0.045 * Cp
    var S_H = 1 + 0.015 * Cp * T

    var delta_ro = RAD_30 * exp(-(pow((Hp - RAD_275) / RAD_25, 2)))

    var avg_cp_e7 = pow(Cp, 7)
    var R_C = pow(avg_cp_e7 / (avg_cp_e7 + V25_E7), 0.5)
    var R_T = -2 * R_C * sin(2 * delta_ro)

    return pow(
        pow(delta_Lp / S_L, 2)
        + pow(delta_Cp / S_C, 2)
        + pow(delta_Hp / S_H, 2)
        + R_T * delta_Cp / S_C * delta_Hp / S_H,
        0.5)
enddef

g:aaa = RgbToLab([0, 0, 0])
g:bbb = RgbToLab([25, 25, 25])
g:ccc = DeltaELab(g:aaa, g:bbb)




# (Pdb) _xyz_to_lab((1, .2, .3))
# (51.837211526538496, 216.13675985129578, -13.181163974878519)


# if assert_equal(RgbToXyz([1, 2, 3]), [0.0005065786438414376, 0.0005644171674861896, 0.0009436806896133066]) | throw v:errors[-1] | endif

# (Pdb) _rgb_to_xyz((1, 2, 3))
# (0.0005065786438414376, 0.0005644171674861896, 0.0009436806896133066)
# (Pdb) _rgb_to_xyz((3, 2, 1))
# (0.0006473908820494143, 0.0006496902810659868, 0.00037844569902264314)
# (Pdb) _rgb_to_xyz((255, 255, 255))
# (0.950467, 0.9999995999999999, 1.0889693999999999)
# (Pdb) _rgb_to_xyz((0, 0, 0))
# (0.0, 0.0, 0.0)
#  
