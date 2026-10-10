"""Helper functions for the project.

:author: Shay Hill
:created: 2026-10-10
"""


def _split_float(f: float, num: int) -> tuple[float, ...]:
    """Divide a float into num parts.

    :param f: float to divide
    :param num: number of parts
    :return: tuple of floats
    """
    f = max(0, f)
    if num == 0:
        return ()
    return (f / num,) * num


def infer_ps(ratio: float | tuple[float, ...] | None, num: int) -> tuple[float, ...]:
    """Infer p values from a single float or tuple of floats.

    :param ratios: float or tuple of floats
    :param num: number of ratios to return (len of rgb_args)
    :return: tuple of floats summing to 1
    :raise ValueError: if ratios cannot be distributed across values and sum to 1

    Three cases:
    1. ratio is None: return (1/num, ...)
    2. ratio is a float: return (ratios, ((1-ratios) / (num-1), ...)
    3. ratio is a tuple: fill in missing ratios with (1-sum(ratios)) / missing
    """
    # preserve ratio arg for error messages
    ratio_arg = ratio
    if ratio is None:
        return _split_float(1, num)
    if isinstance(ratio, (float, int)):
        ratio = (ratio,)

    if any(r < 0 for r in ratio):
        msg = f"ratios must be >= 0, not {ratio_arg}"
        raise ValueError(msg)
    if len(ratio) > num:
        msg = f"ratios has {len(ratio)} elements, but only <= {num} are allowed"
        raise ValueError(msg)

    sum_ratios = sum(ratio)
    missing = num - len(ratio)
    filled = ratio + _split_float(1 - sum_ratios, missing)
    sum_ratios = sum(filled)
    if sum_ratios <= 0:
        msg = f"ratios must sum to > 0, not {sum_ratios}"
        raise ValueError(msg)
    if sum_ratios == 1:
        return filled
    return tuple(r / sum_ratios for r in filled)
