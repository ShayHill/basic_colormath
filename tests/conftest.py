"""Shared fixtures

:author: Shay Hill
:created: 2023-04-30
"""

import random
from typing import Any, List, Tuple

import pytest


def pytest_assertrepr_compare(config: Any, op: str, left: str, right: str):
    """See full error diffs"""
    if op in ("==", "!="):
        return ["{0} {1} {2}".format(left, op, right)]


@pytest.fixture(scope="module", params=range(100))
def rgb_tuple() -> Tuple[int, int, int]:
    """Return a random rgb tuple."""
    red, grn, blu = (random.randint(0, 255) for _ in range(3))
    return red, grn, blu


@pytest.fixture(scope="module", params=range(100))
def rgb_args() -> List[Tuple[int, int, int]]:
    """A random number of rgb tuples."""
    rgbs: List[Tuple[int, int, int]] = []
    for _ in range(random.randint(1, 10)):
        red, grn, blu = tuple(random.randint(0, 255) for _ in range(3))
        rgbs.append((red, grn, blu))
    return rgbs
