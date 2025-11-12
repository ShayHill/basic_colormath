"""See full diffs in pytest.

:author: Shay Hill
:created: 2023-04-30
"""

from __future__ import annotations

import random
from typing import Any

import pytest


def pytest_assertrepr_compare(
    config: Any, op: str, left: str, right: str
) -> list[str] | None:
    """See full error diffs"""
    del config
    if op in ("==", "!="):
        return [f"{left} {op} {right}"]


@pytest.fixture(scope="module", params=range(100))
def rgb_tuple() -> tuple[int, int, int]:
    """Return a random rgb tuple."""
    red, grn, blu = (random.randint(0, 255) for _ in range(3))
    return red, grn, blu


@pytest.fixture(scope="module", params=range(100))
def rgb_args() -> list[tuple[int, int, int]]:
    """A random number of rgb tuples."""
    rgbs: list[tuple[int, int, int]] = []
    for _ in range(random.randint(1, 10)):
        red, grn, blu = tuple(random.randint(0, 255) for _ in range(3))
        rgbs.append((red, grn, blu))
    return rgbs
