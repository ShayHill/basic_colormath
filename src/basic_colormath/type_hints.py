"""Type hints for color tuples.

:author: Shay Hill
:created: 2023-04-30

Type annotations lose Python 3.8 compatibility. Not enough justification IMO, but the
information is useful.

RGB = Annotated[tuple[float, float, float], ([0, 255], [0, 255], [0, 255])]
HSV = Annotated[tuple[float, float, float], ([0, 365), [0, 100], [0, 100])]
HSL = Annotated[tuple[float, float, float], ([0, 365), [0, 100], [0, 100])]
Lab = Annotated[tuple[float, float, float], ([0, 100], [-128, 127], [-128, 127])]
Hex = Annotated[str, "#000fff"]
"""

from typing import Tuple

RGB = Tuple[float, float, float]
HSV = Tuple[float, float, float]
HSL = Tuple[float, float, float]
Lab = Tuple[float, float, float]
Hex = str
