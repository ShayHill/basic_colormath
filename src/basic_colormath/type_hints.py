"""Type hints for color tuples.

:author: Shay Hill
:created: 2023-04-30

Type annotations lose Python 3.8 compatibility. Not enough justification IMO, but the
information is useful.

Rgb = Annotated[tuple[float, float, float], ([0, 255], [0, 255], [0, 255])]
Hsv = Annotated[tuple[float, float, float], ([0, 365), [0, 100], [0, 100])]
Hsl = Annotated[tuple[float, float, float], ([0, 365), [0, 100], [0, 100])]
Lab = Annotated[tuple[float, float, float], ([0, 100], [-128, 127], [-128, 127])]
Hex = Annotated[str, "#000fff"]
"""

from collections.abc import Iterable

Rgb = tuple[float, float, float]
Hsv = tuple[float, float, float]
Hsl = tuple[float, float, float]
Lab = tuple[float, float, float]
Hex = str

RgbLike = Rgb | Iterable[float]
HsvLike = Hsv | Iterable[float]
HslLike = Hsl | Iterable[float]
LabLike = Lab | Iterable[float]
