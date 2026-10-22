"""AOI geometry and demand generation; application semantics are a later stage."""

__version__ = "0.1.3"

from .geometry import Aoi, Grid, TileRect
from .generate import GenerationConfig, generate

__all__ = ["Aoi", "Grid", "TileRect", "GenerationConfig", "generate"]
