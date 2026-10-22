"""Seeded location sampling with rectangular or connected mosaic footprints."""

from dataclasses import dataclass
from math import asin, atan2, cos, degrees, hypot, isfinite, log, radians, sin, sqrt

import numpy as np

from .demand import DemandField, checked_keys
from .geometry import Aoi, Grid, TileRect


@dataclass(frozen=True)
class GenerationConfig:
    schema_version: int = 1
    count: int = 10000
    seed: int = 42
    tile_edge_km: float = 0.1
    min_area_km2: float = 1.0
    max_area_km2: float = 5_000_000.0
    min_aspect: float = 0.5
    max_aspect: float = 2.0
    max_bands: int = 6
    hotspot_count: int = 0
    hotspot_spread_km: float = 10.0
    hotspot_fraction: float = 1.0
    shape: str = "mosaic"
    random_rotation: bool = True

    def __post_init__(self):
        if self.shape not in ("rectangle", "mosaic", "oriented_rectangle"):
            raise ValueError("shape must be rectangle, oriented_rectangle or mosaic")
        if type(self.random_rotation) is not bool:
            raise ValueError("random_rotation must be a boolean")
        if self.schema_version != 1:
            raise ValueError("unsupported generation schema")
        for name in ("count", "seed", "max_bands", "hotspot_count"):
            n = getattr(self, name)
            if type(n) is not int or n < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.count < 1 or not 1 <= self.max_bands <= 64:
            raise ValueError("count must be positive; max_bands must be in [1, 64]")
        for name in ("tile_edge_km", "min_area_km2", "max_area_km2", "min_aspect", "max_aspect"):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_area_km2 > self.max_area_km2 or self.min_aspect > self.max_aspect:
            raise ValueError("minimum must not exceed maximum")
        if not isfinite(self.hotspot_spread_km) or self.hotspot_spread_km < 0:
            raise ValueError("hotspot_spread_km must be finite and nonnegative")
        if not isfinite(self.hotspot_fraction) or not 0 <= self.hotspot_fraction <= 1:
            raise ValueError("hotspot_fraction must be in [0, 1]")
        grid = self.grid
        if self.shape != "oriented_rectangle" and self.min_area_km2 < grid.cell_area_km2:
            raise ValueError("minimum AOI area is smaller than one tile; reduce tile_edge_km")
        if self.max_area_km2 > grid.cell_area_km2 * grid.columns * grid.rows / 4:
            raise ValueError("this mosaic generator supports AOIs up to one quarter of Earth's area")

    @property
    def grid(self):
        return Grid.with_tile_edge(self.tile_edge_km)

    @classmethod
    def from_dict(cls, data):
        checked_keys(data, cls.__dataclass_fields__, "generation config")
        return cls(**data)


class LocationSampler:
    def __init__(self, demand: DemandField, config: GenerationConfig, rng, parent_rng):
        self.demand, self.config, self.rng = demand, config, rng
        self.cdf = np.cumsum(demand.probability.ravel())
        self.cdf[-1] = 1.0
        self.parents = [self.direct(parent_rng) for _ in range(config.hotspot_count)]

    def direct(self, rng):
        i = int(np.searchsorted(self.cdf, rng.random(), side="right"))
        row, col = divmod(i, self.demand.grid.columns)
        # Uniform in projected x/y is uniform in ground area, including high latitudes.
        return self.demand.grid.lonlat(col + rng.random(), row + rng.random())

    def sample(self):
        if not self.parents or self.rng.random() >= self.config.hotspot_fraction:
            return (*self.direct(self.rng), None)
        parent = int(self.rng.integers(len(self.parents)))
        lon, lat = self.parents[parent]
        for _ in range(256):
            east, north = self.rng.normal(0, self.config.hotspot_spread_km, size=2)
            angular = hypot(east, north) / self.demand.grid.radius_km
            bearing = atan2(east, north)
            phi = radians(lat)
            dest_lat = asin(max(-1, min(1, sin(phi) * cos(angular)
                                       + cos(phi) * sin(angular) * cos(bearing))))
            dest_lon = radians(lon) + atan2(sin(bearing) * sin(angular) * cos(phi),
                                           cos(angular) - sin(phi) * sin(dest_lat))
            point = ((degrees(dest_lon) + 180) % 360 - 180, degrees(dest_lat))
            x, y = self.demand.grid.cell(*point)
            if self.demand.eligible[y, x]:
                return (*point, parent)
        raise ValueError("hotspot jitter cannot reach eligible cells; reduce hotspot_spread_km")


def rectangle(grid: Grid, lon: float, lat: float, tiles: int, aspect: float):
    """One grid-aligned footprint, split only when it crosses the dateline.

    Round dimensions, not the outline: adding remainder tiles would introduce
    steps. Actual area is recorded separately from the sampled target area.
    """
    anchor_col, anchor_row = grid.cell(lon, lat)
    height = max(1, min(grid.rows, tiles, round(sqrt(tiles / aspect))))
    width = max(1, round(tiles / height))
    if width > grid.columns:
        raise ValueError("AOI too wide for grid; reduce area or aspect ratio")
    row = min(max(0, anchor_row - height // 2), grid.rows - height)
    return tuple(TileRect(anchor_col - width // 2, row, width, height).split_at_seam(grid))


def mosaic(grid: Grid, lon: float, lat: float, tiles: int, aspect: float, max_bands: int, rng):
    """Stack edge-connected bands; each band is one rectangle plus a remainder.

    Work and output size depend on max_bands, not on tile count. All bands
    contain the anchor column. Each AOI therefore includes its sampled anchor.
    """
    anchor_col, anchor_row = grid.cell(lon, lat)
    height = max(1, min(grid.rows, tiles, round(sqrt(tiles / aspect))))
    bands = int(rng.integers(1, min(max_bands, height) + 1))
    heights = np.full(bands, height // bands, dtype="int64")
    heights[:height % bands] += 1
    rng.shuffle(heights)
    # At least one column per band, then allocate the remaining cells exactly.
    weights = heights * rng.uniform(0.7, 1.3, size=bands)
    shares = (tiles - height) * weights / weights.sum()
    extra = np.floor(shares).astype("int64")
    remaining = int(tiles - height - extra.sum())
    extra[np.argsort(-(shares - extra), kind="stable")[:remaining]] += 1
    counts = heights + extra
    row = min(max(0, anchor_row - height // 2), grid.rows - height)
    rectangles = []
    for h, n in zip(heights.tolist(), counts.tolist()):
        width, remainder = divmod(n, h)
        if width + bool(remainder) > grid.columns:
            raise ValueError("AOI too wide for grid; reduce area or aspect ratio")
        col = anchor_col - width // 2
        rectangles.extend(TileRect(col, row, width, h).split_at_seam(grid))
        if remainder:
            rectangles.extend(TileRect(col + width, row, 1, remainder).split_at_seam(grid))
        row += h
    return tuple(rectangles)


def generate(demand: DemandField, config: GenerationConfig):
    if demand.grid.radius_km != config.grid.radius_km:
        raise ValueError("demand and tile grids must use the same Earth radius")
    seeds = np.random.SeedSequence(config.seed).spawn(4)
    location_rng, shape_rng, parent_rng, rotation_rng = [np.random.Generator(np.random.PCG64(s)) for s in seeds]
    sampler = LocationSampler(demand, config, location_rng, parent_rng)
    grid = config.grid
    for i in range(config.count):
        lon, lat, hotspot = sampler.sample()
        area = float(np.exp(shape_rng.uniform(log(config.min_area_km2), log(config.max_area_km2))))
        aspect = float(np.exp(shape_rng.uniform(log(config.min_aspect), log(config.max_aspect))))
        if config.shape == "oriented_rectangle":
            from .oriented import oriented_rectangle
            angle = float(rotation_rng.uniform(0, 180)) if config.random_rotation else 0.0
            yield oriented_rectangle(f"aoi-{i:08d}", lon, lat, area, hotspot, aspect, angle)
            continue
        tiles = max(1, round(area / grid.cell_area_km2))
        if config.shape == "rectangle":
            rects = rectangle(grid, lon, lat, tiles, aspect)
        else:
            rects = mosaic(grid, lon, lat, tiles, aspect, config.max_bands, shape_rng)
        yield Aoi(f"aoi-{i:08d}", lon, lat, area, hotspot, rects)
