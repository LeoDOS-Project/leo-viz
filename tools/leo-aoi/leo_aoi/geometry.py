"""Compact rectangular mosaics on a common spherical equal-area grid.

The Earth radius matches leo-sim::geometry::EARTH_RADIUS_KM. Geometry stays
separate from leo-sim's circular acquisition Target: an AOI is a requirement,
not an acquisition footprint or an instruction to the sensor.
"""

from dataclasses import asdict, dataclass
from math import asin, degrees, isfinite, pi, radians, sin
from typing import Iterator

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class Grid:
    columns: int
    rows: int
    radius_km: float = EARTH_RADIUS_KM

    def __post_init__(self):
        if any(type(n) is not int or not 1 <= n <= 2**31 for n in (self.columns, self.rows)):
            raise ValueError("grid dimensions must be positive integers <= 2^31")
        if not isfinite(self.radius_km) or self.radius_km <= 0:
            raise ValueError("radius_km must be finite and positive")

    @classmethod
    def with_tile_edge(cls, edge_km: float):
        if not isfinite(edge_km) or edge_km <= 0:
            raise ValueError("tile_edge_km must be finite and positive")
        return cls(max(1, round(2 * pi * EARTH_RADIUS_KM / edge_km)),
                   max(1, round(2 * EARTH_RADIUS_KM / edge_km)))

    @property
    def cell_area_km2(self):
        return 4 * pi * self.radius_km**2 / (self.columns * self.rows)

    @property
    def crs(self):
        return f"+proj=cea +lat_ts=0 +lon_0=0 +R={self.radius_km * 1000} +units=m +no_defs"

    @property
    def bounds_m(self):
        r = self.radius_km * 1000
        return (-pi * r, -r, pi * r, r)

    def lonlat(self, x: float, y: float):
        """Convert grid-edge coordinates (north-west origin) to lon/lat degrees."""
        return (360 * x / self.columns - 180,
                degrees(asin(max(-1, min(1, 1 - 2 * y / self.rows)))))

    def cell(self, lon: float, lat: float):
        if not isfinite(lon) or not isfinite(lat) or not -90 <= lat <= 90:
            raise ValueError("invalid longitude/latitude")
        x = int(((lon + 180) % 360) / 360 * self.columns)
        y = min(self.rows - 1, int((1 - sin(radians(lat))) / 2 * self.rows))
        return x, y

    def as_dict(self):
        return {**asdict(self), "projection": "spherical_cylindrical_equal_area",
                "origin": "north_west", "cell_area_km2": self.cell_area_km2}


@dataclass(frozen=True)
class TileRect:
    col: int
    row: int
    width: int
    height: int

    @property
    def tile_count(self):
        return self.width * self.height

    def split_at_seam(self, grid: Grid):
        """Canonical non-wrapping rectangles; longitude wraps, latitude does not."""
        if not (0 < self.width <= grid.columns and self.height > 0
                and 0 <= self.row < self.row + self.height <= grid.rows):
            raise ValueError("rectangle outside grid")
        start = self.col % grid.columns
        width = min(self.width, grid.columns - start)
        parts = [TileRect(start, self.row, width, self.height)]
        if width < self.width:
            parts.append(TileRect(0, self.row, self.width - width, self.height))
        return parts

    def demand_slices(self, source: Grid, target: Grid):
        # Integer arithmetic avoids loss of precision at a shared tile boundary.
        x0 = self.col * target.columns // source.columns
        x1 = ((self.col + self.width) * target.columns + source.columns - 1) // source.columns
        y0 = self.row * target.rows // source.rows
        y1 = ((self.row + self.height) * target.rows + source.rows - 1) // source.rows
        return slice(y0, y1), slice(x0, x1)

    def shared_tiles(self, other):
        width = max(0, min(self.col + self.width, other.col + other.width) - max(self.col, other.col))
        height = max(0, min(self.row + self.height, other.row + other.height) - max(self.row, other.row))
        return width * height


@dataclass(frozen=True)
class Aoi:
    id: str
    anchor_lon: float
    anchor_lat: float
    requested_area_km2: float
    hotspot: int | None
    rectangles: tuple[TileRect, ...]

    @property
    def tile_count(self):
        return sum(r.tile_count for r in self.rectangles)

    def as_dict(self, grid: Grid):
        return {"id": self.id, "anchor_lon": self.anchor_lon, "anchor_lat": self.anchor_lat,
                "requested_area_km2": self.requested_area_km2,
                "area_km2": self.tile_count * grid.cell_area_km2,
                "tile_count": self.tile_count, "hotspot": self.hotspot,
                "rectangles": [asdict(r) for r in self.rectangles]}

    def tiles(self) -> Iterator[tuple[int, int]]:
        """Lazy (column, row) keys. Do not materialize continent-sized AOIs."""
        for r in self.rectangles:
            for row in range(r.row, r.row + r.height):
                for col in range(r.col, r.col + r.width):
                    yield col, row

    def shared_tiles(self, other):
        """Exact pairwise overlap, for AOIs in the same grid, without expansion."""
        return sum(a.shared_tiles(b) for a in self.rectangles for b in other.rectangles)

    def geojson(self, grid: Grid):
        # Dissolve adjacent rectangles: an undissolved MultiPolygon with shared
        # edges is invalid. Dateline-crossing AOIs correctly remain multipolygons.
        from shapely.geometry import box, mapping
        from shapely.ops import unary_union

        polygons = []
        for r in self.rectangles:
            west, north = grid.lonlat(r.col, r.row)
            east, south = grid.lonlat(r.col + r.width, r.row + r.height)
            polygons.append(box(west, south, east, north))
        return {"type": "Feature", "id": self.id,
                "properties": {k: v for k, v in self.as_dict(grid).items() if k != "rectangles"},
                "geometry": mapping(unary_union(polygons))}
