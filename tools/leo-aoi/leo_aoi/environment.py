"""Bounded, seeded monitoring footprints guided by real geographic features."""
from dataclasses import asdict, dataclass
import json
from math import ceil, floor, log, pi, radians
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Polygon, box
from shapely.ops import transform, unary_union

from .demand import checked_keys, sha256
from .environment_data import ROOT, ensure, progress, segment_lengths
from .geometry import Aoi, Grid, TileRect


@dataclass(frozen=True)
class Config:
    kind: str = "forest"
    count: int = 1000
    seed: int = 42
    mean_area_km2: float = 1000.0
    size_ratio: float = 100.0
    mean_length_km: float = 100.0
    length_ratio: float = 20.0
    buffer_km: float = 2.0
    inland_km: float = 2.0
    offshore_km: float = 1.0
    min_discharge: float = 10.0
    max_elongation: float = 3.0
    outline_variation: float = 0.7
    forest_adherence: float = 0.8
    compactness: float | None = None  # Explicit legacy requests retain old geometry.
    max_latitude: float = 70.0
    tile_edge_km: float = 0.1
    data_dir: str = str(ROOT)

    def __post_init__(self):
        if self.kind not in ("forest", "river", "coast"):
            raise ValueError("Unknown environmental source")
        if type(self.count) is not int or not 1 <= self.count <= 50000:
            raise ValueError("Environmental count must be in [1, 50000]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        ranges = {"mean_area_km2": (10, 100000), "size_ratio": (1, 1000),
                  "mean_length_km": (1, 1000), "length_ratio": (1, 100),
                  "buffer_km": (.1, 50), "inland_km": (.1, 50), "offshore_km": (.1, 50),
                  "min_discharge": (1, 1e6), "max_elongation": (1, 10),
                  "outline_variation": (0, 1), "forest_adherence": (0, 1),
                  "max_latitude": (1, 85), "tile_edge_km": (.05, 10)}
        if self.compactness is not None:
            ranges["compactness"] = (0, 1)
        for name, (lo, hi) in ranges.items():
            value = getattr(self, name)
            if not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name} must be in [{lo}, {hi}]")
        mean, ratio, limit = ((self.mean_area_km2, self.size_ratio, 1_000_000)
                              if self.kind == "forest" else (self.mean_length_km, self.length_ratio, 3000))
        maximum = mean if ratio == 1 else mean * log(ratio) * ratio / (ratio - 1)
        if maximum > limit:
            raise ValueError(f"Requested size range exceeds {limit:,}; reduce the mean or max/min ratio")

    @classmethod
    def from_dict(cls, data):
        checked_keys(data, cls.__dataclass_fields__, "environment config")
        return cls(**data)


def draw_size(rng, mean, ratio):
    minimum = mean if ratio == 1 else mean * log(ratio) / (ratio - 1)
    return float(np.exp(rng.uniform(log(minimum), log(minimum * ratio))))


def grid_polygon(geometry, grid, center_lon):
    """Geographic polygon to unwrapped tile coordinates, with a common seam."""
    def project(x, y, z=None):
        x, y = np.asarray(x), np.asarray(y)
        lon = center_lon + (x - center_lon + 180) % 360 - 180
        return ((lon + 180) / 360 * grid.columns,
                (1 - np.sin(np.radians(y))) / 2 * grid.rows)
    return transform(project, geometry)


def rectangles_from_polygon(polygon, grid):
    """Rasterize in bounded row strips and merge identical horizontal runs.

    all_touched preserves narrow connected corridors. Work scales with the local
    bounding box, never the global 100 m grid; no full-resolution global bitmap.
    """
    import rasterio.features
    from affine import Affine
    x0, y0, x1, y1 = polygon.bounds
    x0, x1 = floor(x0), ceil(x1)
    y0, y1 = max(0, floor(y0)), min(grid.rows, ceil(y1))
    width = x1 - x0
    if width <= 0 or y1 <= y0 or width > grid.columns:
        raise ValueError("Invalid environmental footprint extent")
    if width * (y1 - y0) > 100_000_000:
        raise ValueError("Footprint bounding box exceeds 100 million tiles; reduce size or increase tile_edge_km")
    active, result = {}, []
    for start in range(y0, y1, 128):
        height = min(128, y1 - start)
        pixels = rasterio.features.rasterize([(polygon, 1)], out_shape=(height, width),
                    transform=Affine.translation(x0, start), all_touched=True, dtype="uint8")
        for j, row in enumerate(pixels):
            edges = np.flatnonzero(np.diff(np.pad(row, (1, 1)).astype("int16")))
            runs = {(x0 + int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2])}
            current = {}
            for col, w in sorted(runs):
                previous = active.pop((col, w), None)
                current[(col, w)] = TileRect(col, previous.row if previous else start + j,
                                            w, previous.height + 1 if previous else 1)
            result.extend(active.values())
            active = current
    result.extend(active.values())
    return tuple(part for rect in result for part in rect.split_at_seam(grid))


class Forest:
    def __init__(self, data, config):
        self.mask = data["mask"].astype(bool)
        self.rows, self.cols = self.mask.shape
        self.step = 360 / self.cols
        if abs(self.rows * self.step - 180) > 1e-6:
            raise ValueError("Invalid forest sampling grid")
        north = np.radians(90 - np.arange(self.rows) * self.step)
        south = north - radians(self.step)
        self.area = 6371.0 ** 2 * radians(self.step) * (np.sin(north) - np.sin(south))
        eligible = np.abs(90 - (np.arange(self.rows) + .5) * self.step) <= config.max_latitude
        self.weights = np.cumsum(self.mask.sum(axis=1) * self.area * eligible)
        if self.weights[-1] <= 0:
            raise ValueError("No tree-cover cells in the latitude band")
        self.columns = {}

    def sample(self, location_rng, shape_rng, config, grid):
        row = int(np.searchsorted(self.weights, location_rng.random() * self.weights[-1], side="right"))
        if row not in self.columns:
            self.columns[row] = np.flatnonzero(self.mask[row])
        col = int(location_rng.choice(self.columns[row]))
        lon, lat = (col + .5) * self.step - 180, 90 - (row + .5) * self.step
        requested = draw_size(shape_rng, config.mean_area_km2, config.size_ratio)
        if requested > 1_000_000:
            raise ValueError("Forest target exceeds 1 million km²; reduce the size range")
        from .forest_growth import grow, grow_legacy
        if config.compactness is None:
            # A fixed draw per AOI keeps target areas stable when shape controls change.
            morphology_rng = np.random.default_rng(int(shape_rng.integers(0, 2**63)))
            selected, extra = grow(self.mask, self.area, row, col, self.step,
                                   requested, morphology_rng, config)
        else:
            selected, extra = grow_legacy(self.mask, self.area, row, col, self.step,
                                          requested, shape_rng, config.compactness)
        # Build contiguous row runs before unioning, keeping large patches compact.
        by_row = {}
        for y, x in selected:
            x = col + int((x-col+self.cols//2) % self.cols-self.cols//2)
            by_row.setdefault(y, []).append(x)
        pieces = []
        for y, xs in sorted(by_row.items()):
            xs.sort()
            start = previous = xs[0]
            for x in xs[1:] + [xs[-1] + 2]:
                if x != previous + 1:
                    pieces.append(box(start, y, previous + 1, y + 1))
                    start = x
                previous = x
        polygon = unary_union(pieces)
        # New growth preserves clearings; only legacy replay fills holes.
        if config.compactness is not None and polygon.geom_type == "Polygon":
            polygon = Polygon(polygon.exterior)
        geographic = transform(lambda x, y, z=None: (np.asarray(x) * self.step - 180,
                                                     90 - np.asarray(y) * self.step), polygon)
        rects = rectangles_from_polygon(grid_polygon(geographic, grid, lon), grid)
        extra["tree_cover_cell_fraction"] = float(np.mean([self.mask[y, x % self.cols] for y, x in selected]))
        return lon, lat, requested, rects, extra


class Network:
    def __init__(self, data, config):
        self.points, self.offsets = data["points"], data["offsets"]
        self.next = data["next_index"]
        self.length = data["length"]
        self.ids = data["ids"]
        # Weight eligible geographic length, including portions of lines that
        # cross the latitude band. Most records are wholly inside or outside.
        low = np.minimum.reduceat(self.points[:, 1], self.offsets[:-1])
        high = np.maximum.reduceat(self.points[:, 1], self.offsets[:-1])
        band = config.max_latitude
        eligible = (low <= band) & (high >= -band)
        if config.kind == "river":
            eligible &= data["flow"] >= config.min_discharge
        weights = self.length * eligible
        self.clipped = {}
        for index in np.flatnonzero(eligible & ((low < -band) | (high > band))):
            points = self.points_for(index)
            lengths = segment_lengths(points)
            a, b = points[:-1, 1], points[1:, 1]
            difference = b-a
            nonzero = difference != 0
            t0 = np.zeros(len(lengths))
            t1 = np.ones(len(lengths))
            t0[nonzero] = (-band-a[nonzero]) / difference[nonzero]
            t1[nonzero] = (band-a[nonzero]) / difference[nonzero]
            begin = np.clip(np.minimum(t0, t1), 0, 1)
            end = np.clip(np.maximum(t0, t1), 0, 1)
            allowed = lengths * (end-begin)
            allowed[~nonzero & (np.abs(a) > band)] = 0
            cumulative = np.cumsum(allowed)
            weights[index] = cumulative[-1]
            self.clipped[index] = (cumulative, np.r_[0, np.cumsum(lengths)[:-1]], begin, lengths)
        self.weights = np.cumsum(weights)
        if not len(self.weights) or self.weights[-1] <= 0:
            raise ValueError("No lines meet the latitude/discharge filters")

    def points_for(self, index):
        return self.points[self.offsets[index]:self.offsets[index + 1]].astype("float64")

    def sample(self, location_rng, shape_rng, config, grid):
        index = int(np.searchsorted(self.weights, location_rng.random() * self.weights[-1], side="right"))
        feature = int(self.ids[index])
        fraction = location_rng.random()
        if index in self.clipped:
            cumulative, starts, begin, lengths = self.clipped[index]
            distance = fraction * cumulative[-1]
            segment = int(np.searchsorted(cumulative, distance, side="right"))
            offset = (starts[segment] + begin[segment] * lengths[segment] + distance
                      - (cumulative[segment-1] if segment else 0))
        else:
            offset = fraction * self.length[index]
        requested_length = draw_size(shape_rng, config.mean_length_km, config.length_ratio)
        if requested_length > 3000:
            raise ValueError("Corridor exceeds 3,000 km; reduce length or range")
        initial_index, initial_offset = index, offset
        remaining, path, seen = requested_length, [], set()
        while index >= 0 and remaining > 1e-6:
            if index in seen:
                if config.kind != "coast" or index != initial_index:
                    break
                # Complete the first partial record when returning around an
                # island, then stop exactly at the sampled starting position.
                remaining = min(remaining, initial_offset)
                if remaining <= 1e-6:
                    break
            seen.add(index)
            points = self.points_for(index)
            lengths = segment_lengths(points)
            for a, b, length in zip(points[:-1], points[1:], lengths):
                if offset >= length:
                    offset -= length
                    continue
                if length <= 0:
                    continue
                # Coordinates are locally unwrapped so interpolation crosses the seam correctly.
                b[0] = a[0] + (b[0] - a[0] + 180) % 360 - 180
                start = a + (b-a) * offset / length
                use = min(remaining, length-offset)
                end = a + (b-a) * (offset+use) / length
                if path:
                    shift = 360 * round((path[-1][0] - start[0]) / 360)
                    start[0] += shift
                    end[0] += shift
                    # A missing or inconsistent connection must never draw a false bridge.
                    if float(segment_lengths(np.array([path[-1], start]))[0]) > 0.5:
                        remaining = 0
                        break
                else:
                    path.append(start.copy())
                path.append(end.copy())
                remaining -= use
                offset = 0
                if remaining <= 1e-6:
                    break
            index = int(self.next[index])
        if len(path) < 2:
            raise ValueError("Could not form a nonempty geographic corridor")
        path = np.asarray(path)
        lon, lat = float((path[0, 0] + 180) % 360 - 180), float(path[0, 1])
        local = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
        forward = Transformer.from_crs("EPSG:4326", local, always_xy=True)
        backward = Transformer.from_crs(local, "EPSG:4326", always_xy=True)
        line = transform(forward.transform, LineString(path))
        if config.kind == "coast":
            # OSM coastline direction has land on the left, water on the right.
            footprint = unary_union([line.buffer(config.inland_km*1000, single_sided=True),
                                     line.buffer(-config.offshore_km*1000, single_sided=True)])
        else:
            footprint = line.buffer(config.buffer_km * 1000, quad_segs=4)
        # Metre-based simplification stays below one projected tile edge.
        footprint = footprint.simplify(config.tile_edge_km * 250, preserve_topology=True)
        geographic = transform(backward.transform, footprint)
        polygon = grid_polygon(geographic, grid, lon)
        rects = rectangles_from_polygon(polygon, grid)
        actual_length = float(segment_lengths(path).sum())
        requested_area = (requested_length * (config.inland_km+config.offshore_km) if config.kind == "coast"
                          else requested_length * 2 * config.buffer_km + pi * config.buffer_km**2)
        return lon, lat, requested_area, rects, {"source_feature_id": feature,
            "requested_length_km": requested_length, "path_length_km": actual_length,
            "path_truncated": actual_length < requested_length * .99}


def prepare(options):
    config = Config.from_dict(options)
    path = ensure(config.kind, Path(config.data_dir))
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["metadata"]))
    if meta.get("schema_version") != 1 or meta.get("kind") != config.kind:
        raise ValueError("Environmental cache has the wrong kind/version")
    sampler = Forest(data, config) if config.kind == "forest" else Network(data, config)
    data.close()
    grid = Grid.with_tile_edge(config.tile_edge_km)
    header = {"schema_version": 1, "grid": grid.as_dict(), "config": asdict(config),
              "demand_provenance": {**meta, "cache_sha256": sha256(path)}}
    seeds = np.random.SeedSequence(config.seed).spawn(2)
    location_rng, shape_rng = [np.random.Generator(np.random.PCG64(s)) for s in seeds]
    def records():
        for i in range(config.count):
            lon, lat, requested, rects, extra = sampler.sample(location_rng, shape_rng, config, grid)
            aoi = Aoi(f"{config.kind}-{i:08d}", lon, lat, requested, None, rects)
            record = aoi.as_dict(grid)
            record["geometry"] = aoi.geojson(grid)["geometry"]
            record["source_kind"] = config.kind
            record.update(extra)
            yield record
            if i % 100 == 0 or i+1 == config.count:
                progress(f"Generating {config.kind} AOIs: {i+1:,}/{config.count:,}")
    return header, records()
