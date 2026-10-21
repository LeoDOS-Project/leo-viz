"""Streaming artifacts and coarse overlap statistics without expanding tiles."""

from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np

from . import __version__
from .demand import sha256
from .generate import generate


class Coverage:
    """AOIs touching each demand cell, counted once even for multi-part mosaics."""
    def __init__(self, grid):
        self.grid = grid
        self.counts = np.zeros((grid.rows, grid.columns), dtype="int64")
        self.last = np.full(self.counts.shape, -1, dtype="int64")

    def add(self, aoi, tile_grid, index):
        for rect in aoi.rectangles:
            rows, cols = rect.demand_slices(tile_grid, self.grid)
            previous = self.last[rows, cols]
            self.counts[rows, cols] += previous != index
            previous[:] = index

    def statistics(self):
        touched = self.counts[self.counts > 0]
        return {"definition": "distinct AOIs intersecting each coarse demand cell; not exact tile overlap",
                "grid": self.grid.as_dict(),
                "touched_cells": int(touched.size),
                "multiply_touched_cells": int((touched > 1).sum()),
                "multiply_touched_fraction": float((touched > 1).mean()) if touched.size else 0,
                "maximum_aois_per_cell": int(touched.max()) if touched.size else 0,
                "mean_aois_per_touched_cell": float(touched.mean()) if touched.size else 0}


def write_workload(demand, config, destination, demand_path, geojson=False):
    import rasterio
    import shapely

    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".aoi-", dir=destination.parent))
    grid, coverage = config.grid, Coverage(demand.grid)
    areas, total_tiles, max_rectangles = [], 0, 0
    try:
        with open(staging / "aois.jsonl", "w") as out:
            geo = open(staging / "aois.geojson", "w") if geojson else None
            try:
                if geo:
                    geo.write('{"type":"FeatureCollection","features":[')
                for i, aoi in enumerate(generate(demand, config)):
                    out.write(json.dumps(aoi.as_dict(grid), separators=(",", ":"), allow_nan=False) + "\n")
                    if geo:
                        if i:
                            geo.write(",")
                        json.dump(aoi.geojson(grid), geo, separators=(",", ":"), allow_nan=False)
                    coverage.add(aoi, grid, i)
                    areas.append(aoi.tile_count * grid.cell_area_km2)
                    total_tiles += aoi.tile_count
                    max_rectangles = max(max_rectangles, len(aoi.rectangles))
                if geo:
                    geo.write("]}\n")
            finally:
                if geo:
                    geo.close()
        summary = {"aoi_count": len(areas), "total_tile_memberships": total_tiles,
                   "sum_aoi_area_km2": sum(areas), "maximum_rectangles_per_aoi": max_rectangles,
                   "area_km2_quantiles": dict(zip(["min", "p25", "p50", "p75", "max"],
                                                  np.quantile(areas, [0, .25, .5, .75, 1]).tolist())),
                   "coarse_coverage": coverage.statistics()}
        with open(staging / "coverage.npz", "wb") as f:
            np.savez_compressed(f, aoi_count=coverage.counts,
                                grid=json.dumps(demand.grid.as_dict(), sort_keys=True))
        manifest = {"schema_version": 1, "generator": "leo-aoi", "generator_version": __version__,
                    "config": asdict(config), "grid": grid.as_dict(),
                    "area_distribution": "log_uniform", "aspect_distribution": "log_uniform",
                    "rng": "NumPy PCG64, independent location/shape/parent streams",
                    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
                                 "shapely": shapely.__version__},
                    "demand_sha256": sha256(demand_path), "demand_provenance": demand.provenance,
                    "artifacts": {p.name: sha256(p) for p in sorted(staging.iterdir())}}
        for name, value in (("manifest.json", manifest), ("summary.json", summary)):
            (staging / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        # Fail instead of replacing an existing directory created by another run.
        if destination.exists():
            raise ValueError(f"output already exists: {destination}")
        staging.rename(destination)
        return summary
    except BaseException:
        shutil.rmtree(staging)
        raise
