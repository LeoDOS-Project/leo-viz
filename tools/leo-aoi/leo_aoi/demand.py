"""GeoTIFF ingestion, independent layer normalization, masks and provenance."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import Grid


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_keys(value, allowed, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    unknown = value.keys() - set(allowed)
    if unknown:
        raise ValueError(f"unknown {context} fields: {sorted(unknown)}")


@dataclass
class DemandField:
    grid: Grid
    probability: np.ndarray
    eligible: np.ndarray
    provenance: dict

    def __post_init__(self):
        shape = (self.grid.rows, self.grid.columns)
        if self.probability.shape != shape or self.eligible.shape != shape:
            raise ValueError("demand arrays do not match grid")
        if not np.isfinite(self.probability).all() or (self.probability < 0).any():
            raise ValueError("demand must be finite and nonnegative")
        self.eligible = self.eligible.astype(bool)
        if (self.probability[~self.eligible] > 0).any():
            raise ValueError("demand has mass in ineligible cells")
        total = self.probability.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("demand has no positive finite mass")
        self.probability = self.probability / total

    def save(self, path):
        metadata = {"schema_version": 1, "grid": {"columns": self.grid.columns,
                    "rows": self.grid.rows, "radius_km": self.grid.radius_km},
                    "provenance": self.provenance}
        with open(path, "xb") as f:
            np.savez_compressed(f, probability=self.probability, eligible=self.eligible,
                                metadata=json.dumps(metadata, sort_keys=True))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["schema_version"] != 1:
                raise ValueError("unsupported demand schema")
            return cls(Grid(**meta["grid"]), data["probability"], data["eligible"], meta["provenance"])


def read_raster(spec, base: Path, grid: Grid):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject

    checked_keys(spec, {"path", "band", "kind", "weight", "nodata", "citation", "label"}, "layer")
    kind = spec.get("kind", "mass")
    if kind not in ("mass", "density", "mask"):
        raise ValueError("layer kind must be mass, density or mask")
    path = (base / spec["path"]).resolve()
    band = spec.get("band", 1)
    if type(band) is not int or band < 1:
        raise ValueError("raster band must be a positive integer (one-based)")
    output = np.zeros((grid.rows, grid.columns), dtype="float64")
    with rasterio.open(path) as src:
        if src.crs is None or band > src.count:
            raise ValueError(f"missing CRS or invalid band for {path}")
        if src.scales[band - 1] != 1 or src.offsets[band - 1] != 0:
            raise ValueError(f"apply raster scale/offset before ingestion: {path}")
        nodata = spec.get("nodata", src.nodata)
        # GDAL streams source blocks; do not read an entire fine-resolution global
        # source into Python. sum is for extensive values; average for densities.
        reproject(source=rasterio.band(src, band), destination=output,
                  src_nodata=nodata, dst_nodata=0,
                  dst_transform=from_bounds(*grid.bounds_m, grid.columns, grid.rows),
                  dst_crs=grid.crs,
                  resampling=Resampling.sum if kind == "mass" else Resampling.average,
                  num_threads=1, warp_mem_limit=128)
        # GDAL may retain the source sentinel in empty destination cells (in
        # particular during identity warps). It must not become demand mass.
        if nodata is not None:
            output[np.isnan(output) if np.isnan(nodata) else output == nodata] = 0
        metadata = {**spec, "path": str(path), "sha256": sha256(path), "band": band,
                    "kind": kind, "source_crs": src.crs.to_string(),
                    "band_description": src.descriptions[band - 1],
                    "source_shape": [src.height, src.width], "nodata": str(nodata),
                    "resampling": "sum" if kind == "mass" else "average"}
    if not np.isfinite(output).all() or (output < 0).any():
        raise ValueError(f"invalid raster values in {path}; declare its nodata value explicitly")
    if kind == "mask" and (output > 1).any():
        raise ValueError("mask raster must contain fractions from zero to one")
    if kind == "density":
        # Density inputs are amounts per km²; every destination cell has equal area.
        output *= grid.cell_area_km2
    metadata["resampled_total"] = float(output.sum())
    return output, metadata


def ingest(config, base=Path(".")):
    checked_keys(config, {"schema_version", "grid", "layers", "mask", "mask_threshold",
                          "background_fraction", "concentration"}, "ingestion config")
    if config.get("schema_version", 1) != 1:
        raise ValueError("unsupported ingestion schema")
    grid = Grid(**config.get("grid", {"columns": 1440, "rows": 720}))
    if grid.rows * grid.columns > 20_000_000:
        raise ValueError("demand grid exceeds 20 million cells; use a coarser demand grid")
    shape = (grid.rows, grid.columns)
    eligible = np.ones(shape, dtype=bool)
    provenance = {"config": config, "sources": []}
    if config.get("mask"):
        mask_spec = {**config["mask"], "kind": "mask"}
        mask, meta = read_raster(mask_spec, base, grid)
        threshold = config.get("mask_threshold", 0.5)
        if not np.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("mask_threshold must be in (0, 1]")
        eligible = mask >= threshold
        provenance["mask"] = meta
    if not eligible.any():
        raise ValueError("mask excludes every demand cell")
    probability = np.zeros(shape, dtype="float64")
    weights = 0.0
    for layer in config.get("layers", []):
        if layer.get("kind") == "mask":
            raise ValueError("put masks in the mask field, not in the demand mixture")
        weight = layer.get("weight", 1.0)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("mixture weights must be finite and positive")
        values, meta = read_raster(layer, base, grid)
        values[~eligible] = 0
        total = values.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"layer has no positive mass after masking: {layer['path']}")
        probability += weight * values / total
        weights += weight
        provenance["sources"].append(meta)
    background = config.get("background_fraction", 0.0 if weights else 1.0)
    concentration = config.get("concentration", 1.0)
    if not np.isfinite(background) or not 0 <= background <= 1:
        raise ValueError("background_fraction must be in [0, 1]")
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be finite and positive")
    if weights:
        probability /= weights
        # Scale before powering to avoid underflow of small probabilities.
        probability = (probability / probability.max()) ** concentration
        probability /= probability.sum()
    elif background != 1:
        raise ValueError("without demand layers background_fraction must be 1")
    probability = ((1 - background) * probability
                   + background * eligible / eligible.sum())
    provenance["resolved_background_fraction"] = background
    provenance["resolved_concentration"] = concentration
    return DemandField(grid, probability, eligible, provenance)
