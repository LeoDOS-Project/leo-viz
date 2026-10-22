"""Freeze full AOI records and display geometry into a reproducible gzip preset."""

import argparse
import gzip
import io
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import rasterio
import shapely

from . import __version__
from .demand import sha256
from .viewer import write_preview


def freeze(request, destination, name):
    """Write a new gzip and provenance sidecar; never replace an existing preset."""
    destination = Path(destination)
    manifest_path = destination.with_suffix(".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise ValueError(f"preset already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preset-", dir=destination.parent) as staging:
        archive = Path(staging) / destination.name
        with archive.open("wb") as raw:
            # A zero timestamp and empty original filename make gzip deterministic.
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=9) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                    write_preview(request, stream)
        with gzip.open(archive, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
        aois = document["aois"]
        areas = np.array([a["area_km2"] for a in aois])
        manifest = {
            "schema_version": 1, "name": name, "generator": "leo-aoi",
            "generator_version": __version__, "config": document["config"],
            "grid": document["grid"], "demand_provenance": document["demand_provenance"],
            "demand_sha256": sha256(request["demand_path"]) if request.get("demand_path") else None,
            "requested_mean_area_km2": request["mean_area_km2"],
            "size_ratio": request["size_ratio"],
            "rng": "NumPy PCG64, independent location/shape/parent/rotation streams",
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
                         "shapely": shapely.__version__},
            "summary": {"count": len(aois), "mean_area_km2": float(areas.mean()),
                        "min_area_km2": float(areas.min()), "max_area_km2": float(areas.max()),
                        "total_tile_memberships": (sum(a["tile_count"] for a in aois) if all(a["tile_count"] is not None for a in aois) else None)},
            "artifact": {"file": destination.name, "format": "gzip-compressed viewer schema v1 JSON",
                         "bytes": archive.stat().st_size, "sha256": sha256(archive)},
        }
        metadata = Path(staging) / manifest_path.name
        metadata.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        # Exclusive creates also protect against another writer arriving during generation.
        # Link rather than rename: rename can silently replace a destination on POSIX.
        import os
        os.link(archive, destination)
        try:
            os.link(metadata, manifest_path)
        except BaseException:
            destination.unlink()
            raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("settings", type=Path)
    parser.add_argument("demand", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    request = json.loads(args.settings.read_text())
    request["demand_path"] = str(args.demand)
    manifest = freeze(request, args.destination, args.name)
    print(json.dumps({"name": manifest["name"], **manifest["summary"], **manifest["artifact"]}, indent=2))


if __name__ == "__main__":
    main()
