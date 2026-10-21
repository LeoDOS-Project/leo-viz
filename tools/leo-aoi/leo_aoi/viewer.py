"""JSON/stdin adapter for the native visualizer; reuses the batch generator."""

from dataclasses import asdict
import json
from math import isfinite, log
from pathlib import Path
import sys

from .demand import DemandField, checked_keys, ingest
from .generate import GenerationConfig, generate


def area_bounds(mean, ratio):
    """Log-uniform limits with the requested arithmetic expectation and max/min."""
    if not isfinite(mean) or mean <= 0 or not isfinite(ratio) or ratio < 1:
        raise ValueError("average area must be positive and size ratio must be at least 1")
    minimum = mean if ratio == 1 else mean * log(ratio) / (ratio - 1)
    return minimum, minimum * ratio


def prepare(request):
    checked_keys(request, {"schema_version", "demand_path", "mean_area_km2", "size_ratio",
                           "generation"}, "viewer request")
    if request.get("schema_version") != 1:
        raise ValueError("unsupported viewer request schema")
    options = dict(request.get("generation", {}))
    if "min_area_km2" in options or "max_area_km2" in options:
        raise ValueError("viewer requests use mean_area_km2 and size_ratio")
    minimum, maximum = area_bounds(request["mean_area_km2"], request["size_ratio"])
    config = GenerationConfig.from_dict({**options, "min_area_km2": minimum,
                                        "max_area_km2": maximum})
    if config.count > 50_000:
        raise ValueError("interactive generation is limited to 50,000 AOIs")
    path = request.get("demand_path")
    if path:
        if not Path(path).is_file():
            raise ValueError(f"Demand field not found: {path}. Run leo_aoi ingest first, "
                             "choose an existing demand.npz, or select Uniform globe.")
        demand = DemandField.load(path)
    else:
        demand = ingest({"grid": {"columns": 360, "rows": 180}, "layers": [],
                         "background_fraction": 1})
    return demand, config


def write_preview(request, stream):
    if "environment" in request:
        checked_keys(request, {"schema_version", "environment"}, "environment viewer request")
        if request.get("schema_version") != 1:
            raise ValueError("unsupported viewer request schema")
        from .environment import prepare as prepare_environment
        header, records = prepare_environment(request["environment"])
    else:
        demand, config = prepare(request)
        header = {"schema_version": 1, "grid": config.grid.as_dict(),
                  "config": asdict(config), "demand_provenance": demand.provenance}
        def batch():
            for aoi in generate(demand, config):
                record = aoi.as_dict(config.grid)
                record["geometry"] = aoi.geojson(config.grid)["geometry"]
                yield record
        records = batch()
    stream.write(json.dumps(header, separators=(",", ":"), allow_nan=False)[:-1])
    stream.write(',"aois":[')
    for i, record in enumerate(records):
        if i:
            stream.write(",")
        json.dump(record, stream, separators=(",", ":"), allow_nan=False)
    stream.write("]}\n")


def main():
    try:
        write_preview(json.load(sys.stdin), sys.stdout)
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"AOI generation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
