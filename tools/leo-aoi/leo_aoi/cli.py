import argparse
import json
from pathlib import Path

from .demand import DemandField, ingest
from .generate import GenerationConfig
from .output import write_workload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate unspecified AOIs from spatial demand rasters")
    commands = parser.add_subparsers(dest="command", required=True)
    ingestion = commands.add_parser("ingest", help="combine GeoTIFF layers into a reusable demand grid")
    ingestion.add_argument("config", type=Path)
    ingestion.add_argument("output", type=Path)
    generation = commands.add_parser("generate", help="generate rectangular AOI mosaics")
    generation.add_argument("demand", type=Path)
    generation.add_argument("config", type=Path)
    generation.add_argument("output_directory", type=Path)
    generation.add_argument("--geojson", action="store_true", help="also export dissolved AOI polygons")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text())
        if args.command == "ingest":
            field = ingest(config, args.config.resolve().parent)
            field.save(args.output)
            print(json.dumps({"demand_cells": field.probability.size,
                              "positive_cells": int((field.probability > 0).sum())}))
        else:
            summary = write_workload(DemandField.load(args.demand), GenerationConfig.from_dict(config),
                                     args.output_directory, args.demand, args.geojson)
            print(json.dumps(summary, indent=2))
    except (ValueError, TypeError, KeyError, OSError) as e:
        parser.exit(2, f"leo-aoi: {e}\n")


if __name__ == "__main__":
    main()
