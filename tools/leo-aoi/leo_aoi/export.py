"""Export prepared NumPy caches for the Python-free Rust/browser AOI generator."""
import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np


def export(source, destination):
    source, destination = Path(source), Path(destination)
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported cache schema")
        document = {"schema_version": 1, "metadata": metadata}
        if "probability" in data:
            probability = data["probability"]
            eligible = data["eligible"]
            rows, columns = probability.shape
            indices = np.flatnonzero(probability.ravel() > 0)
            document.update(kind="demand", columns=columns, rows=rows,
                            cells=[[int(i), float(probability.flat[i])] for i in indices])
            if not eligible.all():
                document["eligible"] = base64.b64encode(
                    np.packbits(eligible.ravel(), bitorder="little").tobytes()).decode("ascii")
        elif metadata["kind"] == "forest":
            mask = data["mask"]
            rows, columns = mask.shape
            document.update(kind="forest", columns=columns, rows=rows,
                            mask=base64.b64encode(np.packbits(mask.ravel(),
                                                          bitorder="little").tobytes()).decode("ascii"))
        elif metadata["kind"] in ("river", "coast"):
            document.update(kind=metadata["kind"], network={
                key: base64.b64encode(data[key].astype(dtype, copy=False).tobytes()).decode("ascii")
                for key, dtype in (("points", "<f4"), ("offsets", "<u8"),
                                   ("next_index", "<i4"), ("length", "<f8"),
                                   ("ids", "<i8"), ("flow", "<f4"))})
        else:
            raise ValueError("Unknown geographic cache kind")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A fixed gzip timestamp makes committed datasets reproducible.
    payload = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
    destination.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return document["kind"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"Exported {export(args.source, args.destination)} dataset to {args.destination}")


if __name__ == "__main__":
    main()
