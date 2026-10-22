"""Fetch a small, pinned real-data example. Larger rasters can be supplied directly."""

import argparse
import hashlib
import os
from pathlib import Path
import ssl
import tempfile
from urllib.request import urlopen

import certifi

URL = "https://zenodo.org/records/18429133/files/rast_gdpTot_1990_2024_30arcmin.tif?download=1"
SHA256 = "29c32a5e7f96e6f07f4103ba9099e3181408619580c2ed30e9fe5264fea20f58"


def main():
    parser = argparse.ArgumentParser(description="Download the pinned 9.5 MB GDP example raster")
    parser.add_argument("destination", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/gdp-total-1990-2024-30arcmin.tif")
    path = parser.parse_args().destination
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
            parser.error(f"existing file has a different checksum: {path}")
        print(f"Verified {path}")
        return
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            name = Path(f.name)
            digest = hashlib.sha256()
            with urlopen(URL, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    f.write(chunk)
            if digest.hexdigest() != SHA256:
                raise ValueError("download checksum differs from the pinned dataset version")
        # Atomic exclusive publication: never replace another download or user file.
        os.link(name, path)
        print(f"Downloaded and verified {path}")
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
