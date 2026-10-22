"""Prepare reusable geographic sources. Downloads and caches stay outside Git.

WorldCover is sampled from categorical COG overviews, not a forest-fraction
product. This deliberately coarse location/shape proxy is independent of GSD.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import ssl
import sys
import tempfile
import time
from urllib.request import Request, urlopen
import zipfile

import certifi
import numpy as np

from .demand import sha256

ROOT = Path(__file__).resolve().parents[1] / "data/environment"
URLS = {
    "worldcover-grid.geojson": "https://esa-worldcover.s3.eu-central-1.amazonaws.com/esa_worldcover_grid.geojson",
    "rivers.zip": "https://data.hydrosheds.org/file/hydrorivers/HydroRIVERS_v10_shp.zip",
    "coasts.zip": "https://osmdata.openstreetmap.de/download/coastlines-split-4326.zip",
}
WORLD_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"
SOURCES = {
    "forest": ("ESA WorldCover 2021 v200", "https://doi.org/10.5281/zenodo.7254221", "CC BY 4.0; © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium"),
    "river": ("HydroRIVERS v1.0", "https://www.hydrosheds.org/products/hydrorivers", "HydroSHEDS license; Lehner & Grill (2013), doi:10.1002/hyp.9740"),
    "coast": ("OpenStreetMap coastline extract", "https://osmdata.openstreetmap.de/data/coastlines.html", "© OpenStreetMap contributors, ODbL 1.0"),
}

def progress(message):
    print(message, file=sys.stderr, flush=True)

def download(name, root=ROOT):
    path = root / "raw" / name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    progress(f"Downloading {name}…")
    # Atomic publication: interruption never leaves a completed-looking source.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            with urlopen(Request(URLS[name], headers={"User-Agent": "leo-aoi/0.2"}),
                         context=ssl.create_default_context(cafile=certifi.where()), timeout=60) as response:
                if response.status != 200:
                    raise ValueError(f"Unexpected download HTTP status {response.status}")
                expected = int(response.headers.get("Content-Length", 0))
                total, reported = 0, 0
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
                    total += len(chunk)
                    if total - reported > 64 * 1024 * 1024:
                        progress(f"{name}: {total / 1e6:.0f} / {expected / 1e6:.0f} MB")
                        reported = total
                if expected and total != expected:
                    raise ValueError("Incomplete dataset download")
            f.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path

def save(path, metadata, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            np.savez_compressed(f, metadata=json.dumps(metadata, sort_keys=True), **arrays)
            f.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

def provenance(kind, **extra):
    name, citation, attribution = SOURCES[kind]
    return {"schema_version": 1, "kind": kind, "name": name, "citation": citation,
            "attribution": attribution, **extra}

def prepare_forest(root=ROOT):
    import rasterio
    os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())
    index = download("worldcover-grid.geojson", root)
    features = json.loads(index.read_text())["features"]
    cache = root / "worldcover-overviews"
    cache.mkdir(parents=True, exist_ok=True)
    # 40 cells/degree, about 2.8 km N-S. Native 10 m pixels are never fabricated.
    per_degree = 40
    mask = np.zeros((180 * per_degree, 360 * per_degree), dtype="uint8")
    def read(feature):
        name = feature["properties"]["ll_tile"]
        path = cache / (name + ".npy")
        if path.exists():
            data = np.load(path, allow_pickle=False)
        else:
            for attempt in range(3):
                try:
                    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                                      GDAL_HTTP_TIMEOUT="30", GDAL_HTTP_MAX_RETRY="2"):
                        with rasterio.open(WORLD_URL.format(name)) as src:
                            data = src.read(1, out_shape=(3 * per_degree, 3 * per_degree))
                    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as f:
                        np.save(f, data, allow_pickle=False)
                        tmp = Path(f.name)
                    os.replace(tmp, path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        if data.shape != (3 * per_degree, 3 * per_degree):
            raise ValueError(f"Invalid WorldCover cached tile {path}")
        west, south = feature["geometry"]["coordinates"][0][0]
        return int(round((90 - south - 3) * per_degree)), int(round((west + 180) * per_degree)), data
    hashes = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (row, col, data) in enumerate(pool.map(read, features), 1):
            mask[row:row + len(data), col:col + len(data)] = data == 10
            hashes.update(data.tobytes())
            if i % 100 == 0 or i == len(features):
                progress(f"WorldCover sampling map: {i}/{len(features)} tiles")
    meta = provenance("forest", resolution_degrees=1 / per_degree,
                      sampling="nearest categorical COG overview; tree-cover class 10",
                      tile_index_sha256=sha256(index), sampled_tiles_sha256=hashes.hexdigest(),
                      source_url_template=WORLD_URL, tile_count=len(features))
    save(root / "forest.npz", meta, mask=mask)

def segment_lengths(points):
    """Spherical geodesic lengths in km, robust across the dateline."""
    points = np.asarray(points, dtype="float64")
    lon, lat = np.radians(points[:, 0]), np.radians(points[:, 1])
    h = np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

def prepare_lines(kind, root=ROOT):
    import shapefile
    from shapely.geometry import LineString
    source = download("rivers.zip" if kind == "river" else "coasts.zip", root)
    directory = root / "raw" / kind
    directory.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith((".shp", ".shx", ".dbf"))]
        for name in members:
            # Flatten names; never allow archive paths to escape the cache.
            path = directory / Path(name).name
            if not path.exists():
                progress(f"Extracting {path.name}…")
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as dst:
                    temporary = Path(dst.name)
                    try:
                        with archive.open(name) as src:
                            import shutil
                            shutil.copyfileobj(src, dst, 1024 * 1024)
                        dst.close()
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
    shp = next(directory.glob("*.shp"))
    reader = shapefile.Reader(str(shp))
    coordinates, offsets, ids, downstream, flows, lengths = [], [0], [], [], [], []
    names = [f[0] for f in reader.fields[1:]]
    indexes = {name: names.index(name) for name in ("HYRIV_ID", "NEXT_DOWN", "DIS_AV_CMS")} if kind == "river" else {}
    records = reader.iterShapeRecords() if kind == "river" else reader.iterShapes()
    progress(f"Indexing {SOURCES[kind][0]}…")
    for i, record in enumerate(records):
        shape = record.shape if kind == "river" else record
        if kind == "river":
            flow = float(record.record[indexes["DIS_AV_CMS"]])
            if flow < 1.0:
                continue
        points = np.asarray(shape.points, dtype="float64")[:, :2]
        if len(shape.parts) != 1 or len(points) < 2:
            raise ValueError("Expected single-part geographic line records")
        # Keep endpoints and dateline continuity while dropping sub-100 m detail.
        points[:, 0] = np.degrees(np.unwrap(np.radians(points[:, 0])))
        line = LineString(points).simplify(0.0005, preserve_topology=True)
        points = np.asarray(line.coords, dtype="float32")
        length = float(segment_lengths(points).sum())
        if length <= 0:
            continue
        coordinates.append(points)
        offsets.append(offsets[-1] + len(points))
        lengths.append(length)
        ids.append(int(record.record[indexes["HYRIV_ID"]]) if kind == "river" else i)
        downstream.append(int(record.record[indexes["NEXT_DOWN"]]) if kind == "river" else -1)
        flows.append(flow if kind == "river" else 0)
        if i % 200000 == 0:
            progress(f"{kind}: read {i:,} source lines; retained {len(ids):,}")
    reader.close()
    points = np.concatenate(coordinates)
    del coordinates
    offsets = np.asarray(offsets, dtype="int64")
    if kind == "river":
        lookup = {value: i for i, value in enumerate(ids)}
        next_index = np.asarray([lookup.get(value, -1) for value in downstream], dtype="int32")
    else:
        # OSM coastline ways have a consistent direction; join split records.
        def key(p):
            return (round(((float(p[0]) + 180) % 360 - 180) * 1e5), round(float(p[1]) * 1e5))
        starts = {}
        for i, offset in enumerate(offsets[:-1]):
            k = key(points[offset])
            starts[k] = i if k not in starts else -1  # stop at ambiguous junctions
        next_index = np.asarray([starts.get(key(points[end - 1]), -1) for end in offsets[1:]], dtype="int32")
    meta = provenance(kind, source_url=URLS[source.name], source_sha256=sha256(source),
                      min_discharge_m3_s=1.0 if kind == "river" else None,
                      simplify_degrees=0.0005, feature_count=len(ids), length_precision="float64 spherical distances")
    save(root / (kind + ".npz"), meta, points=points, offsets=offsets,
         next_index=next_index, flow=np.asarray(flows, dtype="float32"),
         length=np.asarray(lengths, dtype="float64"), ids=np.asarray(ids, dtype="int64"))
    progress(f"{kind}: ready, {len(ids):,} indexed lines")

def ensure(kind, root=ROOT):
    root = Path(root)
    path = root / (kind + ".npz")
    if not path.exists():
        if kind == "forest":
            prepare_forest(root)
        elif kind in ("river", "coast"):
            prepare_lines(kind, root)
        else:
            raise ValueError("Unknown environmental source")
    return path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["forest", "river", "coast", "all"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    for kind in (["forest", "river", "coast"] if args.kind == "all" else [args.kind]):
        print(ensure(kind, args.root), flush=True)

if __name__ == "__main__":
    main()
