from dataclasses import replace
import json
from math import pi
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import shape

from leo_aoi.demand import DemandField, ingest
from leo_aoi.generate import GenerationConfig, generate, mosaic, rectangle
from leo_aoi.geometry import Aoi, Grid, TileRect
from leo_aoi.output import Coverage, write_workload


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def raster(self, name, values, grid, nodata=-9999):
        path = self.base / name
        with rasterio.open(path, "w", driver="GTiff", width=grid.columns, height=grid.rows,
                           count=1, dtype="float64", crs=grid.crs, nodata=nodata,
                           transform=from_bounds(*grid.bounds_m, grid.columns, grid.rows)) as dst:
            dst.write(np.asarray(values, dtype="float64"), 1)
        return name

    def uniform(self):
        return ingest({"grid": {"columns": 72, "rows": 36}, "layers": []})

    def test_sphere_area_and_coordinate_round_trip(self):
        grid = Grid.with_tile_edge(.1)
        self.assertAlmostEqual(grid.cell_area_km2 * grid.columns * grid.rows, 4*pi*6371**2, delta=1e-6)
        for lon, lat in [(11.97, 57.7), (179.99, 85), (-179.99, -85), (0, 0)]:
            x, y = grid.cell(lon, lat)
            west, north = grid.lonlat(x, y)
            east, south = grid.lonlat(x+1, y+1)
            self.assertTrue(west <= lon <= east and south <= lat <= north)

    def test_mass_resampling_conserves_total_and_distribution(self):
        grid = Grid(4, 2)
        data = [[1, 2, -9999, 0], [3, 4, 0, 0]]
        name = self.raster("mass.tif", data, grid)
        for cols, rows in [(2, 1), (4, 2), (8, 4)]:
            field = ingest({"grid": {"columns": cols, "rows": rows},
                            "layers": [{"path": name, "kind": "mass"}]}, self.base)
            self.assertAlmostEqual(field.provenance["sources"][0]["resampled_total"], 10, places=5)
            self.assertAlmostEqual(field.probability[:, :cols//2].sum(), 1)

    def test_normalized_mixture_and_mask(self):
        grid = Grid(4, 2)
        a = self.raster("a.tif", [[1000, 0, 0, 0], [0, 0, 0, 0]], grid)
        b = self.raster("b.tif", [[0, 1, 0, 0], [0, 0, 0, 0]], grid)
        mask = self.raster("mask.tif", [[1, 1, 0, 0], [0, 0, 0, 0]], grid)
        f = ingest({"grid": {"columns": 4, "rows": 2}, "layers": [
            {"path": a, "weight": 3}, {"path": b, "weight": 1}],
            "mask": {"path": mask}, "background_fraction": .2}, self.base)
        np.testing.assert_allclose(f.probability[0], [.7, .3, 0, 0])
        self.assertEqual(f.probability[1].sum(), 0)
        f.save(self.base / "field.npz")
        np.testing.assert_array_equal(f.probability, DemandField.load(self.base / "field.npz").probability)

    def test_density_is_ground_area_weighted(self):
        path = self.base / "density.tif"
        with rasterio.open(path, "w", driver="GTiff", width=36, height=18, count=1,
                           dtype="float64", crs="EPSG:4326",
                           transform=from_bounds(-180, -90, 180, 90, 36, 18)) as dst:
            dst.write(np.ones((18, 36)), 1)
        f = ingest({"grid": {"columns": 36, "rows": 18},
                    "layers": [{"path": str(path), "kind": "density"}]})
        np.testing.assert_allclose(f.probability, 1/(36*18), atol=1e-12)

    def test_replay_prefix_and_independent_shape_stream(self):
        demand = self.uniform()
        c = GenerationConfig(count=25, hotspot_count=4, max_area_km2=1000)
        first = list(generate(demand, c))
        self.assertEqual(first, list(generate(demand, c)))
        self.assertEqual(first, list(generate(demand, replace(c, count=50)))[:25])
        changed = list(generate(demand, replace(c, max_area_km2=10000)))
        self.assertEqual([(a.anchor_lon, a.anchor_lat) for a in first],
                         [(a.anchor_lon, a.anchor_lat) for a in changed])

    def test_small_mosaics_have_no_duplicate_tiles_and_are_connected(self):
        grid, rng = Grid(360, 180), np.random.default_rng(1)
        for n in [1, 2, 7, 35, 250, 999]:
            for lon in [-179.9, 0, 179.9]:
                for lat in [-89.9, 0, 89.9]:
                    rects = mosaic(grid, lon, lat, n, 1, 6, rng)
                    aoi = Aoi("test", lon, lat, n*grid.cell_area_km2, None, rects)
                    tiles = list(aoi.tiles())
                    self.assertEqual(len(tiles), n)
                    self.assertEqual(len(set(tiles)), n)
                    remaining, todo = set(tiles), [tiles[0]]
                    while todo:
                        x, y = todo.pop()
                        if (x, y) not in remaining:
                            continue
                        remaining.remove((x, y))
                        todo.extend([((x-1) % grid.columns, y), ((x+1) % grid.columns, y),
                                     (x, y-1), (x, y+1)])
                    self.assertFalse(remaining)
                    self.assertIn(grid.cell(lon, lat), set(tiles))
                    self.assertTrue(shape(aoi.geojson(grid)["geometry"]).is_valid)

    def test_rectangles_have_no_steps_and_preserve_anchors_at_seams_and_poles(self):
        grid = Grid(360, 180)
        for n in [1, 7, 997, 10000]:
            for lon in [-179.9, 0, 179.9]:
                for lat in [-89.9, 0, 89.9]:
                    for aspect in [.5, 1, 2]:
                        rects = rectangle(grid, lon, lat, n, aspect)
                        aoi = Aoi("test", lon, lat, n * grid.cell_area_km2, None, rects)
                        self.assertLessEqual(len(rects), 2)
                        self.assertEqual(len({(r.row, r.height) for r in rects}), 1)
                        if len(rects) == 2:
                            self.assertEqual(rects[0].col + rects[0].width, grid.columns)
                            self.assertEqual(rects[1].col, 0)
                        tiles = set(aoi.tiles())
                        self.assertEqual(len(tiles), aoi.tile_count)
                        self.assertIn(grid.cell(lon, lat), tiles)
                        self.assertLessEqual(abs(aoi.tile_count - n), rects[0].height / 2)
                        geometry = shape(aoi.geojson(grid)["geometry"])
                        self.assertTrue(geometry.is_valid)
                        pieces = [geometry] if geometry.geom_type == "Polygon" else geometry.geoms
                        for piece in pieces:
                            self.assertEqual(len(piece.exterior.coords), 5)

    def test_rectangle_mode_replays_and_keeps_location_sampling_independent(self):
        demand = self.uniform()
        config = GenerationConfig(count=25, hotspot_count=4, max_area_km2=1000,
                                  shape="rectangle")
        aois = list(generate(demand, config))
        self.assertEqual(aois, list(generate(demand, config)))
        self.assertEqual(aois, list(generate(demand, replace(config, count=50)))[:25])
        mosaics = list(generate(demand, replace(config, shape="mosaic")))
        self.assertEqual([(a.anchor_lon, a.anchor_lat, a.hotspot) for a in aois],
                         [(a.anchor_lon, a.anchor_lat, a.hotspot) for a in mosaics])
        overlap = list(generate(demand, replace(config, hotspot_count=1, hotspot_spread_km=0,
                        min_area_km2=10, max_area_km2=10, min_aspect=1, max_aspect=1)))
        self.assertTrue(all(a.rectangles == overlap[0].rectangles for a in overlap))

    def test_continental_scale_is_compact(self):
        grid = Grid.with_tile_edge(.1)
        c = GenerationConfig(count=1, min_area_km2=5e6, max_area_km2=5e6)
        aoi = next(generate(self.uniform(), c))
        self.assertGreater(aoi.tile_count, 400_000_000)
        self.assertLessEqual(len(aoi.rectangles), c.max_bands * 4)
        self.assertLessEqual(abs(aoi.tile_count * grid.cell_area_km2 - 5e6), grid.cell_area_km2 / 2)

    def test_hotspots_preserve_overlap(self):
        c = GenerationConfig(count=10, hotspot_count=1, hotspot_spread_km=0,
                             min_area_km2=10, max_area_km2=10, max_bands=1, min_aspect=1, max_aspect=1)
        aois = list(generate(self.uniform(), c))
        self.assertTrue(all(a.rectangles == aois[0].rectangles for a in aois))

    def test_coverage_counts_each_aoi_once(self):
        grid = Grid(10, 10)
        aoi = Aoi("a", 0, 0, 1, None, (TileRect(0, 0, 1, 1), TileRect(1, 0, 1, 1)))
        cov = Coverage(Grid(2, 2))
        cov.add(aoi, grid, 0)
        self.assertEqual(cov.counts.max(), 1)
        cov.add(aoi, grid, 1)
        self.assertEqual(cov.counts.max(), 2)

    def test_exact_overlap_without_expanding_continental_tiles(self):
        a = Aoi("a", 0, 0, 1, None, (TileRect(0, 0, 100000, 100000),))
        b = Aoi("b", 0, 0, 1, None, (TileRect(50000, 50000, 100000, 100000),))
        self.assertEqual(a.shared_tiles(b), 2_500_000_000)

    def test_statistical_weights_and_mask_apply_to_anchors(self):
        g = Grid(4, 2)
        p = np.array([[.75, .25, 0, 0], [0, 0, 0, 0]])
        f = DemandField(g, p, p > 0, {})
        c = GenerationConfig(count=4000, min_area_km2=1, max_area_km2=1)
        locations = [g.cell(a.anchor_lon, a.anchor_lat) for a in generate(f, c)]
        self.assertTrue(all(y == 0 and x in (0, 1) for x, y in locations))
        fraction = sum(x == 0 for x, _ in locations) / len(locations)
        self.assertAlmostEqual(fraction, .75, delta=.025)

    def test_end_to_end_output_and_no_overwrite(self):
        field = self.uniform()
        path = self.base / "demand.npz"
        field.save(path)
        c = GenerationConfig(count=12)
        dest = self.base / "run"
        result = write_workload(field, c, dest, path, geojson=True)
        self.assertEqual(result["aoi_count"], 12)
        self.assertEqual(len((dest / "aois.jsonl").read_text().splitlines()), 12)
        self.assertEqual(len(json.loads((dest / "aois.geojson").read_text())["features"]), 12)
        self.assertIn("demand_sha256", json.loads((dest / "manifest.json").read_text()))
        with self.assertRaises(ValueError):
            write_workload(field, c, dest, path)

    def test_invalid_inputs_fail_clearly(self):
        for config in [{"count": 0}, {"seed": -1}, {"max_bands": 0}, {"hotspot_fraction": 2},
                       {"min_area_km2": 0}, {"tile_edge_km": float("nan")}, {"shape": "unknown"}, {"mispelled": 1}]:
            with self.assertRaises((ValueError, TypeError)):
                GenerationConfig.from_dict(config)
        with self.assertRaises(ValueError):
            ingest({"background_fraction": .5})
        grid = Grid(2, 1)
        empty = self.raster("empty.tif", [[0, -9999]], grid)
        with self.assertRaises(ValueError):
            ingest({"layers": [{"path": empty}]}, self.base)


if __name__ == "__main__":
    unittest.main()
