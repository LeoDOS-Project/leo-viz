import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from pyproj import Geod, Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform

from leo_aoi.demand import ingest
from leo_aoi.generate import GenerationConfig, generate
from leo_aoi.geometry import EARTH_RADIUS_KM, Grid
from leo_aoi.oriented import oriented_rectangle
from leo_aoi.output import Coverage, write_workload
from leo_aoi.preset import freeze


class OrientedTests(unittest.TestCase):
    def test_area_and_shape_at_high_latitudes_seams_and_poles(self):
        geod = Geod(a=EARTH_RADIUS_KM*1000, b=EARTH_RADIUS_KM*1000)
        for lon, lat in [(0, 0), (12, 60), (120, 80), (179.99, 45), (-179.99, -45), (30, 89.99), (30, -89.99)]:
            for angle in [0, 23, 90, 145]:
                aoi = oriented_rectangle('a', lon, lat, 100, None, 2, angle)
                polygon = shape(aoi.geometry)
                self.assertTrue(polygon.is_valid, (lon, lat, angle))
                self.assertTrue(polygon.covers(Point(lon, lat)), (lon, lat, angle))
                area, _ = geod.geometry_area_perimeter(polygon)
                self.assertAlmostEqual(abs(area)/1e6, 100, delta=.02)
                record = aoi.as_dict(None)
                self.assertAlmostEqual(record['width_km']*record['height_km'], 100)
                self.assertIsNone(record['tile_count'])
                self.assertGreaterEqual(polygon.bounds[0], -180)
                self.assertLessEqual(polygon.bounds[2], 180)

    def test_ground_dimensions_are_independent_of_latitude(self):
        # The original cylindrical grid stretched N-S rectangles at high latitude.
        for latitude in [0, 60, 80]:
            projection = Transformer.from_crs('EPSG:4326',
                f'+proj=laea +lat_0={latitude} +lon_0=12 +R={EARTH_RADIUS_KM*1000}',
                always_xy=True)
            for angle, expected in [(0, (20000, 5000)), (90, (5000, 20000))]:
                aoi = oriented_rectangle('a', 12, latitude, 100, None, 4, angle)
                bounds = transform(projection.transform, shape(aoi.geometry)).bounds
                self.assertAlmostEqual(bounds[2]-bounds[0], expected[0], delta=.001)
                self.assertAlmostEqual(bounds[3]-bounds[1], expected[1], delta=.001)
        # Vector geometry is independent of the legacy tile grid's minimum area.
        GenerationConfig(shape='oriented_rectangle', min_area_km2=.001)

    def test_independent_rotation_stream_and_replay(self):
        demand = ingest({'grid': {'columns': 36, 'rows': 18}, 'layers': [], 'background_fraction': 1})
        c = GenerationConfig(count=80, shape='oriented_rectangle', max_area_km2=1000)
        first = list(generate(demand, c))
        self.assertEqual(first, list(generate(demand, replace(c, count=100)))[:80])
        aligned = list(generate(demand, replace(c, random_rotation=False)))
        self.assertEqual([(a.anchor_lon, a.anchor_lat, a.requested_area_km2, a.aspect) for a in first],
                         [(a.anchor_lon, a.anchor_lat, a.requested_area_km2, a.aspect) for a in aligned])
        self.assertTrue(all(a.rotation_deg == 0 for a in aligned))
        self.assertGreater(np.std([a.rotation_deg for a in first]), 40)

    def test_vector_output_coverage_and_frozen_preset(self):
        demand = ingest({'grid': {'columns': 360, 'rows': 180}, 'layers': [], 'background_fraction': 1})
        config = GenerationConfig(count=5, shape='oriented_rectangle', max_area_km2=1000)
        aoi = oriented_rectangle('a', 179.99, 0, 100, None, 1, 45)
        coverage = Coverage(demand.grid)
        coverage.add(aoi, config.grid, 0)
        self.assertEqual(coverage.counts.max(), 1)
        self.assertGreater(coverage.counts[:, 0].sum(), 0)
        self.assertGreater(coverage.counts[:, -1].sum(), 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demand.save(root/'demand.npz')
            summary = write_workload(demand, config, root/'out', root/'demand.npz', geojson=True)
            self.assertIsNone(summary['total_tile_memberships'])
            rows = [json.loads(line) for line in (root/'out/aois.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertTrue(all('geometry' in row for row in rows))
            manifest = freeze({'schema_version':1, 'mean_area_km2':100, 'size_ratio':100,
                'generation':{'count':5, 'shape':'oriented_rectangle'}, 'demand_path':str(root/'demand.npz')},
                root/'preset.json.gz', 'Example')
            self.assertIsNone(manifest['summary']['total_tile_memberships'])


if __name__ == '__main__':
    unittest.main()
