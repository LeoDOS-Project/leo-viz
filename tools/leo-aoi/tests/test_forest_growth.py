"""Shape controls must change geometry without losing area, connectivity or replay."""
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from shapely.geometry import box, shape
from shapely.ops import unary_union

from leo_aoi.environment import Config, Forest
from leo_aoi.environment_data import save
from leo_aoi.forest_growth import grow
from leo_aoi.viewer import write_preview


class ForestGrowthTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((1800, 3600), dtype=bool)
        self.config = Config(mean_area_km2=100000, size_ratio=1)
        self.areas = Forest({'mask': self.mask}, self.config).area

    def sample(self, config=None, row=900, col=1800, seed=42):
        return grow(self.mask, self.areas, row, col, .1, 100000,
                    np.random.default_rng(seed), config or self.config)

    @staticmethod
    def polygon(cells):
        return unary_union([box(x, y, x+1, y+1) for y, x in cells])

    def test_connected_area_bounded_growth_including_dateline_and_high_latitude(self):
        for row, col in [(900, 1800), (900, 0), (100, 1800), (1699, 3599)]:
            cells, extra = self.sample(row=row, col=col)
            pending, remaining = [cells[0]], set(cells[1:])
            while pending:
                y, x = pending.pop()
                for cell in [(y-1, x), (y+1, x), (y, (x-1)%3600), (y, (x+1)%3600)]:
                    if cell in remaining:
                        remaining.remove(cell); pending.append(cell)
            self.assertFalse(remaining)
            self.assertIn((row, col), cells)
            self.assertGreaterEqual(extra['growth_area_km2'], 100000)
            self.assertLess(extra['growth_area_km2'], 100000 + max(self.areas[y] for y, x in cells))
            self.assertEqual(len(cells), len(set(cells)))

    def test_outline_variation_creates_large_scale_concavities(self):
        regular = replace(self.config, max_elongation=1, outline_variation=0, forest_adherence=0)
        base = self.polygon(self.sample(regular)[0])
        self.assertGreater(base.area/base.convex_hull.area, .9)
        solidity = []
        for seed in range(6):
            p = self.polygon(self.sample(replace(regular, outline_variation=.7), seed=seed)[0])
            self.assertTrue(p.is_valid)
            self.assertEqual(p.geom_type, 'Polygon')
            solidity.append(p.area/p.convex_hull.area)
        self.assertLess(np.mean(solidity), .8)

    def test_elongation_changes_ground_aspect_and_randomizes_direction(self):
        config = replace(self.config, max_elongation=6, outline_variation=0, forest_adherence=0)
        bearings = []
        for seed in range(5):
            cells, extra = self.sample(config, seed=seed)
            bounds = self.polygon(cells).minimum_rotated_rectangle
            xy = np.asarray(bounds.exterior.coords)
            lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            self.assertAlmostEqual(max(lengths)/min(lengths), extra['growth_elongation'], delta=.5)
            bearings.append(extra['growth_bearing_deg'])
        self.assertGreater(np.std(bearings), 30)

    def test_adherence_follows_connected_forest_and_preserves_clearings(self):
        self.mask[:] = False
        self.mask[890:911, 1700:1900] = True
        regular = replace(self.config, max_elongation=1, outline_variation=0)
        _, low = self.sample(replace(regular, forest_adherence=0))
        _, high = self.sample(replace(regular, forest_adherence=1))
        self.assertGreater(high['tree_cover_area_fraction'], low['tree_cover_area_fraction']+.15)
        self.mask[:] = True
        self.mask[897:903, 1805:1811] = False
        cells, _ = self.sample(replace(regular, forest_adherence=1))
        polygon = self.polygon(cells)
        self.assertGreater(len(polygon.interiors), 0)
        self.assertTrue(set(cells).isdisjoint((y, x) for y in range(897,903) for x in range(1805,1811)))

    def test_forest_viewer_replay_and_shape_controls_preserve_locations_and_areas(self):
        with tempfile.TemporaryDirectory() as directory:
            save(Path(directory)/'forest.npz', {'schema_version':1, 'kind':'forest'}, mask=self.mask)
            config = {'kind':'forest', 'count':5, 'seed':42, 'mean_area_km2':10000,
                      'size_ratio':10, 'data_dir':directory, 'tile_edge_km':1}
            def preview(**changes):
                out = StringIO()
                write_preview({'schema_version':1, 'environment':{**config, **changes}}, out)
                return json.loads(out.getvalue())['aois']
            first = preview()
            self.assertEqual(first, preview())
            self.assertEqual(first, preview(count=7)[:5])
            changed = preview(max_elongation=8, outline_variation=1, forest_adherence=0)
            fields = ('id','anchor_lon','anchor_lat','requested_area_km2')
            self.assertEqual([[a[k] for k in fields] for a in first], [[a[k] for k in fields] for a in changed])
            self.assertNotEqual([a['geometry'] for a in first], [a['geometry'] for a in changed])
            self.assertTrue(all(shape(a['geometry']).is_valid for a in first))

    def test_invalid_shape_parameters(self):
        for field, value in [('max_elongation', .5), ('outline_variation', 1.1),
                             ('forest_adherence', -.1), ('max_elongation', float('nan'))]:
            with self.assertRaises(ValueError):
                Config(**{field:value})


if __name__ == '__main__':
    unittest.main()
