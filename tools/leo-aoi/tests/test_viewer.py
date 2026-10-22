from io import StringIO
import json
import math
from pathlib import Path
import tempfile
import unittest

from leo_aoi.generate import generate
from leo_aoi.viewer import area_bounds, prepare, write_preview


class ViewerTests(unittest.TestCase):
    def request(self):
        return {"schema_version": 1, "demand_path": None,
                "mean_area_km2": 1000, "size_ratio": 100,
                "generation": {"count": 25, "seed": 42, "shape": "rectangle"}}

    def test_expected_mean_and_degenerate_range(self):
        for mean, ratio in [(1, 1), (100, 1.00001), (1000, 100), (1e6, 10000)]:
            lo, hi = area_bounds(mean, ratio)
            expected = lo if ratio == 1 else (hi - lo) / math.log(hi / lo)
            self.assertAlmostEqual(expected / mean, 1, places=9)
        for mean, ratio in [(0, 1), (1, 0.5), (float('nan'), 2)]:
            with self.assertRaises(ValueError): area_bounds(mean, ratio)

    def test_preview_matches_batch_geometry_and_is_reproducible(self):
        request = self.request()
        demand, config = prepare(request)
        out = StringIO()
        write_preview(request, out)
        document = json.loads(out.getvalue())
        self.assertEqual(document['config']['count'], 25)
        for actual, aoi in zip(document['aois'], generate(demand, config), strict=True):
            expected = aoi.as_dict(config.grid)
            self.assertEqual(actual['geometry'], json.loads(json.dumps(aoi.geojson(config.grid)['geometry'])))
            self.assertEqual({k: v for k, v in actual.items() if k != 'geometry'}, expected)
        replay = StringIO()
        write_preview(request, replay)
        self.assertEqual(out.getvalue(), replay.getvalue())

    def test_invalid_request_fails_before_emitting_partial_output(self):
        for changes in [{'generation': {'count': 50001}}, {'demand_path': '/nonexistent/demand.npz'},
                        {'size_ratio': 10000, 'mean_area_km2': 1}, {'unknown': True}]:
            out = StringIO()
            with self.assertRaises(ValueError): write_preview({**self.request(), **changes}, out)
            self.assertEqual(out.getvalue(), '')

    def test_custom_demand_field_uses_existing_ingestion(self):
        demand, _ = prepare(self.request())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.npz'
            demand.save(path)
            loaded, _ = prepare({**self.request(), 'demand_path': str(path)})
            self.assertEqual(loaded.provenance, demand.provenance)


if __name__ == '__main__':
    unittest.main()
