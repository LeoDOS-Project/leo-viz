import gzip
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from leo_aoi.preset import freeze
from leo_aoi.viewer import write_preview


class PresetTests(unittest.TestCase):
    def test_lossless_reproducible_archive_and_no_overwrite(self):
        request = {"schema_version": 1, "demand_path": None,
                   "mean_area_km2": 1000, "size_ratio": 1000,
                   "generation": {"count": 20, "seed": 42, "shape": "rectangle", "hotspot_count": 6,
                                  "hotspot_fraction": 0.4, "hotspot_spread_km": 100}}
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "a.json.gz"
            b = Path(directory) / "b.json.gz"
            manifest = freeze(request, a, "Example")
            freeze(request, b, "Example")
            self.assertEqual(a.read_bytes(), b.read_bytes())
            expected = StringIO()
            write_preview(request, expected)
            self.assertEqual(gzip.decompress(a.read_bytes()).decode(), expected.getvalue())
            actual = json.loads(gzip.decompress(a.read_bytes()))
            self.assertEqual(len(actual['aois']), 20)
            self.assertIn('rectangles', actual['aois'][0])
            self.assertIn('tile_count', actual['aois'][0])
            self.assertIn('geometry', actual['aois'][0])
            self.assertEqual(manifest['summary']['count'], 20)
            self.assertEqual(manifest['name'], "Example")
            self.assertEqual(json.loads(a.with_suffix('.manifest.json').read_text()), manifest)
            before = a.read_bytes()
            with self.assertRaises(ValueError): freeze(request, a, "Replacement")
            self.assertEqual(a.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
