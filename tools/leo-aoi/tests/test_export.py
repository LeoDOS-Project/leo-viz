import base64
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from leo_aoi.export import export


class ExportTests(unittest.TestCase):
    def round_trip(self, metadata, **arrays):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.npz"
            target = Path(directory) / "data.json.gz"
            np.savez_compressed(source, metadata=json.dumps(metadata), **arrays)
            export(source, target)
            first = target.read_bytes()
            export(source, target)
            self.assertEqual(first, target.read_bytes())
            return json.loads(gzip.decompress(first))

    def test_demand_preserves_weights_and_mask(self):
        data = self.round_trip({"schema_version": 1}, probability=np.array([[0.0, 1.0]]),
                               eligible=np.array([[False, True]]))
        self.assertEqual(data["cells"], [[1, 1.0]])
        self.assertEqual(base64.b64decode(data["eligible"]), b"\x02")

    def test_forest_packs_in_little_bit_order(self):
        data = self.round_trip({"schema_version": 1, "kind": "forest"},
                               mask=np.array([[True, False, True, False]]))
        self.assertEqual(base64.b64decode(data["mask"]), b"\x05")
        self.assertEqual((data["columns"], data["rows"]), (4, 1))

    def test_network_preserves_connections_and_ids(self):
        for kind in ("river", "coast"):
            data = self.round_trip({"schema_version": 1, "kind": kind},
                                   points=np.array([[0., 0.], [1., 0.]]), offsets=np.array([0, 2]),
                                   next_index=np.array([-1]), length=np.array([111.]),
                                   ids=np.array([9]), flow=np.array([100.]))
            self.assertEqual(data["kind"], kind)
            self.assertEqual(np.frombuffer(base64.b64decode(data["network"]["next_index"]), dtype="<i4").tolist(), [-1])
            self.assertEqual(np.frombuffer(base64.b64decode(data["network"]["ids"]), dtype="<i8").tolist(), [9])

    def test_schema_is_checked(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            self.round_trip({"schema_version": 2})
