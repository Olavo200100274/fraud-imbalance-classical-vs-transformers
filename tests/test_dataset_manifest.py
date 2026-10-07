"""Small checks for the standard-library raw-input verifier."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verify_datasets import verify


class DatasetManifestTests(unittest.TestCase):
    def test_exact_bytes_and_missing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"a,b\n1,2\n"
            path = root / "fixture.csv"
            path.write_bytes(content)
            manifest = {"files": [{"name": "fixture.csv", "bytes": len(content),
                                   "sha256": hashlib.sha256(content).hexdigest()}]}
            self.assertEqual(verify(root, manifest), [])
            path.write_bytes(b"a,b\n2,1\n")
            self.assertIn("SHA-256 mismatch", verify(root, manifest)[0])
            self.assertIn("missing", verify(root / "absent", manifest)[0])

    def test_unknown_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            verify(Path.cwd(), {"files": []}, ["unknown.csv"])


if __name__ == "__main__":
    unittest.main()
