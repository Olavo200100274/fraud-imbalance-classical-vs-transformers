"""Portable article evidence guards; no training or private archive required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import article_release as release


class ArticleReleaseTests(unittest.TestCase):
    def test_relative_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            for unsafe in ("../secret", "a/../../secret", str(Path(temporary).resolve())):
                with self.assertRaises(ValueError):
                    release.safe_path(temporary, unsafe)

    def test_changed_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "results/example.json"
            payload.parent.mkdir()
            payload.write_text("{}", encoding="utf-8")
            record = {"sha256": release.digest(payload), "bytes": payload.stat().st_size}
            keys = [f"{dataset}/{model}/{strategy}" for dataset in ("ulb_2013", "baf_base")
                    for model in ("logreg", "rf", "lgbm", "catboost", "fttransformer")
                    for strategy in ("none", "rus", "ros", "smote", "smote_tomek", "smoteenn", "weights")]
            keys += ["ulb_2013/ocsvm/none", "baf_base/ocsvm/none"]
            primary = {key: {"archive": "results/example.json", "metrics": "results/example.json", "config": "results/example.json"} for key in keys}
            derived = {str(index): {"archive": "results/example.json", "metrics": "results/example.json"} for index in range(30)}
            manifest = {"schema": "dedicated_article_evidence_v1", "article": 1, "primary_runs": primary,
                        "control_runs": dict(list(derived.items())[:8]), "transfer_runs": derived, "threshold_studies": {},
                        "files": {"results/example.json": record}}
            target = root / release.MANIFEST
            target.parent.mkdir()
            target.write_bytes(release.json_bytes(manifest))
            self.assertEqual(release.verify(root)["article"], 1)
            payload.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                release.verify(root)

    def test_frozen_metrics_and_infeasible_threshold(self):
        import numpy as np
        labels, scores = np.array([0, 1, 1, 0]), np.array([.1, .8, .6, .3])
        values = release.frozen_metrics(labels, scores, .5)
        self.assertEqual((values["TP"], values["FP"], values["FN"], values["TN"]), (2, 0, 0, 2))
        self.assertEqual(values["F2"], 1.)
        none = release.frozen_metrics(labels, scores, float("inf"))
        self.assertIsNone(none["threshold"])
        self.assertEqual(none["TP"] + none["FP"], 0)
        self.assertIsNotNone(json.loads(release.json_bytes(none)))


if __name__ == "__main__":
    unittest.main()
