"""Export the frozen thesis evidence without modifying training archives.

The author-side export reads the explicitly selected October 2026 archive. The
public release contains metrics, configurations, aggregate explanations, plots,
tables and hashes, not model weights or per-row predictions. ``--verify`` only
needs the exported release and the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DIRECTORIES = {
    "metrics", "figures", "tables", "provenance", "thresholds", "transfer",
    "controls", "interpretability",
}
SCHEMA_VERSION = 1


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def portable(value, root: Path):
    """Remove machine-specific prefixes while retaining logical archive IDs.

    Relative archive paths describe evidence retained by the author; they do not
    assert that omitted model/score files are distributed in the public release.
    """
    if isinstance(value, dict):
        return {portable(str(key), root): portable(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item, root) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "+Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if not isinstance(value, str):
        return value
    normal = value.replace("\\", "/")
    root_text = str(root.resolve()).replace("\\", "/").rstrip("/") + "/"
    normal = re.sub(re.escape(root_text), "", normal, flags=re.IGNORECASE)
    # External runtime locations are not useful reproducibility information.
    if re.search(r"(?i)(?:^|[\s\"'(])[a-z]:/", normal):
        return "[external local path omitted]"
    if normal.startswith("/") and not normal.startswith("//"):
        return "[external local path omitted]"
    return normal


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def source_path(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source is outside the project archive")
    return path


def checked_source(path: Path, expected: str | None = None) -> str:
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Pinned source hash mismatch: {path.name}")
    return actual


def recover_metrics(run: Path):
    """Recover precision from saved scores and authoritative confusion counts.

    No model is fitted, no threshold is selected on TEST, and historical rounded
    thresholds are never used to replace their original confusion counts.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    config = read_json(run / "config.json")
    metrics = read_json(run / "metrics_test.json")
    labels = np.load(run / "y_test.npy", allow_pickle=False)
    scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid saved TEST arrays")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("TEST labels must contain both binary classes")
    tp, fp, tn, fn = (int(metrics[name]) for name in ("TP", "FP", "TN", "FN"))
    if tp + fp + tn + fn != len(labels) or tp + fn != int(labels.sum()):
        raise ValueError("Confusion counts and TEST population differ")
    threshold = float(config.get("threshold_exact", metrics["threshold"]))
    if "threshold_exact" in config:
        prediction = scores >= threshold
        counts = (int(np.sum(prediction & (labels == 1))), int(np.sum(prediction & (labels == 0))),
                  int(np.sum(~prediction & (labels == 0))), int(np.sum(~prediction & (labels == 1))))
        if counts != (tp, fp, tn, fn):
            raise ValueError("Frozen exact threshold does not reproduce saved counts")
    full = dict(metrics)
    for name, value in (("PR-AUC", average_precision_score(labels, scores)),
                        ("ROC-AUC", roc_auc_score(labels, scores))):
        if not np.isclose(value, metrics[name], rtol=0, atol=1e-6):
            raise ValueError(f"Saved {name} differs from score evidence")
        full[name] = float(value)
    full["F1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    full["F2"] = 5 * tp / (5 * tp + fp + 4 * fn) if tp else 0.0
    full["precision"] = tp / (tp + fp) if tp + fp else 0.0
    full["recall"] = tp / (tp + fn)
    full["alert_rate"] = (tp + fp) / len(labels)
    full["FP/TP"] = fp / tp if tp else float("inf")
    full["threshold"] = threshold
    k = int(metrics["k_used"])
    if not 1 <= k <= len(labels):
        raise ValueError("Invalid saved workload budget")
    top_positive = int(labels[np.argsort(scores)[::-1][:k]].sum())
    full["precision_at_k"] = top_positive / k
    full["recall_at_k"] = top_positive / int(labels.sum())
    full["brier_score"] = None
    if config["model"] != "ocsvm":
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Supervised scores are outside [0, 1]")
        full["brier_score"] = float(brier_score_loss(labels, scores))
    full["test_samples"] = int(len(labels))
    full["test_fraud_samples"] = int(labels.sum())
    return full


def compact_curve(curve, max_points=3000):
    """Bound plot data size; reported AP is always computed from full scores."""
    if max_points < 2:
        raise ValueError("Plotting subset needs at least two endpoint slots")
    precision = curve.get("precision", curve.get("precisions"))
    recall = curve.get("recall", curve.get("recalls"))
    if precision is None or recall is None or len(precision) != len(recall) or not precision:
        raise ValueError("Invalid precision--recall coordinates")
    count = len(precision)
    indices = sorted({round(i * (count - 1) / (min(max_points, count) - 1))
                      for i in range(min(max_points, count))}) if count > 1 else [0]
    return {"precision": [precision[i] for i in indices],
            "recall": [recall[i] for i in indices],
            "original_points": count, "retained_points": len(indices),
            "representation": "uniform-index subset for plotting only; do not recompute AP from this subset"}


class ExportPlan:
    """Collect and validate every destination before writing any export file."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files: dict[str, bytes] = {}
        self.provenance: dict[str, dict] = {}

    def add(self, target: str, data: bytes, sources=(), transformation="byte-identical copy"):
        path = Path(target)
        if (path.is_absolute() or ".." in path.parts or len(path.parts) < 2
                or path.parts[0] not in ALLOWED_DIRECTORIES):
            raise ValueError("Export target is outside the public evidence directories")
        target = path.as_posix()
        if target in self.files:
            raise ValueError(f"Duplicate export target: {target}")
        self.files[target] = data
        self.provenance[target] = {
            "public_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "transformation": transformation,
            "sources": [{"archive_path": portable(str(p), self.root), "sha256": digest(p)}
                        for p in sources],
        }

    def add_json(self, target, data, sources=(), transformation="JSON: machine-specific paths normalised; numeric values preserved"):
        self.add(target, json_bytes(portable(data, self.root)), sources, transformation)

    def copy(self, target, source: Path, expected=None):
        checked_source(source, expected)
        self.add(target, source.read_bytes(), (source,))

    def write(self, output: Path):
        output = output.resolve()
        manifest_file = output / "provenance" / "release_manifest.json"
        previous = read_json(manifest_file) if manifest_file.is_file() else None
        old_files = previous.get("files", {}) if previous else {}
        # Never overwrite an unrelated file, including a previous release altered
        # by a human. Directories not owned by this plan remain untouched.
        for target in self.files:
            destination = output / target
            if target == "provenance/release_manifest.json":
                continue  # Its exporter identity is checked separately below.
            if destination.is_file():
                record = old_files.get(target)
                if not record or digest(destination) != record["public_sha256"]:
                    raise FileExistsError(f"Unowned or modified export destination: {target}")
        if manifest_file.exists() and (not previous or previous.get("exporter") != "export_public_evidence.py"):
            raise FileExistsError("Unowned release manifest")
        for target, data in self.files.items():
            destination = output / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        return manifest_file


def build_export(root: Path, revision: Path, thesis: Path):
    qa_file = revision / "derived" / "reporting" / "generation_qa.json"
    manifest_file = revision / "revision_manifest.json"
    qa, original = read_json(qa_file), read_json(manifest_file)
    checked_source(manifest_file, qa["manifest_sha256"])
    if qa["required_runs"] != 72 or len(qa["source_runs"]) != 72:
        raise ValueError("The complete 72-cell primary evidence grid is required")
    plan = ExportPlan(root)
    metrics, selections = {}, {}
    for key, record in sorted(qa["source_runs"].items()):
        run = source_path(record["run_dir"], root)
        selection = original.get("runs", {}).get(key)
        if selection is None:
            selection = original.get("historical_runs", {}).get(key)
        if (selection is None or source_path(selection["run_dir"], root) != run
                or selection["config_sha256"] != record["config_sha256"]):
            raise ValueError(f"Reporting source is not the manifest-pinned primary selection: {key}")
        for name, field in (("config.json", "config_sha256"), ("metrics_test.json", "metrics_sha256"),
                            ("metrics_cv.json", "metrics_cv_sha256"), ("y_test.npy", "y_test_sha256"),
                            ("y_test_scores.npy", "y_test_scores_sha256"),
                            ("pr_curve_data.json", "pr_curve_data_sha256")):
            checked_source(run / name, record[field])
        full = recover_metrics(run)
        if full["test_samples"] != record["test_samples"] or full["test_fraud_samples"] != record["test_fraud_samples"]:
            raise ValueError("Reporting QA and source TEST population differ")
        metrics[key] = full
        prefix = "metrics/" + key
        plan.add_json(prefix + "/config.json", read_json(run / "config.json"), (run / "config.json",))
        plan.add_json(prefix + "/metrics_saved.json", read_json(run / "metrics_test.json"), (run / "metrics_test.json",))
        plan.add_json(prefix + "/metrics_full_precision.json", full,
                      tuple(run / name for name in ("config.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy")),
                      "AP/ROC-AUC/Brier/top-k recomputed from full frozen TEST arrays; operating metrics from original integer counts; no fitting/TEST threshold selection")
        plan.add_json(prefix + "/metrics_cv.json", read_json(run / "metrics_cv.json"), (run / "metrics_cv.json",))
        if key.endswith("/none"):
            curve = compact_curve(read_json(run / "pr_curve_data.json"))
            curve["average_precision_full_scores"] = full["PR-AUC"]
            plan.add_json("metrics/curves/" + key.replace("/none", "") + ".json", curve,
                          (run / "pr_curve_data.json", run / "y_test.npy", run / "y_test_scores.npy"),
                          "At most 3000 original PR coordinates retained for plotting; AP comes from complete TEST scores")
        selections[key] = {**record, "archive_run_id": portable(str(run), root),
                           "public_config": "results/" + prefix + "/config.json",
                           "public_metrics": "results/" + prefix + "/metrics_full_precision.json",
                           "model_weights_and_row_predictions_distributed": False}
        selections[key].pop("run_dir", None)
    plan.add_json("metrics/primary_metrics.json", {"cells": metrics, "precision_basis": qa["metric_precision_basis"]}, (qa_file,))
    fields = ["cell", "PR-AUC", "ROC-AUC", "F1", "F2", "precision", "recall", "threshold", "brier_score",
              "TP", "FP", "TN", "FN", "alert_rate", "FP/TP", "precision_at_k", "recall_at_k", "k_used",
              "test_samples", "test_fraud_samples"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fields, lineterminator="\n")
    writer.writeheader()
    for key, cell in sorted(metrics.items()):
        writer.writerow({field: portable(key if field == "cell" else cell.get(field), root) for field in fields})
    plan.add("metrics/primary_metrics.csv", buffer.getvalue().encode(), (qa_file,), "Full-precision exported metric grid rendered as CSV")
    plan.add_json("provenance/selected_runs.json", selections, (qa_file, manifest_file))
    # Preserve original selection identities, including historical BAF final
    # models reused by validation-only recovery, without copying obsolete ULB.
    historical = {k: v for k, v in original.get("historical_runs", {}).items()
                  if k.startswith("baf_base/") and k.split("/")[1] != "fttransformer"}
    plan.add_json("provenance/historical_baf_selection.json", historical, (manifest_file,))
    for key, record in historical.items():
        run = source_path(record["run_dir"], root)
        checked_source(run / "config.json", record["config_sha256"])
        plan.add_json("provenance/historical_baf/" + key.replace("baf_base/", "") + "/config.json",
                      read_json(run / "config.json"), (run / "config.json",))
    plan.add_json("provenance/dataset_protocol.json", original["datasets"], (manifest_file,))
    plan.add_json("provenance/reporting_source_audit.json", qa, (qa_file,))
    for relative, expected in qa["output_sha256"].items():
        path = Path(relative.replace("\\", "/"))
        plan.copy(path.as_posix(), qa_file.parent / path, expected)
    for key, record in qa["threshold_studies"].items():
        path = source_path(record["path"], root)
        checked_source(path, record["sha256"])
        plan.add_json("thresholds/" + key + ".json", read_json(path), (path,))
    for model, record in qa["transfer_sources"].items():
        path = source_path(record["path"], root)
        checked_source(path, record["sha256"])
        plan.add_json("transfer/" + model + "/source_audit.json", read_json(path), (path,))
        for variant, variant_record in record["variants"].items():
            path = source_path(variant_record["path"], root)
            checked_source(path, variant_record["sha256"])
            plan.add_json("transfer/" + model + "/" + variant + "/metrics_test.json", read_json(path), (path,))
    partition = source_path(original["transfer_partitions"]["path"], root)
    checked_source(partition, original["transfer_partitions"]["sha256"])
    plan.add_json("transfer/partition_manifest.json", read_json(partition), (partition,))
    paired = revision / "derived" / "paired" / "paired_analysis.json"
    checked_source(paired, qa["paired_analysis"]["sha256"])
    for path in sorted(paired.parent.glob("*.json")):
        plan.add_json("controls/" + path.name, read_json(path), (path,))
    sensitivity_manifest = revision / "sensitivity" / "revision_manifest.json"
    checked_source(sensitivity_manifest, qa["paired_analysis"]["sensitivity_manifest"]["sha256"])
    plan.add_json("controls/source_manifest.json", read_json(sensitivity_manifest), (sensitivity_manifest,))
    plan.copy("tables/baf/sensitivity_controls.tex", revision / "derived" / "sensitivity_text" / "sensitivity_tables.tex")
    interp = revision / "derived" / "interpretability"
    explanation = read_json(interp / "interpretability_qa.json")
    # The complete attribution matrices remain local. Public aggregates preserve
    # all 30 original-feature importances, log-odds units, grouping and cases.
    explanation.pop("outputs", None)
    plan.add_json("interpretability/shap_summary.json", explanation, (interp / "interpretability_qa.json",))
    stability = interp / "variant_stability" / "shap_variant_stability.json"
    checked_source(stability, original["interpretability"]["variant_stability_sha256"])
    plan.add_json("interpretability/shap_variant_stability.json", read_json(stability), (stability,))
    attention_dir = interp / "attention"
    attention_file = attention_dir / "attention_summary.json"
    attention = read_json(attention_file)
    plan.add_json("interpretability/attention_summary.json", attention, (attention_file,))
    for path in sorted(attention_dir.glob("*.png")):
        plan.copy("figures/baf/" + path.name, path)
    for name in ("baf_absence_codes.json", "sampler_identity.json", "lgbm_score_diagnostics.json",
                 "execution_provenance_notes.json"):
        path = revision / "audit" / name
        plan.add_json("provenance/" + name, read_json(path), (path,))
    limits = (
        "# Evidence release\n\n"
        "This release accompanies the final October 2026 thesis. It contains the complete 72-cell primary metric grid, "
        "frozen configurations, validation summaries, 12 threshold studies, corrected BAF transfer metrics, predefined "
        "sensitivity controls, aggregate interpretability evidence, and thesis tables/figures.\n\n"
        "The original manifest and reporting QA are immutable author-side records; their SHA-256 identities are pinned "
        "in release_manifest.json. The original manifest's administrative completion flags were not rewritten. The release "
        "inventory instead verifies the selected completed evidence. Machine-specific path prefixes are normalised; both "
        "the original and public file hashes are recorded. Archive-relative paths refer to author-retained material, not "
        "files promised in this public release. Historical BAF final-model selection is explicit.\n\n"
        "Per-row TEST/validation predictions, model weights, raw attribution matrices, local logs, checkpoints and "
        "editorial records are not distributed here. Hashes identify that evidence but are not a substitute for its "
        "contents. Accordingly this bundle supports inspection and verification of reported metrics, tables, plots and "
        "provenance, not independent exact recomputation of historical fitted models from this bundle alone. Compact PR "
        "coordinates are for plotting only; reported average precision is recovered from complete frozen scores.\n\n"
        "Verify public file integrity locally with `python src/export_public_evidence.py --verify`. This is not a "
        "cross-computer reproducibility claim. The author-side export requires the retained local result archives.\n"
    )
    plan.add("provenance/README.md", limits.encode(), (), "Release scope and limitations")
    release = {
        "schema_version": SCHEMA_VERSION, "exporter": "export_public_evidence.py", "release_id": "thesis_20261007_v6",
        "evidence_status": "complete selected evidence, checked against frozen reporting QA",
        "original_manifest_sha256": digest(manifest_file), "original_reporting_qa_sha256": digest(qa_file),
        "thesis_pdf_sha256": digest(thesis), "thesis_public_path": "publications/thesis.pdf",
        "reference_commit": original["reference_commit"], "primary_cell_count": len(metrics),
        "threshold_study_count": len(qa["threshold_studies"]), "transfer_model_count": len(qa["transfer_sources"]),
        "reporting_output_count": len(qa["output_sha256"]),
        "omitted": ["raw datasets", "per-row scores/labels/indices", "weights/checkpoints", "raw SHAP matrices",
                    "editorial QA", "private writing sources", "temporary files", "locks/logs"],
        "normalisation": "Archive-relative source identifiers; original and public hashes differ for normalised JSON",
        "files": dict(sorted(plan.provenance.items())),
    }
    # The manifest deliberately does not hash itself.
    plan.files["provenance/release_manifest.json"] = json_bytes(release)
    return plan, release


def verify_primary_metrics(document, expected_cells: int):
    """Check the public aggregate independently of private score archives."""
    cells = document.get("cells", {})
    if expected_cells != 72 or len(cells) != expected_cells:
        raise ValueError("Public primary metric grid must contain all 72 cells")
    for key, cell in cells.items():
        names = ("TP", "FP", "TN", "FN", "test_samples", "test_fraud_samples")
        if any(type(cell.get(name)) is not int or cell[name] < 0 for name in names):
            raise ValueError(f"Invalid integer population/count evidence: {key}")
        tp, fp, tn, fn = (cell[name] for name in ("TP", "FP", "TN", "FN"))
        total, positives = cell["test_samples"], cell["test_fraud_samples"]
        if (total != tp + fp + tn + fn or positives != tp + fn
                or not 0 < positives < total):
            raise ValueError(f"Public confusion counts and TEST population differ: {key}")
        expected = {
            "F1": 2 * tp / (2 * tp + fp + fn),
            "F2": 5 * tp / (5 * tp + fp + 4 * fn),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / positives,
            "alert_rate": (tp + fp) / total,
        }
        for name, value in expected.items():
            actual = cell.get(name)
            if (type(actual) not in (int, float) or not math.isfinite(actual)
                    or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"Public count-derived {name} is inconsistent: {key}")
    return len(cells)


def verify_release(output: Path, *, project_root: Path | None = None):
    output = output.resolve()
    manifest = read_json(output / "provenance" / "release_manifest.json")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["exporter"] != "export_public_evidence.py":
        raise ValueError("Unsupported public release schema")
    for relative, record in manifest["files"].items():
        destination = (output / relative).resolve()
        if not destination.is_relative_to(output):
            raise ValueError("Release inventory path escapes output root")
        checked_source(destination, record["public_sha256"])
        if destination.stat().st_size != record["bytes"]:
            raise ValueError("Public file size changed")
    primary_path = output / "metrics" / "primary_metrics.json"
    arithmetic_cells = 0
    if "metrics/primary_metrics.json" in manifest["files"]:
        arithmetic_cells = verify_primary_metrics(read_json(primary_path), manifest["primary_cell_count"])
    thesis_verified = False
    if "thesis_public_path" in manifest or "thesis_pdf_sha256" in manifest:
        if project_root is None:
            raise ValueError("An explicit project root is required to verify the thesis PDF")
        project_root = project_root.resolve()
        thesis_reference = Path(manifest["thesis_public_path"])
        if thesis_reference.is_absolute():
            raise ValueError("The public thesis path must be project-relative")
        thesis = (project_root / thesis_reference).resolve()
        if not thesis.is_relative_to(project_root):
            raise ValueError("Public thesis path escapes the project root")
        checked_source(thesis, manifest["thesis_pdf_sha256"])
        thesis_verified = True
    return {"verified_files": len(manifest["files"]), "primary_cells": manifest["primary_cell_count"],
            "count_derived_cells_verified": arithmetic_cells, "thesis_pdf_verified": thesis_verified,
            "release_id": manifest["release_id"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--revision-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thesis", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / "results"
    if args.verify:
        print(json.dumps(verify_release(output, project_root=root), indent=2))
        return
    revision = args.revision_root or root / "results_revision" / "20261005"
    thesis = args.thesis or root / "deliverables" / "2026_10_07_Thesis_MSc_Olavo_V6.pdf"
    plan, release = build_export(root, revision, thesis)
    plan.write(output)
    verified = verify_release(output, project_root=root)
    print(json.dumps({**verified, "bytes": sum(len(data) for data in plan.files.values()),
                      "threshold_studies": release["threshold_study_count"],
                      "transfer_models": release["transfer_model_count"]}, indent=2))


if __name__ == "__main__":
    main()
