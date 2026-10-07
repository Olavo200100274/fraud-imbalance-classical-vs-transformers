"""Bounded, immutable compatibility evidence for the first revision FT run.

This is not a default for absent metadata. Only the exact already-completed
baseline below is eligible. A separate report independently reproduces its
saved intervals with the prescribed bootstrap; original artefacts are untouched.
Matching rounded intervals do not uniquely establish the original iteration
count, and no retrospective launch snapshot is claimed.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from experiment_protocol import PROTOCOL_VERSION, file_sha256

ROOT = Path(__file__).resolve().parents[1]
KEY = "ulb_2013/fttransformer/none"
RUN_NAME = "run_20261006_040431_457756"
CONFIG_SHA256 = "93181c268ef56c52e491328f1642c2801d47682e00e7e17ef388badcd8f1d551"
METRICS_SHA256 = "37daa037da9791c25503e4cb04726f8185f458f0eaa56a35dbf4d29e149926d7"
SOURCE_FILES = ("config.json", "completed.json", "metrics_test.json", "metrics_cv.json",
                "y_test.npy", "y_test_scores.npy", "test_row_indices.npy", "dev_row_indices.npy",
                "y_val.npy", "y_val_scores.npy", "validation_row_indices.npy", "validation_fold_ids.npy",
                "optuna_trials.json", "model.pt", "preprocessors.joblib", "validation_model.pt",
                "validation_preprocessors.joblib")
CI_KEYS = ("PR-AUC_ci", "ROC-AUC_ci", "F2_ci")
ROLE = "independent_prescribed_bootstrap_verification_for_missing_legacy_metadata"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _check_identity(manifest, run, config, expected_iterations):
    run = _resolve(run)
    reference = manifest.get("runs", {}).get(KEY, {})
    if (expected_iterations != 1000 or run.name != RUN_NAME
            or (config.get("dataset"), config.get("model"), config.get("strategy"))
            != ("ulb_2013", "fttransformer", "none")
            or config.get("protocol_version") != PROTOCOL_VERSION
            or "bootstrap_iterations" in config):
        raise ValueError("Bootstrap compatibility is restricted to the known missing-field baseline.")
    if (not reference.get("run_dir") or _resolve(reference["run_dir"]) != run
            or reference.get("config_sha256") != CONFIG_SHA256
            or file_sha256(run / "config.json") != CONFIG_SHA256
            or _read(run / "config.json") != config):
        raise ValueError("Bootstrap compatibility does not identify the unchanged pinned legacy configuration.")
    if (config.get("sample_fraction") is not None or config.get("n_trials") != 50
            or config.get("split_seed") != 42 or config.get("split_ratio") != "80/20 stratified"
            or config.get("missing_policy") != "preserve"
            or config.get("max_epochs") != 200 or config.get("scheduler_horizon") != 200
            or config.get("early_stopping_patience") != 15
            or not 1 <= config.get("best_epoch", 0) <= 200
            or not np.isfinite(config.get("threshold_exact", np.nan))):
        raise ValueError("The legacy run is not the prescribed full-data baseline.")
    trials = _read(run / "optuna_trials.json")
    if len(trials) != 50 or any(trial["state"] not in ("COMPLETE", "PRUNED", "FAIL") for trial in trials):
        raise ValueError("The legacy run lacks its terminal HPO evidence.")
    completion = _read(run / "completed.json")
    if completion.get("status") != "complete" or completion.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("The legacy completion marker is invalid.")
    metrics_source = ROOT / "src/evaluation/metrics.py"
    if (file_sha256(metrics_source) != METRICS_SHA256
            or config.get("source_files_sha256", {}).get("src\\evaluation\\metrics.py") != METRICS_SHA256):
        raise ValueError("The prescribed bootstrap implementation no longer matches the recorded source hash.")
    return run


def source_hashes(run):
    return {name: file_sha256(run / name) for name in SOURCE_FILES}


def _check_intervals(intervals):
    if not isinstance(intervals, dict) or set(intervals) != set(CI_KEYS):
        raise ValueError("All three independently verified confidence intervals are required.")
    for bounds in intervals.values():
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or not np.isfinite(bounds).all() or not 0 <= bounds[0] <= bounds[1] <= 1):
            raise ValueError("Malformed bootstrap interval evidence.")


def verify_bootstrap_compatibility(manifest, run, config, expected_iterations=1000):
    """Verify the pinned sidecar without recomputing bootstrap on every poll."""
    run = _check_identity(manifest, run, config, expected_iterations)
    reference = manifest.get("bootstrap_compatibility", {}).get(KEY)
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Missing registered bootstrap compatibility verification.")
    path = _resolve(reference["path"])
    if not path.is_file() or file_sha256(path) != reference["sha256"]:
        raise ValueError("The bootstrap compatibility report has changed or is missing.")
    report = _read(path)
    if (report.get("status") != "complete" or report.get("role") != ROLE
            or report.get("protocol_version") != PROTOCOL_VERSION or report.get("key") != KEY
            or _resolve(report.get("run_dir", "")) != run
            or report.get("config_sha256") != CONFIG_SHA256
            or report.get("original_bootstrap_field") != "absent"
            or report.get("verified_iterations") != expected_iterations or report.get("random_state") != 42
            or report.get("confidence_level") != 0.95
            or report.get("threshold_exact") != config["threshold_exact"]
            or report.get("metrics_source_sha256") != METRICS_SHA256
            or report.get("original_launch_snapshot_available") is not False
            or report.get("interval_match_proves_original_iteration_count") is not False
            or report.get("original_artefacts_modified") is not False):
        raise ValueError("The bootstrap compatibility report has inconsistent identity or scope.")
    if report.get("source_artefacts_sha256") != source_hashes(run):
        raise ValueError("A bootstrap compatibility source artefact has changed.")
    stored = _read(run / "metrics_test.json").get("bootstrap_ci")
    _check_intervals(stored)
    _check_intervals(report.get("recomputed_intervals"))
    if report.get("stored_intervals") != stored or report["recomputed_intervals"] != stored:
        raise ValueError("The prescribed bootstrap did not reproduce all saved intervals exactly.")
    return {"path": str(path), "sha256": reference["sha256"]}


def produce_and_register(manifest_path, output):
    """Independently check saved scores, then register only new sidecar metadata."""
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if (not manifest_path.is_relative_to(ROOT / "results_revision")
            or not output.is_relative_to(manifest_path.parent / "audit")
            or output.exists()):
        raise ValueError("Use a new isolated audit report; never overwrite existing evidence.")
    manifest = _read(manifest_path)
    config_path = _resolve(manifest["runs"][KEY]["run_dir"]) / "config.json"
    config = _read(config_path)
    run = _check_identity(manifest, config_path.parent, config, 1000)
    if KEY in manifest.get("bootstrap_compatibility", {}):
        raise ValueError("A compatibility report is already registered; verify it rather than replace it.")
    before = source_hashes(run)
    stored = _read(run / "metrics_test.json")["bootstrap_ci"]
    _check_intervals(stored)
    labels = np.load(run / "y_test.npy", allow_pickle=False)
    scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
    if (labels.ndim != 1 or scores.shape != labels.shape or len(labels) != config["test_samples"]
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()):
        raise ValueError("The saved TEST arrays are not valid aligned evidence.")
    from evaluation.metrics import bootstrap_ci
    recomputed = bootstrap_ci(labels, scores, config["threshold_exact"],
                              n_bootstrap=1000, ci=0.95, random_state=42)
    recomputed = {key: list(value) for key, value in recomputed.items()}
    if recomputed != stored or source_hashes(run) != before:
        raise ValueError("The canonical bootstrap differs or a source changed during verification.")
    report = {"status": "complete", "role": ROLE, "protocol_version": PROTOCOL_VERSION,
              "key": KEY, "run_dir": str(run), "config_sha256": CONFIG_SHA256,
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "original_bootstrap_field": "absent", "verified_iterations": 1000,
              "random_state": 42, "confidence_level": 0.95, "threshold_exact": config["threshold_exact"],
              "metrics_source_sha256": METRICS_SHA256, "source_artefacts_sha256": before,
              "stored_intervals": stored, "recomputed_intervals": recomputed,
              "original_artefacts_modified": False, "original_launch_snapshot_available": False,
              "interval_match_proves_original_iteration_count": False,
              "scope": "Independent prescribed 1000-draw verification of immutable saved TEST scores; no fitting, threshold selection or parameter selection.",
              "provenance_limit": "The original command was observed with --bootstrap-iterations 1000 in read-only process checks, but no immutable initial launch snapshot was saved. Matching rounded intervals alone do not uniquely prove the original iteration count. Source hashes are recorded at save time, not an initial in-memory code snapshot."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    reference = {"path": str(output), "sha256": file_sha256(output)}
    from revision_audit import update_manifest

    def register(latest):
        if KEY in latest.get("bootstrap_compatibility", {}):
            raise ValueError("A concurrent writer registered compatibility evidence; do not replace it.")
        latest.setdefault("bootstrap_compatibility", {})[KEY] = reference
        verify_bootstrap_compatibility(latest, run, config, 1000)

    update_manifest(manifest_path, register)
    return verify_bootstrap_compatibility(_read(manifest_path), run, config, 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(produce_and_register(args.manifest, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
