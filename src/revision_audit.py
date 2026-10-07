"""Preflight evidence and immutable historical pins for the thesis revision."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from data import load_dataset
from experiment_protocol import file_sha256, software_versions, replace_with_retry

ROOT = Path(__file__).resolve().parent.parent
ULB_RAW_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
HISTORICAL_ULB_LGBM = {
    "none": "run_20260309_181852",
    "rus": "run_20260313_122938",
    "ros": "run_20260313_131944",
    "smote": "run_20260313_144628",
    "smote_tomek": "run_20260313_163214",
    "smoteenn": "run_20260313_192034",
    "weights": "run_20260313_210148",
}


def update_manifest(path, update):
    """Share the training writer's lock and update the latest manifest atomically."""
    path = Path(path).resolve()
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + 30
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError("The revision manifest remains locked.")
            time.sleep(0.1)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        update(manifest)
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
        lock.unlink()


def profile_overlap(left, right):
    left_hash = pd.util.hash_pandas_object(left, index=False).to_numpy()
    right_hash = pd.util.hash_pandas_object(right, index=False).to_numpy()
    # A non-zero value is conservatively rejected; hash collisions cannot cause
    # a false declaration that equal profiles are disjoint.
    return int(pd.Series(right_hash).isin(left_hash).sum())


def preflight(output):
    X_dev, X_test, y_dev, y_test, metadata = load_dataset("ulb", return_metadata=True)
    outer_overlap = profile_overlap(X_dev, X_test)
    inner_overlap = []
    for train, validation in StratifiedKFold(5, shuffle=True, random_state=42).split(X_dev, y_dev):
        inner_overlap.append(profile_overlap(X_dev.iloc[train], X_dev.iloc[validation]))
    train, validation, _, _ = train_test_split(
        X_dev, y_dev, stratify=y_dev, test_size=0.2, random_state=42)
    ft_overlap = profile_overlap(train, validation)
    if outer_overlap or any(inner_overlap) or ft_overlap:
        raise AssertionError("Exact predictor profiles cross a protected split boundary.")
    metadata = {k: v for k, v in metadata.items() if not isinstance(v, np.ndarray)}
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "ulb": {"dev_rows": len(X_dev), "test_rows": len(X_test),
                      "dev_fraud": int(y_dev.sum()), "test_fraud": int(y_test.sum()),
                      "outer_exact_profile_overlap": outer_overlap,
                      "inner_fold_exact_profile_overlap": inner_overlap,
                      "ft_holdout_exact_profile_overlap": ft_overlap,
                      "provenance": metadata},
              "software_versions": software_versions()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def historical_pins():
    references = {}
    for path in sorted((ROOT / "results/baf_base").glob("*/*/run_*/config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("train_samples") != 800000 or config.get("test_samples") != 200000:
            continue
        model = config["model"]
        strategy = "none" if config["strategy"] == "n/a" else config["strategy"]
        key = f"baf_base/{model}/{strategy}"
        if key in references:
            raise ValueError(f"Ambiguous historical source: {key}")
        references[key] = {"run_dir": str(path.parent), "config_sha256": file_sha256(path),
                           "role": "historical_baf_primary_representation",
                           "dataset_hash_sha256": config["dataset_hash_sha256"]}
    if len(references) != 36:
        raise AssertionError("The BAF historical matrix must contain exactly 36 runs.")
    return references


def merge_historical_pins(manifest, pins):
    """Preserve every existing historical pin and reject any changed source."""
    merged = dict(manifest.get("historical_runs", {}))
    for key, reference in pins.items():
        if key in merged and merged[key] != reference:
            raise ValueError(f"Conflicting historical source: {key}")
        merged[key] = reference
    manifest["historical_runs"] = merged


def historical_ulb_lgbm_pins(manifest):
    """Verify seven named original-protocol runs without loading any dataset.

    These pins support historical score diagnostics only. They never become
    corrected primary runs or corrected baseline selectors.
    """
    references = {}
    for strategy, run_name in HISTORICAL_ULB_LGBM.items():
        run = (ROOT / "results/ulb_2013/lgbm" / strategy / run_name).resolve()
        config_path = run / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (config.get("dataset"), config.get("model"), config.get("strategy")) != (
                "ulb_2013", "lgbm", strategy):
            raise ValueError(f"The explicit historical ULB LGBM identity differs: {strategy}")
        if (config.get("train_samples") != 227845 or config.get("test_samples") != 56962
                or config.get("dataset_hash_sha256") != ULB_RAW_SHA256
                or config.get("sample_fraction") not in (None, 1, 1.0)):
            raise ValueError(f"The historical ULB LGBM source is not the preserved full raw-data split: {strategy}")
        if strategy == "none":
            baseline = manifest["historical_baseline_runs"]["ulb_2013"]["lgbm"]
            baseline = Path(baseline["run_dir"] if isinstance(baseline, dict) else baseline)
            baseline = baseline.resolve() if baseline.is_absolute() else (ROOT / baseline).resolve()
            if baseline != run:
                raise ValueError("The historical ULB LGBM None run differs from its preserved baseline pin.")
        references[f"ulb_2013/lgbm/{strategy}"] = {
            "run_dir": str(run), "config_sha256": file_sha256(config_path),
            "role": "historical_ulb_diagnostic_only", "dataset_hash_sha256": ULB_RAW_SHA256,
        }
    return references


def _project_source(path):
    """Resolve a recorded source without permitting paths outside this project."""
    path = Path(path)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("A citation source snapshot must remain inside this project.")
    return path


def validate_citation_claim_review(report):
    """Verify the inspected source snapshot, without certifying a final PDF."""
    snapshot = report.get("source_snapshot", {})
    files = snapshot.get("files")
    if (report.get("schema_version") != 1 or snapshot.get("hash_algorithm") != "SHA-256"
            or not isinstance(files, list) or not files):
        raise ValueError("The citation claim review requires a non-empty SHA-256 source snapshot.")
    seen = set()
    for reference in files:
        if not isinstance(reference, dict) or not reference.get("path") or not reference.get("sha256"):
            raise ValueError("The citation claim review contains an incomplete source reference.")
        source = _project_source(reference["path"])
        if source in seen:
            raise ValueError("The citation source snapshot repeats a source file.")
        seen.add(source)
        if file_sha256(source) != reference["sha256"]:
            raise ValueError(f"A citation source changed after the claim review: {source}")


def validate_historical_lgbm_diagnostics(report, manifest, manifest_path):
    """Verify original-protocol diagnostic identity and its pinned configs only.

    Registration performs no inference, fitting, resampling or numerical
    re-analysis. The report's historical manifest snapshot need not equal the
    current manifest: ongoing corrected training legitimately adds new pins.
    """
    if (report.get("status") != "complete"
            or report.get("role") != "historical_original_protocol_diagnostic_only"
            or report.get("protocol_version") is not None
            or report.get("fit_or_resampling_performed") is not False
            or report.get("threshold_selection_performed") is not False
            or report.get("test_labels_used_for_fitting") is not False):
        raise ValueError("Only a completed, no-fit original-protocol LGBM diagnostic can be registered.")
    recorded_manifest = Path(report.get("source_manifest", ""))
    recorded_manifest = (recorded_manifest if recorded_manifest.is_absolute()
                         else ROOT / recorded_manifest).resolve()
    if recorded_manifest != Path(manifest_path).resolve():
        raise ValueError("The historical diagnostic identifies another source manifest.")
    split = report.get("split_audit", {})
    if (split.get("raw_file_sha256") != ULB_RAW_SHA256
            or split.get("dev_rows") != 227845 or split.get("test_rows") != 56962
            or split.get("dev_fraud") != 394 or split.get("test_fraud") != 98
            or split.get("split_seed") != 42 or split.get("split_ratio") != "80/20 stratified"
            or split.get("all_saved_label_sequences_match_raw_test") is not True
            or report.get("historical_hpo", {}).get("baseline_trials") != 50):
        raise ValueError("The historical diagnostic does not identify the original full ULB split/search.")
    strategies = report.get("strategies")
    if not isinstance(strategies, dict) or set(strategies) != set(HISTORICAL_ULB_LGBM):
        raise ValueError("The historical diagnostic must reference all seven original strategies.")
    baseline_params = None
    for strategy in HISTORICAL_ULB_LGBM:
        key = f"ulb_2013/lgbm/{strategy}"
        pin = manifest.get("historical_runs", {}).get(key)
        record = strategies[strategy]
        if (not isinstance(pin, dict) or pin.get("role") != "historical_ulb_diagnostic_only"
                or not pin.get("config_sha256") or not isinstance(record, dict)):
            raise ValueError(f"The historical diagnostic lacks a compatible explicit source pin: {key}")
        source = Path(pin["run_dir"])
        source = (source if source.is_absolute() else ROOT / source).resolve()
        reported_source = Path(record.get("source_run", ""))
        reported_source = (reported_source if reported_source.is_absolute()
                           else ROOT / reported_source).resolve()
        config_path = source / "config.json"
        digest = file_sha256(config_path)
        if (reported_source != source or digest != pin["config_sha256"]
                or digest != record.get("source_config_sha256")):
            raise ValueError(f"The historical diagnostic source/config differs from its immutable pin: {key}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if ((config.get("dataset"), config.get("model"), config.get("strategy"))
                != ("ulb_2013", "lgbm", strategy)
                or config.get("protocol_version") is not None
                or config.get("dataset_hash_sha256") != ULB_RAW_SHA256
                or pin.get("dataset_hash_sha256") != ULB_RAW_SHA256
                or config.get("train_samples") != split["dev_rows"]
                or config.get("test_samples") != split["test_rows"]
                or config.get("train_fraud") != split["dev_fraud"]
                or config.get("test_fraud") != split["test_fraud"]
                or config.get("split_seed") != 42 or config.get("split_ratio") != "80/20 stratified"
                or config.get("sample_fraction") not in (None, 1, 1.0)
                or record.get("rows") != split["test_rows"] or record.get("fraud") != split["test_fraud"]):
            raise ValueError(f"A historical diagnostic source has incompatible original-protocol identity: {key}")
        if strategy == "none":
            if config.get("n_trials") != 50:
                raise ValueError("The historical diagnostic baseline lacks its original 50-trial search.")
            baseline_params = config.get("best_params")
        if (not isinstance(baseline_params, dict) or config.get("best_params") != baseline_params
                or record.get("best_params") != baseline_params):
            raise ValueError("Historical diagnostic strategies must retain the same baseline-selected parameters.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results_revision/20261005/revision_manifest.json")
    parser.add_argument("--register-transfer-partitions", type=Path)
    parser.add_argument("--register-bibliography", type=Path)
    parser.add_argument("--register-provenance-notes", type=Path)
    parser.add_argument("--register-citation-claims", type=Path,
                        help="Pin a source-verified citation claim review only; no preflight, fitting or PDF certification.")
    parser.add_argument("--register-historical-lgbm-diagnostics", type=Path,
                        help="Pin a verified original-protocol LGBM diagnostic only; no inference, preflight or fitting.")
    parser.add_argument("--register-historical-ulb-lgbm", action="store_true",
                        help="Register seven explicit original ULB LGBM diagnostic sources only; no preflight or fitting.")
    args = parser.parse_args()
    registrations = {
        "transfer_partitions": args.register_transfer_partitions,
        "bibliography_audit": args.register_bibliography,
        "execution_provenance_notes": args.register_provenance_notes,
        "citation_claim_review": args.register_citation_claims,
        "historical_lgbm_diagnostics": args.register_historical_lgbm_diagnostics,
    }
    if args.register_historical_ulb_lgbm:
        if any(registrations.values()):
            parser.error("Historical ULB registration is a standalone metadata-only action.")
        def register_ulb(manifest):
            merge_historical_pins(manifest, historical_ulb_lgbm_pins(manifest))
        update_manifest(args.manifest, register_ulb)
        print(json.dumps({"historical_ulb_lgbm_runs": len(HISTORICAL_ULB_LGBM),
                          "role": "historical_ulb_diagnostic_only", "preflight_or_fitting_performed": False}, indent=2))
        return
    if any(registrations.values()):
        verified = {}
        reports = {}
        for key, path in registrations.items():
            if path is None:
                continue
            path = path.resolve()
            if not path.is_relative_to(args.manifest.parent.resolve()):
                raise ValueError("Derived evidence must remain inside the revision root.")
            report_bytes = path.read_bytes()
            report = json.loads(report_bytes)
            if key == "transfer_partitions" and report.get("status") != "complete":
                raise ValueError("Incomplete transfer partitions cannot be pinned.")
            if key == "bibliography_audit" and not report["validation"].get("every_doi_has_matching_resolver_url"):
                raise ValueError("Bibliography identifier verification is incomplete.")
            reports[key] = report
            verified[key] = {"path": str(path), "sha256": hashlib.sha256(report_bytes).hexdigest()}
        def register_reports(manifest):
            # Verify against the latest pinned configs under the shared writer
            # lock. Registration changes metadata only, never completion status.
            for key, reference in verified.items():
                if file_sha256(reference["path"]) != reference["sha256"]:
                    raise ValueError("A report changed while its registration was being verified.")
                if key == "citation_claim_review":
                    validate_citation_claim_review(reports[key])
                elif key == "historical_lgbm_diagnostics":
                    validate_historical_lgbm_diagnostics(reports[key], manifest, args.manifest)
            manifest.update(verified)
        update_manifest(args.manifest, register_reports)
        print(json.dumps(verified, indent=2))
        return
    output = args.manifest.parent / "audit/protocol_preflight.json"
    report = preflight(output)
    pins = historical_pins()
    archive = args.manifest.parent / "archive/source_before_revision.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        subprocess.run(["git", "archive", "--format=zip", f"--output={archive}",
                        "c65b11de22e39335ae6a14abc168fd7e284a7cd4",
                        "src", "Overleaf", "requirements.txt"], cwd=ROOT, check=True)
    def register(manifest):
        merge_historical_pins(manifest, pins)
        manifest["status"] = "training_in_progress"
        manifest.setdefault("completion_checks", {})["protocol_tests"] = True
        manifest["preflight_report"] = str(output)
        manifest["source_before_revision_archive"] = {
            "path": str(archive), "sha256": file_sha256(archive),
            "commit": "c65b11de22e39335ae6a14abc168fd7e284a7cd4"}
    update_manifest(args.manifest, register)
    print(json.dumps({"ulb": report["ulb"], "historical_baf_runs": len(pins),
                      "archive": str(archive)}, indent=2))


if __name__ == "__main__":
    main()
