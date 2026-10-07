"""Shared provenance, explicit run selection, and corrected-run manifest tools."""
import hashlib
import importlib.metadata
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_VERSION = "thesis_revision_20261005_v1"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results_revision" / "20261005"
DEFAULT_MANIFEST_PATH = DEFAULT_RESULTS_ROOT / "revision_manifest.json"


def validate_revision_destinations(results_root=None, manifest_path=None):
    """Reject protected destinations before any operational or scientific work.

    This is a path/provenance guard only: it changes no model parameters,
    random seeds, preprocessing, split membership or numerical computation.
    Resolving paths also catches traversals and Windows case-equivalent names.
    A custom result root defaults to its own manifest, not another experiment.
    """
    output_root = Path(results_root or DEFAULT_RESULTS_ROOT).resolve()
    manifest_file = Path(manifest_path or output_root / "revision_manifest.json").resolve()
    for destination in (output_root, manifest_file):
        for name in ("results", "results_thesis", "Overleaf"):
            protected = (PROJECT_ROOT / name).resolve()
            if destination == protected or protected in destination.parents:
                qualification = " (original results archive)" if name == "results" else ""
                raise ValueError(f"Revision destinations must not modify preserved {name} artefacts{qualification}: {destination}")
        if destination == PROJECT_ROOT.resolve():
            raise ValueError("The repository root is not a revision output destination.")
    return output_root, manifest_file


def capture_source_provenance(results_root=None):
    """Capture launch provenance after an operational-only destination guard."""
    output_root, _ = validate_revision_destinations(results_root)
    directory = output_root / "source_snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    snapshot = directory / f"source_{stamp}_{os.getpid()}.zip"
    hashes = {}
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
            content = path.read_bytes()
            name = path.relative_to(PROJECT_ROOT).as_posix()
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    return {
        "source_snapshot_path": str(snapshot), "source_snapshot_sha256": file_sha256(snapshot),
        "source_hashes_at_launch": hashes,
        "source_capture_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot_scope": "on_disk_at_entry_point_launch_before_fitting",
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_with_retry(source, destination, attempts=12):
    """Retry transient Windows/antivirus locks without discarding the source."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.05 * (2 ** attempt), 1.0))


def load_sampler_checkpoint(path):
    """Recover the newest valid committed or interrupted TPE checkpoint."""
    import joblib
    import optuna
    path = Path(path)
    candidates = [candidate for candidate in (path, path.with_name(path.name + ".tmp"))
                  if candidate.exists()]
    if not candidates:
        return None, None
    # A valid temporary file may belong to a trial already committed in SQLite.
    # Prefer it to an older promoted checkpoint rather than resetting its RNG.
    candidate = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    payload = joblib.load(candidate)
    sampler = payload["sampler"] if isinstance(payload, dict) else payload
    if not isinstance(sampler, optuna.samplers.TPESampler):
        raise ValueError("The Optuna sampler checkpoint is not a valid TPESampler.")
    metadata = {"loaded_from": str(candidate), "sha256": file_sha256(candidate),
                "completed_trial_number": (payload.get("completed_trial_number")
                                           if isinstance(payload, dict) else None)}
    if candidate != path:
        replace_with_retry(candidate, path)
        metadata["recovered_temporary_checkpoint"] = True
    return sampler, metadata


def save_sampler_checkpoint(study, trial, path):
    """Persist TPE RNG state after SQLite commits a completed/pruned trial."""
    import joblib
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump({"sampler": study.sampler, "completed_trial_number": trial.number,
                 "study_name": study.study_name}, temporary)
    # Deserialise before promotion; a damaged checkpoint must never replace one.
    payload = joblib.load(temporary)
    if payload["completed_trial_number"] != trial.number:
        raise ValueError("The temporary sampler checkpoint failed validation.")
    replace_with_retry(temporary, path)


def array_sha256(values):
    """Hash shape, dtype and contiguous array bytes, preserving row order."""
    array = np.ascontiguousarray(np.asarray(values))
    if array.dtype.hasobject:
        raise TypeError("Object arrays require table_sha256 instead.")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def table_sha256(values):
    """Hash an ordered feature table without allocating a second dense matrix."""
    frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
    digest = hashlib.sha256()
    digest.update(json.dumps([str(c) for c in frame.columns]).encode("utf-8"))
    digest.update(json.dumps([str(d) for d in frame.dtypes]).encode("utf-8"))
    rows = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    digest.update(memoryview(rows).cast("B"))
    return digest.hexdigest()


def software_versions():
    packages = (
        "numpy", "pandas", "scikit-learn", "imbalanced-learn", "optuna",
        "lightgbm", "catboost", "torch", "joblib", "shap",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def class_counts(labels):
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def sampler_diagnostics(sampler, X_before, y_before, X_after, y_after, stage):
    """Record actual resampling counts and fitted cleaner removals, not guesses."""
    counts_before = class_counts(y_before)
    counts_after = class_counts(y_after)
    result = {
        "stage": stage, "sampler_class": type(sampler).__name__,
        "rows_before": len(y_before), "rows_after": len(y_after),
        "class_counts_before": counts_before, "class_counts_after": counts_after,
        "X_before_sha256": table_sha256(X_before),
        "X_after_sha256": table_sha256(X_after),
        "y_before_sha256": array_sha256(y_before),
        "y_after_sha256": array_sha256(y_after),
        "cleaner_removed_rows": None,
    }
    if hasattr(sampler, "sample_indices_"):
        indices = np.asarray(sampler.sample_indices_, dtype=np.int64)
        result["sample_indices_sha256"] = array_sha256(indices)
        result["distinct_original_indices_retained"] = int(np.unique(indices).size)
    for attribute in ("tomek_", "enn_"):
        cleaner = getattr(sampler, attribute, None)
        if cleaner is not None and hasattr(cleaner, "sample_indices_"):
            indices = np.asarray(cleaner.sample_indices_, dtype=np.int64)
            # SMOTE's fitted sampling_strategy_ records the number generated.
            smote = sampler.smote_
            generated = sum(int(n) for n in smote.sampling_strategy_.values())
            intermediate_rows = len(y_before) + generated
            result.update({
                "smote_generated_rows": generated,
                "rows_after_smote_before_cleaning": intermediate_rows,
                "cleaner_removed_rows": intermediate_rows - len(indices),
                "cleaner_indices_sha256": array_sha256(indices),
            })
    return result


def resolve_baseline_run(dataset_label, model_name, results_root=None,
                         manifest_path=None, explicit_run=None):
    """Select a pinned baseline; never silently choose the latest timestamp."""
    root = Path(results_root or DEFAULT_RESULTS_ROOT).resolve()
    if explicit_run:
        run_dir = Path(explicit_run).resolve()
    else:
        manifest_file = Path(manifest_path or root / "revision_manifest.json")
        if not manifest_file.exists():
            raise FileNotFoundError("No explicit baseline or corrected-run manifest is available.")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        key = f"{dataset_label}/{model_name}"
        reference = manifest.get("baseline_runs", {}).get(key)
        if reference is None:
            raise FileNotFoundError(f"No pinned corrected baseline for {key}.")
        run_dir = Path(reference if isinstance(reference, str) else reference["run_dir"])
        if not run_dir.is_absolute():
            run_dir = manifest_file.parent / run_dir
        run_dir = run_dir.resolve()
        if isinstance(reference, dict) and reference.get("config_sha256"):
            if file_sha256(run_dir / "config.json") != reference["config_sha256"]:
                raise ValueError("The pinned baseline config hash has changed.")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("dataset") != dataset_label or config.get("model") != model_name:
        raise ValueError("The selected baseline has the wrong dataset or model.")
    if config.get("strategy") not in ("none", "n/a"):
        raise ValueError("A baseline run, rather than an intervention run, is required.")
    return run_dir


def record_completed_run(run_dir, manifest_path=None):
    """Register immutable runs with operational guards; never alter numerics."""
    run_dir, manifest_file = validate_revision_destinations(run_dir, manifest_path or DEFAULT_MANIFEST_PATH)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = manifest_file.with_suffix(manifest_file.suffix + ".lock")
    deadline = time.monotonic() + 30
    while True:
        try:
            lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(lock_fd)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Manifest is locked: {lock_file}")
            time.sleep(0.1)
    temporary = manifest_file.with_name(manifest_file.name + f".{os.getpid()}.tmp")
    try:
        manifest = (
            json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest_file.exists() else {}
        )
        manifest.setdefault("schema_version", 1)
        manifest.setdefault("protocol_version", PROTOCOL_VERSION)
        for section in ("datasets", "baseline_runs", "runs", "historical_baseline_runs"):
            manifest.setdefault(section, {})
        strategy_key = "none" if config["strategy"] == "n/a" else config["strategy"]
        key = f"{config['dataset']}/{config['model']}/{strategy_key}"
        entry = {
            "run_dir": str(run_dir), "config_sha256": file_sha256(run_dir / "config.json"),
            "dataset_hash_sha256": config.get("dataset_hash_sha256"),
            "protocol_version": config.get("protocol_version"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "missing_policy": config.get("missing_policy", "preserve"),
            "sample_fraction": config.get("sample_fraction"),
        }
        previous = manifest["runs"].get(key)
        if previous and previous.get("run_dir") != entry["run_dir"]:
            manifest.setdefault("superseded_runs", []).append({"key": key, **previous})
        manifest["runs"][key] = entry
        if config["strategy"] in ("none", "n/a"):
            manifest["baseline_runs"][f"{config['dataset']}/{config['model']}"] = entry
        if config.get("data_provenance"):
            manifest["datasets"][config["dataset"]] = config["data_provenance"]
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        replace_with_retry(temporary, manifest_file)
    finally:
        if temporary.exists():
            temporary.unlink()
        lock_file.unlink()


def optuna_trials_record(study):
    """Serialise all attempted trials, including pruning and intermediate scores."""
    return [
        {
            "number": trial.number, "state": trial.state.name,
            "value": trial.value, "params": trial.params,
            "user_attrs": trial.user_attrs,
            "intermediate_values": trial.intermediate_values,
            "duration_s": trial.duration.total_seconds() if trial.duration else None,
        }
        for trial in study.trials
    ]
