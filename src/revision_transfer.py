"""Audited, frozen-model BAF transfer on exact-profile-disjoint Variant TESTs.

Original Variant TEST membership is retained before excluding any row whose
common 30 predictors exactly match a Base DEV row. Hashes screen candidates;
every candidate pair is checked element by element. Labels do not determine
exclusion. No model, preprocessor, threshold or hyperparameter is fitted here.

All outputs are new, immutable directories beneath the revision transfer root.
Corrected baseline pins take precedence for every family, including recovered
classical validation thresholds. Historical references remain explicit context.
FT-Transformer must use its corrected baseline and persisted fitted preprocessors;
the historical FT checkpoint is never an automatic fallback.

Examples:
    python src/revision_transfer.py --prepare-only
    python src/revision_transfer.py --partitions-dir <prepared-directory> \
        --models logreg lgbm catboost --bootstrap-iterations 0
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from data import load_dataset
from evaluation.metrics import bootstrap_ci, compute_all_metrics
from experiment_protocol import (
    DEFAULT_MANIFEST_PATH, DEFAULT_RESULTS_ROOT, PROJECT_ROOT, PROTOCOL_VERSION,
    array_sha256, file_sha256, software_versions, table_sha256,
)
from revision_resources import exclusive_resource


TRANSFER_ROOT = DEFAULT_RESULTS_ROOT / "transfer"
MODEL_ORDER = ("logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm")
PARTIAL_MODELS = MODEL_ORDER[:4]
VARIANTS = {
    "baf_var1": "Variant I", "baf_var2": "Variant II", "baf_var3": "Variant III",
    "baf_var4": "Variant IV", "baf_var5": "Variant V",
}
FORBIDDEN_PREDICTORS = {"month", "x1", "x2", "fraud_bool", "Class", "index", "Unnamed: 0"}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _serialisable(value):
    if isinstance(value, dict):
        return {str(key): _serialisable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialisable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _serialisable(value.tolist())
    if isinstance(value, np.generic):
        return _serialisable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    """Use exclusive creation, including for small diagnostic artefacts."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_serialisable(value), stream, indent=2, allow_nan=False)


def _save_array(directory, name, values, artefacts):
    values = np.asarray(values)
    path = Path(directory) / name
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    artefacts[name] = {
        "file_sha256": file_sha256(path), "array_sha256": array_sha256(values),
        "shape": list(values.shape), "dtype": str(values.dtype),
    }


def validate_output_root(path):
    destination = Path(path).resolve()
    for name in ("results", "results_thesis", "Overleaf", "Article 1", "Article 2"):
        protected = (PROJECT_ROOT / name).resolve()
        if destination == protected or protected in destination.parents:
            raise ValueError(f"Transfer outputs must not replace preserved {name} artefacts.")
    if destination == PROJECT_ROOT.resolve():
        raise ValueError("The repository root is not a transfer output directory.")
    return destination


def _new_directory(root, kind):
    root = validate_output_root(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = root / kind / stamp
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def validate_schema(frame, columns=None):
    actual = list(frame.columns)
    if len(actual) != 30 or len(set(actual)) != 30:
        raise ValueError("BAF transfer requires exactly 30 distinct common predictors.")
    if FORBIDDEN_PREDICTORS.intersection(actual):
        raise ValueError("A target, index, month or Variant-only feature entered the predictor schema.")
    if columns is not None and actual != list(columns):
        raise ValueError("The Variant predictor schema or column order differs from Base.")
    if frame.isna().to_numpy().any():
        raise ValueError("Unexpected NaN in the sentinel-preserving transfer representation.")
    return actual


def predictor_hashes(frame):
    """Hash equal numeric values consistently across integer/float dtypes.

Signed zero is normalised. Large integers that cannot be represented exactly
as float64 are rejected instead of silently merging distinct predictor values.
The returned hash is only a candidate-screening key, never proof of equality.
"""
    canonical = pd.DataFrame(index=frame.index)
    for name in frame.columns:
        values = frame[name]
        if pd.api.types.is_numeric_dtype(values.dtype):
            numeric = values.to_numpy(dtype=np.float64, copy=True)
            if not np.isfinite(numeric).all():
                raise ValueError("Non-finite numeric predictor encountered during matching.")
            if pd.api.types.is_integer_dtype(values.dtype):
                if np.any(np.abs(numeric) > 2**53):
                    raise ValueError("Integer predictor exceeds the exact float64 matching range.")
            numeric[numeric == 0] = 0.0
            canonical[name] = numeric
        else:
            canonical[name] = values.astype(object)
    return pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype=np.uint64)


def exact_overlap(base_dev, target_test, *, base_hashes=None, target_hashes=None):
    """Return exclusions and all verified position pairs without using labels."""
    if list(base_dev.columns) != list(target_test.columns):
        raise ValueError("Exact matching requires identical ordered feature schemas.")
    if base_dev.isna().to_numpy().any() or target_test.isna().to_numpy().any():
        raise ValueError("Missing values must not silently bypass exact overlap checks.")
    base_hashes = predictor_hashes(base_dev) if base_hashes is None else np.asarray(base_hashes)
    target_hashes = predictor_hashes(target_test) if target_hashes is None else np.asarray(target_hashes)
    if len(base_hashes) != len(base_dev) or len(target_hashes) != len(target_test):
        raise ValueError("Hash arrays do not align with their feature tables.")
    candidates = np.flatnonzero(np.isin(target_hashes, base_hashes))
    pairs = pd.DataFrame({"hash": target_hashes[candidates], "target_position": candidates}).merge(
        pd.DataFrame({"hash": base_hashes, "base_position": np.arange(len(base_dev))}),
        on="hash", how="inner", sort=False,
    )
    target_positions = pairs["target_position"].to_numpy(dtype=np.int64)
    base_positions = pairs["base_position"].to_numpy(dtype=np.int64)
    verified = np.ones(len(pairs), dtype=bool)
    for name in base_dev.columns:
        left = base_dev[name].to_numpy()[base_positions]
        right = target_test[name].to_numpy()[target_positions]
        verified &= left == right
    matches = np.column_stack((target_positions[verified], base_positions[verified]))
    excluded = np.zeros(len(target_test), dtype=bool)
    excluded[np.unique(matches[:, 0])] = True
    return excluded, matches.astype(np.int64), {
        "hash_candidate_target_rows": int(len(candidates)),
        "hash_candidate_pairs": int(len(pairs)),
        "elementwise_verified_pairs": int(verified.sum()),
        "rejected_hash_candidate_pairs": int((~verified).sum()),
        "elementwise_verification_columns": list(base_dev.columns),
        "target_labels_used_for_exclusion": False,
    }


def _population(labels):
    labels = np.asarray(labels)
    positives = int(labels.sum())
    return {"rows": len(labels), "positive_rows": positives,
            "negative_rows": len(labels) - positives,
            "fraud_prevalence": positives / len(labels) if len(labels) else None}


def _metadata_without_indices(metadata):
    return {key: value for key, value in metadata.items()
            if key not in ("dev_indices", "test_indices")}


def prepare_partitions(output_root=TRANSFER_ROOT, manifest_path=DEFAULT_MANIFEST_PATH,
                       *, loader=load_dataset):
    """Prepare five immutable, model-independent disjoint TEST partitions."""
    base_dev, base_test, base_y_dev, base_y_test, base_meta = loader("baf_base", return_metadata=True)
    columns = validate_schema(base_dev)
    validate_schema(base_test, columns)
    if not np.array_equal(base_dev.index.to_numpy(), base_meta["dev_indices"]):
        raise ValueError("Base DEV row positions do not match loader metadata.")
    destination = _new_directory(output_root, "partitions")
    base_hashes = predictor_hashes(base_dev)
    artefacts = {}
    _save_array(destination, "base_dev_row_indices.npy", base_meta["dev_indices"], artefacts)
    _save_array(destination, "base_test_row_indices.npy", base_meta["test_indices"], artefacts)
    _save_array(destination, "base_test_y.npy", np.asarray(base_y_test), artefacts)
    _save_array(destination, "base_dev_predictor_hashes.npy", base_hashes, artefacts)
    document = {
        "status": "preparing", "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(Path(manifest_path).resolve()),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_code_sha256": {"src/revision_transfer.py": file_sha256(__file__),
                               "src/data.py": file_sha256(Path(__file__).with_name("data.py"))},
        "feature_columns": columns, "matching_policy": "any_exact_common_predictor_match_with_Base_DEV",
        "matching_verification": "All hash candidate pairs checked element by element across all 30 predictors.",
        "scope": "Distribution transfer under a shared schema; not prospective temporal drift or the x1/x2 separability intervention.",
        "missing_value_policy": "numeric_sentinels_preserved",
        "base": {"data_provenance": _metadata_without_indices(base_meta),
                 "dev_population": _population(base_y_dev), "test_population": _population(base_y_test),
                 "dev_predictor_table_sha256": table_sha256(base_dev)},
        "artefacts": artefacts, "variants": {},
    }
    del base_test
    gc.collect()
    for variant, label in VARIANTS.items():
        unused_dev, target_test, unused_y_dev, target_y_test, metadata = loader(variant, return_metadata=True)
        del unused_dev, unused_y_dev
        validate_schema(target_test, columns)
        if not np.array_equal(target_test.index.to_numpy(), metadata["test_indices"]):
            raise ValueError("Variant TEST row positions do not match loader metadata.")
        excluded, matches, match_audit = exact_overlap(base_dev, target_test, base_hashes=base_hashes)
        kept_positions = np.flatnonzero(~excluded).astype(np.int64)
        excluded_positions = np.flatnonzero(excluded).astype(np.int64)
        labels = np.asarray(target_y_test)
        row_indices = np.asarray(metadata["test_indices"], dtype=np.int64)
        if len(np.unique(labels[kept_positions])) != 2:
            raise ValueError("A corrected Variant TEST must retain both classes.")
        variant_dir = destination / variant
        variant_dir.mkdir(exist_ok=False)
        files = {}
        for filename, values in (
            ("original_test_row_indices.npy", row_indices),
            ("kept_test_positions.npy", kept_positions),
            ("excluded_test_positions.npy", excluded_positions),
            ("kept_test_row_indices.npy", row_indices[kept_positions]),
            ("excluded_test_row_indices.npy", row_indices[excluded_positions]),
            ("original_test_y.npy", labels), ("kept_test_y.npy", labels[kept_positions]),
            ("excluded_test_y.npy", labels[excluded_positions]),
            ("verified_overlap_row_pairs.npy", np.column_stack((
                row_indices[matches[:, 0]], np.asarray(base_meta["dev_indices"])[matches[:, 1]],
            ))),
        ):
            _save_array(variant_dir, filename, values, files)
        label_disagreements = int(np.sum(labels[matches[:, 0]] != np.asarray(base_y_dev)[matches[:, 1]]))
        variant_audit = {
            "label": label, "data_provenance": _metadata_without_indices(metadata),
            "original_population": _population(labels), "kept_population": _population(labels[kept_positions]),
            "excluded_population": _population(labels[excluded_positions]),
            "excluded_fraction": float(excluded.mean()), "matching_audit": match_audit,
            "matched_pair_label_disagreements": label_disagreements, "artefacts": files,
        }
        _write_json(variant_dir / "partition_audit.json", variant_audit)
        document["variants"][variant] = {
            "directory": variant, "partition_audit_sha256": file_sha256(variant_dir / "partition_audit.json"),
            "original_population": variant_audit["original_population"],
            "kept_population": variant_audit["kept_population"],
            "excluded_population": variant_audit["excluded_population"],
        }
        print(f"{label}: {len(labels):,} original; {len(kept_positions):,} kept; "
              f"{len(excluded_positions):,} exact-profile overlaps excluded", flush=True)
        del target_test, target_y_test
        gc.collect()
    document["status"] = "complete"
    _write_json(destination / "partition_manifest.json", document)
    print(f"Prepared partitions: {destination}", flush=True)
    return destination


def _verify_artefacts(directory, records):
    for filename, record in records.items():
        if file_sha256(Path(directory) / filename) != record["file_sha256"]:
            raise ValueError(f"A prepared partition artefact has changed: {filename}")


def load_partition_manifest(directory):
    directory = Path(directory).resolve()
    document = _read_json(directory / "partition_manifest.json")
    if document.get("status") != "complete" or document.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("A completed partition manifest for the current protocol is required.")
    if set(document.get("variants", {})) != set(VARIANTS):
        raise ValueError("The prepared manifest must contain all five Variants.")
    _verify_artefacts(directory, document["artefacts"])
    for variant, record in document["variants"].items():
        variant_dir = directory / variant
        if record["directory"] != variant:
            raise ValueError("A Variant partition must use its explicit local directory.")
        audit_file = variant_dir / "partition_audit.json"
        if file_sha256(audit_file) != record["partition_audit_sha256"]:
            raise ValueError("A Variant partition audit has changed.")
        _verify_artefacts(variant_dir, _read_json(audit_file)["artefacts"])
    return document


def resolve_transfer_source(manifest_path, model_name):
    """Prefer corrected pins for all families; a corrected FT is mandatory."""
    if model_name not in MODEL_ORDER:
        raise ValueError(f"Unsupported transfer model: {model_name}")
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    reference = manifest.get("baseline_runs", {}).get(f"baf_base/{model_name}")
    corrected_source = reference is not None
    if corrected_source:
        if not isinstance(reference, dict) or not reference.get("config_sha256"):
            raise ValueError("The corrected baseline requires an explicit run and config SHA-256 pin.")
    elif model_name == "fttransformer":
        if reference is None:
            raise FileNotFoundError("No corrected FT baseline is pinned; historical FT fallback is prohibited.")
    else:
        reference = manifest.get("historical_baseline_runs", {}).get("baf_base", {}).get(model_name)
        if reference is None:
            raise FileNotFoundError(f"No historical BAF baseline is explicitly pinned for {model_name}.")
    raw_path = reference if isinstance(reference, str) else reference["run_dir"]
    run_dir = Path(raw_path)
    if not run_dir.is_absolute():
        # Historical manifest paths are repository-relative. Corrected pins
        # written by record_completed_run are absolute or manifest-relative.
        run_dir = (manifest_path.parent if corrected_source else PROJECT_ROOT) / run_dir
    run_dir = run_dir.resolve()
    config_file = run_dir / "config.json"
    if isinstance(reference, dict) and reference.get("config_sha256"):
        if file_sha256(config_file) != reference["config_sha256"]:
            raise ValueError("The explicitly pinned source config hash has changed.")
    config = _read_json(config_file)
    if config.get("dataset") != "baf_base" or config.get("model") != model_name:
        raise ValueError("The pinned source has the wrong dataset or model.")
    if config.get("strategy") not in ("none", "n/a"):
        raise ValueError("Transfer requires the pinned Base baseline, not an intervention.")
    if config.get("sample_fraction") not in (None, 1, 1.0):
        raise ValueError("Transfer requires the complete Base baseline, not a subsample diagnostic.")
    if config.get("missing_policy", "preserve") != "preserve":
        raise ValueError("The principal transfer must preserve the primary sentinel representation.")
    required = ["config.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy"]
    if corrected_source:
        historical = (PROJECT_ROOT / "results").resolve()
        if run_dir == historical or historical in run_dir.parents:
            raise ValueError("A historical checkpoint cannot be relabelled as a corrected baseline.")
        if config.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("The source does not identify the corrected protocol.")
        required += ["completed.json", "test_row_indices.npy"]
    if model_name == "fttransformer":
        required += ["model.pt", "preprocessors.joblib"]
    else:
        required += ["model.joblib"]
    for filename in required:
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(f"Missing immutable source artefact: {run_dir / filename}")
    if corrected_source:
        completion = _read_json(run_dir / "completed.json")
        if (completion.get("status") != "complete"
                or completion.get("protocol_version") != PROTOCOL_VERSION):
            raise ValueError("The corrected source lacks a valid completion marker for this protocol.")
    return run_dir, config


@dataclass
class FrozenScorer:
    model_name: str
    run_dir: Path
    config: dict
    threshold: float
    threshold_source: str
    predictor: object
    source_artefacts: dict

    def score(self, frame):
        scores = np.asarray(self.predictor(frame))
        if scores.shape != (len(frame),) or not np.isfinite(scores).all():
            raise ValueError("Frozen model scores are non-finite or not aligned with input rows.")
        if self.model_name != "ocsvm" and not np.all((scores >= 0) & (scores <= 1)):
            raise ValueError("A supervised probability score lies outside [0, 1].")
        return scores


def load_frozen_scorer(model_name, manifest_path, *, device="cpu", batch_size=2048,
                       selected_source=None):
    if selected_source is None:
        run_dir, config = resolve_transfer_source(manifest_path, model_name)
    else:
        run_dir, config, expected_config_sha256 = selected_source
        if file_sha256(run_dir / "config.json") != expected_config_sha256:
            raise ValueError("The source config changed after the evaluation selection snapshot.")
    metrics = _read_json(run_dir / "metrics_test.json")
    threshold = float(config.get("threshold_exact", metrics["threshold"]))
    threshold_source = ("config.threshold_exact" if "threshold_exact" in config
                        else "historical_metrics_test.threshold_saved_to_six_decimal_places")
    artefact_names = ["config.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy"]
    artefact_names += [name for name in ("completed.json", "test_row_indices.npy")
                       if (run_dir / name).is_file()]
    if model_name == "fttransformer":
        import torch
        from models.fttransformer import TabularDataset, build_model, evaluate
        from torch.utils.data import DataLoader
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(device)
        checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
        threshold = float(checkpoint["threshold"])
        threshold_source = "corrected_checkpoint.threshold_full_precision_selected_on_Base_validation"
        if "threshold_exact" in config and threshold != float(config["threshold_exact"]):
            raise ValueError("FT checkpoint and config disagree about the full-precision frozen threshold.")
        if abs(threshold - float(metrics["threshold"])) > 5.1e-7:
            raise ValueError("FT checkpoint and metrics disagree beyond threshold rounding precision.")
        preprocessors = joblib.load(run_dir / "preprocessors.joblib")
        for key in ("num_preprocessor", "cat_encoder", "num_cols", "cat_cols"):
            if key not in preprocessors:
                raise ValueError(f"Persisted FT preprocessor bundle lacks {key}.")
        if (preprocessors["num_cols"] != checkpoint["num_cols"]
                or preprocessors["cat_cols"] != checkpoint["cat_cols"]):
            raise ValueError("FT checkpoint and fitted preprocessor feature schemas disagree.")
        model = build_model(checkpoint["hyperparams"], checkpoint["d_numerical"],
                            checkpoint["cat_cardinalities"]).to(target_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        def predict(frame):
            numeric = np.asarray(preprocessors["num_preprocessor"].transform(
                frame[preprocessors["num_cols"]]), dtype=np.float32)
            categorical = None
            if preprocessors["cat_cols"]:
                categorical = np.asarray(preprocessors["cat_encoder"].transform(
                    frame[preprocessors["cat_cols"]]), dtype=np.int64)
            loader = DataLoader(TabularDataset(numeric, categorical), batch_size=batch_size,
                                shuffle=False, num_workers=0)
            # The same model forward pass maps unknown category code -1 to
            # its reserved embedding; no independent clamping or refitting.
            return evaluate(model, loader, target_device)[1]

        artefact_names += ["model.pt", "preprocessors.joblib"]
    else:
        model = joblib.load(run_dir / "model.joblib")

        def predict(frame):
            if model_name == "ocsvm":
                return -model.decision_function(frame)
            return model.predict_proba(frame)[:, 1]

        artefact_names.append("model.joblib")
    if not np.isfinite(threshold):
        raise ValueError("The pinned Base decision threshold is not finite.")
    source_hashes = {name: file_sha256(run_dir / name) for name in artefact_names}
    return FrozenScorer(model_name, run_dir, config, threshold, threshold_source, predict, source_hashes)


def validate_base_predictions(scorer, base_test, labels, metadata):
    """Fail before Variant inference if the frozen Base scorer is inconsistent."""
    if scorer.config.get("dataset_hash_sha256") != metadata["raw_file_sha256"]:
        raise ValueError("Base raw-file hash differs from the source model's dataset hash.")
    saved_labels = np.load(scorer.run_dir / "y_test.npy", allow_pickle=False)
    saved_scores = np.load(scorer.run_dir / "y_test_scores.npy", allow_pickle=False)
    if not np.array_equal(np.asarray(labels), saved_labels):
        raise ValueError("Loaded Base TEST labels differ from the frozen source's saved labels.")
    index_file = scorer.run_dir / "test_row_indices.npy"
    if index_file.exists() and not np.array_equal(np.load(index_file, allow_pickle=False), metadata["test_indices"]):
        raise ValueError("Loaded Base TEST row positions differ from the pinned corrected source.")
    scores = scorer.score(base_test)
    atol, rtol = ((1e-6, 1e-5) if scorer.model_name == "fttransformer" else (1e-12, 1e-10))
    if scores.shape != saved_scores.shape or not np.allclose(scores, saved_scores, atol=atol, rtol=rtol):
        raise ValueError("Frozen Base TEST predictions do not reproduce the source's saved score arrays.")
    classification_changes = int(np.sum((scores >= scorer.threshold) != (saved_scores >= scorer.threshold)))
    if classification_changes:
        raise ValueError("Numerical scorer differences changed Base decisions at the frozen threshold.")
    return {"rows": len(scores), "labels_exactly_equal": True,
            "scores_bitwise_equal": bool(np.array_equal(scores, saved_scores)),
            "maximum_absolute_score_difference": float(np.max(np.abs(scores - saved_scores))),
            "score_absolute_tolerance": atol, "score_relative_tolerance": rtol,
            "decisions_changed_at_frozen_threshold": classification_changes,
            "recomputed_metrics": compute_all_metrics(labels, scores, scorer.threshold)}


def _metrics(labels, scores, threshold):
    metrics = compute_all_metrics(labels, scores, threshold)
    metrics["precision"] = metrics["TP"] / (metrics["TP"] + metrics["FP"]) if metrics["TP"] + metrics["FP"] else 0.0
    metrics["recall"] = metrics["TP"] / (metrics["TP"] + metrics["FN"])
    metrics["fraud_prevalence"] = float(np.mean(labels))
    return metrics, {"average_precision": float(average_precision_score(labels, scores)),
                     "roc_auc": float(roc_auc_score(labels, scores)), "threshold": threshold}


def _historical_variant_metrics(manifest_path, model_name, variant):
    manifest = _read_json(manifest_path)
    reference = manifest.get("historical_baseline_runs", {}).get("baf_base", {}).get(model_name)
    if reference is None:
        return None
    path = Path(reference if isinstance(reference, str) else reference["run_dir"])
    path = path if path.is_absolute() else PROJECT_ROOT / path
    historical_file = path / "cross_domain.json"
    if not historical_file.exists():
        return None
    value = _read_json(historical_file).get(variant)
    return {"role": "historical_unfiltered_population_context_not_retuned",
            "same_model_checkpoint_as_corrected_transfer": model_name != "fttransformer",
            "source_file": str(historical_file.resolve()), "source_file_sha256": file_sha256(historical_file),
            "saved_record": value}


def _resume_request(directory, manifest_path, models, device, batch_size, bootstrap_iterations,
                    output_dir=None):
    """Bound the explicit recovery; unavailable CUDA must fail before CSV loading."""
    directory = validate_output_root(directory)
    expected = Path(manifest_path).resolve().parent / "derived/transfer"
    if directory != expected or not directory.is_dir():
        raise ValueError("Partial recovery requires this manifest's existing derived/transfer directory.")
    if output_dir is not None and Path(output_dir).resolve() != directory:
        raise ValueError("The explicit output and verified partial directory must coincide.")
    if (tuple(models) != MODEL_ORDER or device != "cuda" or batch_size != 2048
            or bootstrap_iterations != 1000):
        raise ValueError("Partial recovery requires all six models, explicit CUDA, batch 2048 and bootstrap 1000.")
    state = _read_json(Path(manifest_path).resolve().parent / "analysis_state.json")
    approval = state.get("explicit_transfer_recovery", {}).get("verified_reuse")
    if not isinstance(approval, dict) or set(approval) != set(PARTIAL_MODELS):
        raise ValueError("Partial reuse requires explicit recovery approval with all four hash-pinned model records.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Verified partial transfer requires CUDA; CPU fallback is prohibited.")
    return directory


def _equal_metrics(saved, expected, context):
    expected = _serialisable(expected)
    if set(saved) != set(expected):
        raise ValueError(f"Incomplete or unexpected saved metrics: {context}")
    for key, value in expected.items():
        actual = saved[key]
        if isinstance(value, float):
            if actual is None or not np.isfinite(actual) or not np.isclose(actual, value, rtol=0, atol=1e-12):
                raise ValueError(f"Saved metric differs from its immutable arrays: {context}/{key}")
        elif actual != value:
            raise ValueError(f"Saved metric differs from its immutable arrays: {context}/{key}")


def verify_partial_transfer(directory, manifest_path, partitions_dir):
    """Validate only the four complete classical families, without inference.

    Return their unchanged source-audit records in ``models`` and per-model
    file/hash snapshots in ``reuse_provenance``. Confidence intervals are
    verified as saved evidence, never recomputed. The declared 1000-draw budget
    comes from the recorded analysis command, not from matching interval bounds.
    No completion marker is written and no incomplete or unknown model is skipped.
    """
    directory, manifest_path = Path(directory).resolve(), Path(manifest_path).resolve()
    if (validate_output_root(directory) != manifest_path.parent / "derived/transfer"
            or not directory.is_dir() or {p.name for p in directory.iterdir()} != set(PARTIAL_MODELS)
            or any(not (directory / name).is_dir() for name in PARTIAL_MODELS)):
        raise ValueError("Only the exact four complete classical families may be reused; no other partial outputs.")
    state = _read_json(manifest_path.parent / "analysis_state.json")
    recovery = state.get("explicit_transfer_recovery")
    if recovery is None:
        known_log = manifest_path.parent / "logs/analysis_transfer_six_models_20261006_235034_037097.log"
        if (state.get("status") != "failed" or state.get("last_child_exit_code") != 1
                or state.get("failure") != f"Analysis transfer_six_models failed; inspect {known_log}"
                or not known_log.is_file()
                or "Frozen Base TEST predictions do not reproduce the source's saved score arrays." not in known_log.read_text(encoding="utf-8")):
            raise ValueError("Unapproved partial validation is restricted to the diagnosed CPU transfer failure.")
    elif not isinstance(recovery, dict) or set(recovery.get("verified_reuse", {})) != set(PARTIAL_MODELS):
        raise ValueError("Explicit transfer recovery lacks its four approved reuse hash pins.")
    command = state.get("current_command", [])
    def argument(flag):
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise ValueError("The recorded partial transfer command has missing or ambiguous evidence.")
        return command[command.index(flag) + 1]
    if (state.get("current_step") != "transfer_six_models"
            or Path(state.get("source_manifest", "")).resolve() != manifest_path
            or argument("--bootstrap-iterations") != "1000" or argument("--models") != "all"
            or Path(argument("--manifest")).resolve() != manifest_path
            or Path(argument("--output-dir")).resolve() != directory):
        raise ValueError("The saved transfer command does not establish the prescribed source and bootstrap budget.")
    partitions_dir = Path(partitions_dir).resolve()
    prepared = load_partition_manifest(partitions_dir)
    manifest = _read_json(manifest_path)
    pin = manifest.get("transfer_partitions", {})
    partition_file = partitions_dir / "partition_manifest.json"
    pinned_partition = Path(pin.get("path", ""))
    pinned_partition = (pinned_partition if pinned_partition.is_absolute()
                        else PROJECT_ROOT / pinned_partition).resolve()
    partition_digest = file_sha256(partition_file)
    if pinned_partition != partition_file or pin.get("sha256") != partition_digest:
        raise ValueError("The partial transfer cohorts do not match the explicit prepared-partition pin.")
    base_labels = np.load(partitions_dir / "base_test_y.npy", allow_pickle=False)
    base_indices = np.load(partitions_dir / "base_test_row_indices.npy", allow_pickle=False)
    base_dev_indices = np.load(partitions_dir / "base_dev_row_indices.npy", allow_pickle=False)
    records, provenance = {}, {}
    source_names = {"config.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy",
                    "completed.json", "test_row_indices.npy", "model.joblib"}
    array_names = {"y_test.npy", "y_test_scores.npy", "test_row_indices.npy", "original_y_test.npy",
                   "original_y_test_scores.npy", "original_test_row_indices.npy"}
    for model in PARTIAL_MODELS:
        model_dir = directory / model
        if {p.name for p in model_dir.iterdir()} != {"source_audit.json", *VARIANTS}:
            raise ValueError(f"Incomplete or unexpected partial model files: {model}")
        source_file = model_dir / "source_audit.json"
        record = _read_json(source_file)
        run, config = resolve_transfer_source(manifest_path, model)
        threshold = config.get("threshold_exact")
        if (not isinstance(threshold, (int, float)) or not np.isfinite(threshold)
                or Path(record.get("source_run", "")).resolve() != run
                or record.get("threshold_used") != threshold or record.get("threshold_source") != "config.threshold_exact"
                or config.get("dataset_hash_sha256") != prepared["base"]["data_provenance"]["raw_file_sha256"]
                or set(record.get("source_artefacts_sha256", {})) != source_names
                or set(record.get("variants", {})) != set(VARIANTS)):
            raise ValueError(f"The partial model does not identify its unchanged pinned Base source: {model}")
        for name, digest in record["source_artefacts_sha256"].items():
            if file_sha256(run / name) != digest:
                raise ValueError(f"A partial model source artefact changed: {model}/{name}")
        data_provenance = config.get("data_provenance", {})
        if (data_provenance.get("raw_file_sha256") != config["dataset_hash_sha256"]
                or data_provenance.get("feature_columns") != prepared["feature_columns"]
                or data_provenance.get("test_indices_sha256") != array_sha256(base_indices)
                or data_provenance.get("dev_indices_sha256") != array_sha256(base_dev_indices)
                or config.get("train_samples") != len(base_dev_indices)):
            raise ValueError(f"The partial model source provenance differs from its prepared schema/splits: {model}")
        labels = np.load(run / "y_test.npy", allow_pickle=False)
        scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
        if (not np.array_equal(labels, base_labels)
                or not np.array_equal(np.load(run / "test_row_indices.npy", allow_pickle=False), base_indices)
                or scores.shape != labels.shape or not np.isfinite(scores).all()
                or not np.all((scores >= 0) & (scores <= 1)) or len(labels) != config["test_samples"]):
            raise ValueError(f"The partial Base evidence is not aligned with its prepared population: {model}")
        validation = record.get("base_test_prediction_validation", {})
        error = validation.get("maximum_absolute_score_difference", np.inf)
        if (validation.get("rows") != len(labels) or validation.get("labels_exactly_equal") is not True
                or not isinstance(validation.get("scores_bitwise_equal"), bool)
                or validation.get("score_absolute_tolerance") != 1e-12
                or validation.get("score_relative_tolerance") != 1e-10
                or validation.get("decisions_changed_at_frozen_threshold") != 0
                or not np.isfinite(error) or not 0 <= error <= 1e-12 + 1e-10 * np.max(np.abs(scores))):
            raise ValueError(f"The partial model lacks its successful original Base prediction validation: {model}")
        expected_base = compute_all_metrics(labels, scores, threshold)
        _equal_metrics(validation.get("recomputed_metrics", {}), expected_base, f"{model}/Base")
        saved_base = _read_json(run / "metrics_test.json")
        _equal_metrics({key: saved_base.get(key) for key in expected_base}, expected_base, f"{model}/source Base")
        evidence = {"source_audit": {"path": str(source_file), "sha256": file_sha256(source_file)},
                    "source_artefacts_sha256": record["source_artefacts_sha256"], "variant_metrics": {}}
        for variant, label in VARIANTS.items():
            target, reference = model_dir / variant, record["variants"][variant]
            if reference.get("directory") != f"{model}/{variant}" or {p.name for p in target.iterdir()} != array_names | {"metrics_test.json"}:
                raise ValueError(f"An incomplete or unexpected partial Variant is not reusable: {model}/{variant}")
            metric_file = target / "metrics_test.json"
            if file_sha256(metric_file) != reference.get("metrics_test_sha256"):
                raise ValueError(f"A partial Variant metrics file changed: {model}/{variant}")
            result = _read_json(metric_file)
            if set(result.get("artefacts", {})) != array_names:
                raise ValueError("All six partial Variant arrays must be recorded.")
            arrays = {}
            for name, item in result["artefacts"].items():
                path = target / name
                if file_sha256(path) != item.get("file_sha256"):
                    raise ValueError(f"A partial Variant array changed: {model}/{variant}/{name}")
                values = np.load(path, allow_pickle=False)
                if (values.ndim != 1 or list(values.shape) != item.get("shape") or str(values.dtype) != item.get("dtype")
                        or array_sha256(values) != item.get("array_sha256") or not np.isfinite(values).all()):
                    raise ValueError("Partial Variant array content, shape or dtype is inconsistent.")
                arrays[name] = values
            cohort = partitions_dir / variant
            kept = np.load(cohort / "kept_test_positions.npy", allow_pickle=False)
            original_labels = np.load(cohort / "original_test_y.npy", allow_pickle=False)
            original_indices = np.load(cohort / "original_test_row_indices.npy", allow_pickle=False)
            for name, values in (("original_y_test.npy", original_labels),
                                 ("original_test_row_indices.npy", original_indices),
                                 ("y_test.npy", original_labels[kept]), ("test_row_indices.npy", original_indices[kept])):
                if not np.array_equal(arrays[name], values):
                    raise ValueError("Partial Variant labels or row membership differ from the prepared cohort.")
            original_scores, kept_scores = arrays["original_y_test_scores.npy"], arrays["y_test_scores.npy"]
            if (original_scores.shape != original_labels.shape or not np.array_equal(kept_scores, original_scores[kept])
                    or not np.isin(original_labels, [0, 1]).all()
                    or not np.all((original_scores >= 0) & (original_scores <= 1))
                    or not np.issubdtype(arrays["test_row_indices.npy"].dtype, np.integer)
                    or not np.issubdtype(arrays["original_test_row_indices.npy"].dtype, np.integer)):
                raise ValueError("Partial original and retained scores/indices are not valid aligned probability evidence.")
            audit = _read_json(cohort / "partition_audit.json")
            if (result.get("variant_label") != label or result.get("threshold_used") != threshold
                    or result.get("threshold_source") != record["threshold_source"]
                    or result.get("population") != audit["kept_population"]
                    or result.get("excluded_population") != audit["excluded_population"]
                    or result.get("score_scale") != "probability"
                    or not np.isfinite(result.get("evaluation_seconds", np.nan)) or result["evaluation_seconds"] < 0):
                raise ValueError("The partial Variant has incompatible population, threshold or provenance.")
            expected, full = _metrics(original_labels[kept], kept_scores, threshold)
            metrics = result.get("metrics", {})
            _equal_metrics({key: value for key, value in metrics.items() if key != "bootstrap_ci"}, expected, f"{model}/{variant}")
            _equal_metrics(result.get("metrics_full_precision", {}), full, f"{model}/{variant}/full precision")
            intervals = metrics.get("bootstrap_ci", {})
            if set(intervals) != {"PR-AUC_ci", "ROC-AUC_ci", "F2_ci"}:
                raise ValueError("All three prescribed saved bootstrap intervals are required for partial reuse.")
            for bounds in intervals.values():
                if not isinstance(bounds, list) or len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 <= bounds[0] <= bounds[1] <= 1:
                    raise ValueError("Invalid saved partial bootstrap intervals.")
            if reference.get("metrics") != metrics:
                raise ValueError("Source audit and Variant metrics disagree.")
            original = result.get("original_population_same_frozen_model", {})
            original_metrics, original_full = _metrics(original_labels, original_scores, threshold)
            if original.get("role") != "newly_scored_original_population_not_a_retuned_or_historical_run":
                raise ValueError("The partial original population is not from the same frozen model.")
            _equal_metrics(original.get("metrics", {}), original_metrics, f"{model}/{variant}/original")
            _equal_metrics(original.get("metrics_full_precision", {}), original_full, f"{model}/{variant}/original full precision")
            if result.get("historical_original_record") != _historical_variant_metrics(manifest_path, model, variant):
                raise ValueError("The preserved historical context changed after the partial transfer.")
            evidence["variant_metrics"][variant] = {"path": str(metric_file), "sha256": file_sha256(metric_file)}
        records[model], provenance[model] = record, evidence
    if recovery is not None and provenance != recovery["verified_reuse"]:
        raise ValueError("Partial transfer evidence differs from the explicitly approved reuse hash pins.")
    return {"directory": str(directory), "models": records, "reuse_provenance": provenance,
            "partition_manifest_sha256": partition_digest, "bootstrap_iterations": 1000}


def evaluate_transfer(partitions_dir, models, *, manifest_path=DEFAULT_MANIFEST_PATH,
                      output_root=TRANSFER_ROOT, bootstrap_iterations=0,
                      device="cpu", batch_size=2048, output_dir=None, resume_verified_partial=None):
    """Score original TESTs once, then evaluate their predeclared retained rows."""
    if bootstrap_iterations < 0 or batch_size < 1:
        raise ValueError("Bootstrap iterations must be non-negative and batch size positive.")
    explicit_destination = validate_output_root(output_dir) if output_dir is not None else None
    if resume_verified_partial is not None:
        explicit_destination = _resume_request(resume_verified_partial, manifest_path, models, device,
                                               batch_size, bootstrap_iterations, output_dir)
    elif explicit_destination is not None and explicit_destination.exists():
        raise FileExistsError("An explicit transfer output directory must not already exist, even if incomplete.")
    partitions_dir = Path(partitions_dir).resolve()
    prepared = load_partition_manifest(partitions_dir)
    base_dev, base_test, base_y_dev, base_y_test, base_meta = load_dataset("baf_base", return_metadata=True)
    validate_schema(base_test, prepared["feature_columns"])
    if base_meta["raw_file_sha256"] != prepared["base"]["data_provenance"]["raw_file_sha256"]:
        raise ValueError("Base data changed after partition preparation.")
    for name, values in (("base_dev_row_indices.npy", base_meta["dev_indices"]),
                         ("base_test_row_indices.npy", base_meta["test_indices"]),
                         ("base_test_y.npy", np.asarray(base_y_test))):
        if not np.array_equal(np.load(partitions_dir / name, allow_pickle=False), values):
            raise ValueError("Base split membership changed after partition preparation.")
    del base_dev, base_y_dev
    gc.collect()
    # Resolve all requested pins before creating an evaluation directory.
    # In particular, all-model evaluation cannot silently skip an absent FT run.
    selected_sources = {}
    for model_name in models:
        run_dir, config = resolve_transfer_source(manifest_path, model_name)
        selected_sources[model_name] = (run_dir, config, file_sha256(run_dir / "config.json"))
    reused, prevalidated_ft = None, None
    if resume_verified_partial is not None:
        destination = explicit_destination
        reused = verify_partial_transfer(destination, manifest_path, partitions_dir)
        scorer = load_frozen_scorer("fttransformer", manifest_path, device=device, batch_size=batch_size,
                                    selected_source=selected_sources["fttransformer"])
        validation = validate_base_predictions(scorer, base_test, base_y_test, base_meta)
        # Check the entire partial again after FT inference, still before writing anything.
        if verify_partial_transfer(destination, manifest_path, partitions_dir) != reused:
            raise ValueError("Verified partial evidence changed during FT Base validation.")
        prevalidated_ft = (scorer, validation)
    elif explicit_destination is None:
        destination = _new_directory(output_root, "evaluations")
    else:
        destination = explicit_destination
        destination.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "evaluating", "protocol_version": PROTOCOL_VERSION,
        "partition_directory": str(partitions_dir),
        "partition_manifest_sha256": file_sha256(partitions_dir / "partition_manifest.json"),
        "source_manifest": str(Path(manifest_path).resolve()),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_code_sha256": file_sha256(__file__), "software_versions": software_versions(),
        "bootstrap_iterations": bootstrap_iterations, "bootstrap_seed": 42,
        "bootstrap_interpretation": "Conditional holdout uncertainty with fixed fitted model and fixed Base threshold; no repeated-seed inference.",
        "threshold_selection_on_variants": False, "models": {},
    }
    if reused is not None:
        summary["verified_partial_reuse"] = {
            "mode": "explicit_four_complete_classical_families_only",
            "models": reused["reuse_provenance"], "partition_manifest_sha256": reused["partition_manifest_sha256"],
            "new_fits": 0, "reused_model_inference_repeated": False, "reused_model_bootstrap_recomputed": False,
        }
    for model_name in models:
        if reused is not None and model_name in PARTIAL_MODELS:
            summary["models"][model_name] = reused["models"][model_name]
            print(f"REUSE verified complete transfer for {model_name}; no scoring or bootstrap repeated", flush=True)
            continue
        print(f"Loading explicitly pinned {model_name} baseline", flush=True)
        if model_name == "fttransformer" and prevalidated_ft is not None:
            scorer, validation = prevalidated_ft
            prevalidated_ft = None
        else:
            scorer = load_frozen_scorer(model_name, manifest_path, device=device, batch_size=batch_size,
                                        selected_source=selected_sources[model_name])
            validation = validate_base_predictions(scorer, base_test, base_y_test, base_meta)
        model_dir = destination / model_name
        model_dir.mkdir(exist_ok=False)
        record = {"source_run": str(scorer.run_dir), "source_artefacts_sha256": scorer.source_artefacts,
                  "threshold_used": scorer.threshold, "threshold_source": scorer.threshold_source,
                  "base_test_prediction_validation": validation, "variants": {}}
        if model_name == "fttransformer" and reused is not None:
            record["inference_execution"] = {"device": device, "batch_size": batch_size,
                                             "numeric_dtype": "float32", "categorical_dtype": "int64"}
        for variant, label in VARIANTS.items():
            unused_dev, target_test, unused_y_dev, labels, metadata = load_dataset(variant, return_metadata=True)
            del unused_dev, unused_y_dev
            validate_schema(target_test, prepared["feature_columns"])
            partition_dir = partitions_dir / variant
            audit = _read_json(partition_dir / "partition_audit.json")
            if metadata["raw_file_sha256"] != audit["data_provenance"]["raw_file_sha256"]:
                raise ValueError("Variant raw data changed after partition preparation.")
            original_indices = np.load(partition_dir / "original_test_row_indices.npy", allow_pickle=False)
            original_labels = np.load(partition_dir / "original_test_y.npy", allow_pickle=False)
            if not np.array_equal(original_indices, metadata["test_indices"]) or not np.array_equal(original_labels, np.asarray(labels)):
                raise ValueError("Variant TEST labels or row order changed after preparation.")
            kept = np.load(partition_dir / "kept_test_positions.npy", allow_pickle=False)
            start = time.perf_counter()
            original_scores = scorer.score(target_test)
            kept_scores, kept_labels = original_scores[kept], original_labels[kept]
            metrics, full_precision = _metrics(kept_labels, kept_scores, scorer.threshold)
            original_metrics, original_full = _metrics(original_labels, original_scores, scorer.threshold)
            if bootstrap_iterations:
                metrics["bootstrap_ci"] = bootstrap_ci(kept_labels, kept_scores, scorer.threshold,
                                                         n_bootstrap=bootstrap_iterations, random_state=42)
            target_dir = model_dir / variant
            target_dir.mkdir(exist_ok=False)
            artefacts = {}
            for name, values in (("y_test.npy", kept_labels), ("y_test_scores.npy", kept_scores),
                                 ("test_row_indices.npy", original_indices[kept]),
                                 ("original_y_test.npy", original_labels),
                                 ("original_y_test_scores.npy", original_scores),
                                 ("original_test_row_indices.npy", original_indices)):
                _save_array(target_dir, name, values, artefacts)
            variant_record = {
                "variant_label": label, "metrics": metrics, "metrics_full_precision": full_precision,
                "threshold_used": scorer.threshold, "threshold_source": scorer.threshold_source,
                "population": audit["kept_population"], "excluded_population": audit["excluded_population"],
                "original_population_same_frozen_model": {
                    "role": "newly_scored_original_population_not_a_retuned_or_historical_run",
                    "metrics": original_metrics, "metrics_full_precision": original_full,
                },
                "historical_original_record": _historical_variant_metrics(manifest_path, model_name, variant),
                "score_scale": "negated_decision_function" if model_name == "ocsvm" else "probability",
                "brier_interpretation": ("Scores are clipped to [0,1] for continuity with historical metrics; not a calibrated-probability assessment."
                                         if model_name == "ocsvm" else "Probability Brier score."),
                "evaluation_seconds": time.perf_counter() - start, "artefacts": artefacts,
            }
            _write_json(target_dir / "metrics_test.json", variant_record)
            record["variants"][variant] = {"directory": f"{model_name}/{variant}",
                                            "metrics_test_sha256": file_sha256(target_dir / "metrics_test.json"),
                                            "metrics": metrics}
            print(f"{model_name} / {label}: AP={full_precision['average_precision']:.6f}; "
                  f"F2={metrics['F2']:.6f}; kept n={len(kept_labels):,}", flush=True)
            del target_test, original_scores
            gc.collect()
        _write_json(model_dir / "source_audit.json", record)
        summary["models"][model_name] = record
        del scorer
        gc.collect()
    summary["status"] = "complete"
    _write_json(destination / "transfer_manifest.json", summary)
    print(f"Corrected transfer results: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_ORDER) + ["all"], default=["all"])
    parser.add_argument("--prepare-only", action="store_true", help="Prepare exact-profile-disjoint indices without loading or scoring models.")
    parser.add_argument("--partitions-dir", type=Path, help="Explicit completed partition directory; never choose the latest implicitly.")
    parser.add_argument("--output-dir", type=Path,
                        help="Explicit new evaluation directory; refuse any existing destination, including a partial run.")
    parser.add_argument("--resume-verified-partial", type=Path,
                        help="Explicit recovery of the four verified classical families in derived/transfer; CUDA/2048 only.")
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    transfer_root = args.manifest.resolve().parent / "transfer"
    models = list(MODEL_ORDER) if "all" in args.models else list(dict.fromkeys(args.models))
    if args.bootstrap_iterations < 0 or args.batch_size < 1:
        parser.error("Bootstrap iterations must be non-negative and batch size positive.")
    if args.prepare_only and args.partitions_dir is not None:
        parser.error("--prepare-only creates new partitions; do not supply --partitions-dir.")
    if args.prepare_only and args.output_dir is not None:
        parser.error("--output-dir selects an evaluation destination; do not combine it with --prepare-only.")
    if args.resume_verified_partial is not None:
        if args.prepare_only or args.partitions_dir is None:
            parser.error("Verified partial recovery requires explicit existing partitions and cannot prepare new ones.")
        _resume_request(args.resume_verified_partial, args.manifest, models, args.device,
                        args.batch_size, args.bootstrap_iterations, args.output_dir)
    if args.prepare_only:
        with exclusive_resource(transfer_root, "baf_training_ram", "BAF transfer partition preparation"):
            prepare_partitions(output_root=transfer_root, manifest_path=args.manifest)
        return
    # Check requested source availability before reading all six large CSVs.
    if args.output_dir is not None:
        output_dir = validate_output_root(args.output_dir)
        if output_dir.exists() and args.resume_verified_partial is None:
            parser.error("The explicit transfer output directory already exists; no overwrite or partial-run reuse is allowed.")
    for model_name in models:
        resolve_transfer_source(args.manifest, model_name)
    with exclusive_resource(transfer_root, "baf_training_ram", "Frozen BAF transfer inference"):
        partitions = args.partitions_dir or prepare_partitions(output_root=transfer_root, manifest_path=args.manifest)
        evaluate_transfer(partitions, models, manifest_path=args.manifest,
                          output_root=transfer_root,
                          bootstrap_iterations=args.bootstrap_iterations,
                          device=args.device, batch_size=args.batch_size, output_dir=args.output_dir,
                          resume_verified_partial=args.resume_verified_partial)


if __name__ == "__main__":
    main()
