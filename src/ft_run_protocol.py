"""Operational provenance and exact Optuna-state recovery for FT runs.

These helpers do not fit models or choose experimental outcomes. A legacy TPE
state is reconstructed only when every recorded parameter can be reproduced
exactly; otherwise recovery fails rather than silently resetting its RNG.
"""
import copy
import hashlib
import json
from pathlib import Path

import optuna

from experiment_protocol import (
    DEFAULT_RESULTS_ROOT, PROTOCOL_VERSION, file_sha256, load_sampler_checkpoint,
    resolve_baseline_run, save_sampler_checkpoint,
)
from models.fttransformer import suggest_hyperparams


def validate_reference(config, dataset_name, dataset_hash, split_metadata,
                       missing_policy, scheduler_horizon, historical=False):
    """Validate dataset, representation and split before borrowing baseline HP."""
    if (config.get("dataset") != dataset_name or config.get("model") != "fttransformer"
            or config.get("strategy") != "none"):
        raise ValueError("The FT reference must be a baseline for the same dataset/model.")
    if config.get("dataset_hash_sha256") != dataset_hash:
        raise ValueError("The FT baseline and current raw dataset hashes differ.")
    metadata = split_metadata or {}
    if config.get("sample_fraction") != metadata.get("sample_fraction"):
        raise ValueError("The FT baseline and current sample fractions differ.")
    if config.get("split_seed") != metadata.get("split_seed", 42):
        raise ValueError("The FT baseline and current split seeds differ.")
    if config.get("train_samples") != len(metadata.get("dev_indices", [])):
        raise ValueError("The FT baseline and current DEV sizes differ.")
    if config.get("test_samples") != len(metadata.get("test_indices", [])):
        raise ValueError("The FT baseline and current TEST sizes differ.")
    provenance = config.get("data_provenance") or {}
    for split in ("dev", "test"):
        key = f"{split}_indices_sha256"
        if provenance.get(key) != metadata.get(key) and (not historical or provenance.get(key)):
            raise ValueError(f"The FT baseline and current {split.upper()} indices differ.")
    if historical:
        # Historical BAF HP are deliberately borrowed for a predefined sensitivity.
        if dataset_name != "baf_base":
            raise ValueError("Historical FT parameter borrowing is restricted to BAF.")
        if config.get("max_epochs", scheduler_horizon) != scheduler_horizon:
            raise ValueError("The historical FT search used a different epoch horizon.")
    else:
        if config.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("A revised FT baseline is required for revised interventions.")
        if config.get("missing_policy", "preserve") != missing_policy:
            raise ValueError("The FT baseline and intervention missing-value policies differ.")
        if config.get("scheduler_horizon") != scheduler_horizon:
            raise ValueError("The FT baseline and intervention scheduler horizons differ.")


def load_reference(dataset_name, dataset_hash, split_metadata, missing_policy,
                   scheduler_horizon, *, results_root=None, manifest_path=None,
                   explicit_run=None, historical=False):
    run = resolve_baseline_run(dataset_name, "fttransformer", results_root=results_root,
                               manifest_path=manifest_path, explicit_run=explicit_run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    validate_reference(config, dataset_name, dataset_hash, split_metadata,
                       missing_policy, scheduler_horizon, historical)
    if not historical:
        required = ("completed.json", "model.pt", "preprocessors.joblib",
                    "validation_model.pt", "validation_preprocessors.joblib",
                    "y_val.npy", "y_val_scores.npy")
        if not all((run / filename).is_file() for filename in required):
            raise ValueError("The selected FT baseline is not fully complete.")
    return copy.deepcopy(config["best_params"]), run, config


def reconstruct_tpe_state(study, seed=42):
    """Replay recorded suggestions/feedback in memory, without any model fitting."""
    replay = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    recorded = study.trials
    for position, original in enumerate(recorded):
        if original.number != position or not original.state.is_finished():
            raise ValueError("TPE reconstruction requires consecutive finished trials.")
        trial = replay.ask()
        suggest_hyperparams(trial)
        if trial.params != original.params:
            raise ValueError("Recorded FT parameters cannot be reproduced exactly; RNG recovery aborted.")
        for step, value in sorted(original.intermediate_values.items()):
            trial.report(value, step)
        if original.state == optuna.trial.TrialState.COMPLETE:
            replay.tell(trial, original.value)
        else:
            replay.tell(trial, state=original.state)
    study.sampler = replay.sampler
    return {"policy": "exact_parameter_replay_without_model_fitting", "reconstructed_trials": len(recorded)}


def open_study(dataset_name, dataset_hash, split_metadata, missing_policy,
               scheduler_horizon, *, results_root=None, recover_interrupted=False,
               adopt_legacy=False, seed=42, validation_fraction=0.2, patience=15):
    """Open the same FT study path, with provenance and persistent sampler state."""
    directory = Path(results_root or DEFAULT_RESULTS_ROOT) / "studies"
    directory.mkdir(parents=True, exist_ok=True)
    # Preserve the path/name of the primary FT study already running in memory.
    path = directory / f"{dataset_name}_ft_{missing_policy}.sqlite3"
    sampler_path = path.with_suffix(".sampler.joblib")
    sampler, sampler_info = load_sampler_checkpoint(sampler_path)
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{path.resolve().as_posix()}")
    study = optuna.create_study(
        direction="maximize", sampler=sampler or optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage=storage, study_name=f"{dataset_name}_ft_{missing_policy}_{PROTOCOL_VERSION}",
        load_if_exists=True,
    )
    metadata = split_metadata or {}
    protocol = {
        "protocol_version": PROTOCOL_VERSION, "dataset": dataset_name,
        "dataset_hash_sha256": dataset_hash, "missing_policy": missing_policy,
        "sample_fraction": metadata.get("sample_fraction"),
        "dev_indices_sha256": metadata.get("dev_indices_sha256"),
        "test_indices_sha256": metadata.get("test_indices_sha256"),
        "features": metadata.get("feature_columns"), "scheduler_horizon": scheduler_horizon,
        "validation_fraction": validation_fraction, "patience": patience, "seed": seed,
        "model_source_sha256": file_sha256(Path(__file__).parent / "models/fttransformer.py"),
    }
    try:
        recorded_protocol = study.user_attrs.get("data_protocol")
        if recorded_protocol is not None and recorded_protocol != protocol:
            raise ValueError("The FT study has incompatible dataset/split/sample/representation/training provenance.")
        if recorded_protocol is None and study.trials and not adopt_legacy:
            raise ValueError("Legacy FT studies require explicit --adopt-legacy-study after provenance review.")
        interrupted = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        if interrupted and not recover_interrupted:
            raise ValueError("Confirm that no FT job is active before --recover-interrupted-trials.")
        for trial in interrupted:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        checkpoint_number = (sampler_info or {}).get("completed_trial_number")
        if study.trials and (sampler is None or checkpoint_number != len(study.trials) - 1):
            sampler_info = reconstruct_tpe_state(study, seed)
            save_sampler_checkpoint(study, study.trials[-1], sampler_path)
        study.set_user_attr("data_protocol", protocol)
        study.set_user_attr("data_protocol_sha256", hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest())
        return study, storage, {
            "optuna_storage": str(path.resolve()), "optuna_study_name": study.study_name,
            "optuna_study_signature": study.user_attrs["data_protocol_sha256"],
            "optuna_sampler_recovery": sampler_info,
            "legacy_study_adopted": bool(recorded_protocol is None and study.trials),
        }
    except BaseException:
        storage.remove_session()
        storage.engine.dispose()
        raise
