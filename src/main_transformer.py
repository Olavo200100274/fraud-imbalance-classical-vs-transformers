"""
FT-Transformer Training Script — Fraud Detection Pipeline
==========================================================
Standalone script for training the FT-Transformer model.
Separate from main.py because:
  - Holdout validation (not 5-fold CV) — DL training is too expensive for CV
  - PyTorch training loop with epoch-based early stopping
  - GPU training with CUDA tensors

Follows the same 4-phase protocol as main.py:
  strategy=none (baseline):
    [1/4] Hyperparameter tuning — Optuna TPE, 50 trials, holdout val, PR-AUC
    [2/4] Threshold selection   — max-F2 on validation set
    [3/4] Final training        — best HP on full DEV set (train+val)
    [4/4] Test evaluation       — holdout test with τ from step 2

  strategy=<resampling|weights>:
    [1/3] Load baseline HP      — no re-tuning
    [2/3] Train + threshold     — with resampling/weighted loss, early stop on val
    [3/3] Test evaluation       — holdout test

Usage:
    cd src/
    python main_transformer.py --dataset ulb --n_trials 50
    python main_transformer.py --dataset baf_base --strategy all
    python main_transformer.py --dataset ulb --sample 0.01 --n_trials 3  # smoke test
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from torch.utils.data import DataLoader

optuna.logging.set_verbosity(optuna.logging.WARNING)

from data import load_dataset, get_dataset_info, DATASET_REGISTRY
from evaluation.metrics import (
    find_threshold_maximizing_f2,
    compute_all_metrics,
    bootstrap_ci,
)
from save_load import save_run, finalise_run
from missing_values import numeric_pipeline, MISSING_POLICIES
from experiment_protocol import (resolve_baseline_run, software_versions, PROTOCOL_VERSION,
                                 file_sha256, optuna_trials_record, save_sampler_checkpoint)
from experiment_protocol import capture_source_provenance, validate_revision_destinations
from ft_run_protocol import load_reference, open_study
from revision_resources import baf_training_resource
from strategies.balancing import (
    STRATEGY_NAMES,
    RESAMPLING_STRATEGIES,
    get_sampler,
)
from models.fttransformer import (
    FTTransformer,
    TabularDataset,
    train_one_epoch,
    evaluate,
    build_model,
    suggest_hyperparams,
)


# ── Constants ─────────────────────────────────────────────────────────────
SPLIT_SEED = 42
VAL_FRACTION = 0.2          # 20% of DEV for validation
MAX_EPOCHS = 200
PATIENCE = 15               # early stopping patience (epochs)
EVAL_BATCH_SIZE = 2048
BOOTSTRAP_ITERATIONS = 1000
NUM_WORKERS = 0             # DataLoader workers (0 = main process, safest on Windows)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = _PROJECT_ROOT / "results"


# ── Reproducibility ──────────────────────────────────────────────────────
def _seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Preprocessing ─────────────────────────────────────────────────────────
def preprocess_for_transformer(X_train_df, X_other_df, missing_policy="preserve"):
    """
    Separate numeric and categorical preprocessing.

    - Numeric: SimpleImputer(mean) → StandardScaler  (fitted on X_train)
    - Categorical: OrdinalEncoder → integer indices   (fitted on X_train)

    Returns
    -------
    X_num_train, X_cat_train, X_num_other, X_cat_other,
    cat_cardinalities, d_numerical, num_pipeline, cat_encoder, num_cols, cat_cols
    """
    num_cols = X_train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = X_train_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric branch
    num_pipe = numeric_pipeline(missing_policy)
    X_num_train = np.asarray(num_pipe.fit_transform(X_train_df[num_cols]), dtype=np.float32)
    X_num_other = np.asarray(num_pipe.transform(X_other_df[num_cols]), dtype=np.float32)

    # Categorical branch
    X_cat_train = None
    X_cat_other = None
    cat_cardinalities = []
    cat_encoder = None

    if cat_cols:
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            dtype=np.int64,
        )
        X_cat_train = np.asarray(cat_encoder.fit_transform(X_train_df[cat_cols]), dtype=np.int64)
        X_cat_other = np.asarray(cat_encoder.transform(X_other_df[cat_cols]), dtype=np.int64)
        cat_cardinalities = [len(c) for c in cat_encoder.categories_]

    return (
        X_num_train, X_cat_train,
        X_num_other, X_cat_other,
        cat_cardinalities, X_num_train.shape[1],
        num_pipe, cat_encoder, num_cols, cat_cols,
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def _file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "file_not_found"


def _find_latest_run(dataset_label, model_name, strategy="none"):
    base = RESULTS_ROOT / dataset_label / model_name / strategy
    if not base.exists():
        return None
    runs = sorted(base.iterdir())
    return runs[-1] if runs else None


def _load_baseline_params(dataset_name, results_root=None, manifest_path=None, explicit_run=None,
                          *, dataset_hash=None, split_metadata=None, missing_policy="preserve"):
    """Load a complete, provenance-matched FT baseline without timestamp selection."""
    hp, run, config = load_reference(
        dataset_name, dataset_hash, split_metadata, missing_policy, MAX_EPOCHS,
        results_root=results_root, manifest_path=manifest_path, explicit_run=explicit_run)
    return hp, config.get("best_epoch", MAX_EPOCHS), run


def _save_validation_artefacts(run_dir, state, hp, d_num, cat_cards,
                              num_pipe, cat_enc, num_cols, cat_cols, final_num_pipe,
                              final_cat_enc, y_val, val_scores, best_epoch):
    """Persist the selected validation checkpoint and both fitted preprocessors."""
    torch.save({"model_state_dict": {k: v.cpu() for k, v in state.items()},
                "hyperparams": hp, "d_numerical": d_num,
                "cat_cardinalities": cat_cards, "num_cols": num_cols,
                "cat_cols": cat_cols, "best_epoch": best_epoch,
                "scheduler_horizon": MAX_EPOCHS}, Path(run_dir) / "validation_model.pt")
    joblib.dump({"num_preprocessor": final_num_pipe, "cat_encoder": final_cat_enc,
                 "num_cols": num_cols, "cat_cols": cat_cols},
                Path(run_dir) / "preprocessors.joblib")
    joblib.dump({"num_preprocessor": num_pipe, "cat_encoder": cat_enc,
                 "num_cols": num_cols, "cat_cols": cat_cols},
                Path(run_dir) / "validation_preprocessors.joblib")


def _resample_arrays(strategy, X_num, X_cat, y, d_num, stage, diagnostics):
    """Apply a sampler and record actual class sizes and Tomek removals."""
    from experiment_protocol import sampler_diagnostics
    combined = np.hstack([X_num, X_cat]) if X_cat is not None else X_num
    if strategy == "smotenc_control":
        from imblearn.over_sampling import SMOTENC
        from sklearn.preprocessing import OneHotEncoder
        if X_cat is None:
            raise ValueError("Categorical-aware control requires categorical inputs.")
        sampler = SMOTENC(categorical_features=list(range(d_num, combined.shape[1])),
                          categorical_encoder=OneHotEncoder(handle_unknown="ignore").set_output(transform="default"),
                          random_state=SPLIT_SEED)
    else:
        sampler = get_sampler(strategy, random_state=SPLIT_SEED)
    X_res, y_res = sampler.fit_resample(combined, np.asarray(y))
    diagnostics.append(sampler_diagnostics(sampler, combined, np.asarray(y), X_res, y_res,
                                           stage=stage))
    return (X_res[:, :d_num].astype(np.float32),
            X_res[:, d_num:].astype(np.int64) if X_cat is not None else None,
            np.asarray(y_res, dtype=np.float32))


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _print_results(m, ci=None):
    print("\n  -- Test Results --")
    print(f"  PR-AUC       : {m['PR-AUC']:.4f}", end="")
    if ci:
        print(f"  [95% CI: {ci['PR-AUC_ci'][0]:.4f} - {ci['PR-AUC_ci'][1]:.4f}]")
    else:
        print()
    print(f"  ROC-AUC      : {m['ROC-AUC']:.4f}", end="")
    if ci:
        print(f"  [95% CI: {ci['ROC-AUC_ci'][0]:.4f} - {ci['ROC-AUC_ci'][1]:.4f}]")
    else:
        print()
    print(f"  F1           : {m['F1']:.4f}")
    print(f"  F2           : {m['F2']:.4f}", end="")
    if ci:
        print(f"  [95% CI: {ci['F2_ci'][0]:.4f} - {ci['F2_ci'][1]:.4f}]")
    else:
        print()
    print(f"  Brier score  : {m['brier_score']:.4f}")
    print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
    print(f"  Alert rate   : {m['alert_rate']:.4%}")
    print(f"  FP/TP        : {m['FP/TP']:.2f}")
    print(f"  Threshold    : {m['threshold']:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Training core
# ═══════════════════════════════════════════════════════════════════════════

def _train_model(model, train_loader, val_loader, hp, device,
                 criterion=None, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 trial=None):
    """
    Train an FT-Transformer with early stopping.

    Returns
    -------
    best_state_dict : dict
    best_val_prauc : float
    best_epoch : int
    val_scores : np.ndarray (scores from best epoch)
    val_true : np.ndarray
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-7,
    )

    best_val_prauc = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    best_val_true = None
    best_val_scores = None

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        y_true_val, y_scores_val = evaluate(model, val_loader, device)
        val_prauc = float(average_precision_score(y_true_val, y_scores_val))
        if trial is None or (epoch + 1) % 10 == 0:
            print(f"    epoch={epoch + 1} val_AP={val_prauc:.6f} loss={train_loss:.6f}", flush=True)

        # Report to Optuna for pruning
        if trial is not None:
            trial.report(val_prauc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
            best_val_true = y_true_val
            best_val_scores = y_scores_val
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_state, best_val_prauc, best_epoch, best_val_scores, best_val_true


# ═══════════════════════════════════════════════════════════════════════════
#  Baseline (strategy=none)
# ═══════════════════════════════════════════════════════════════════════════

def run_baseline(X_train, X_test, y_train, y_test,
                 dataset_name, dataset_file, dataset_hash,
                 n_trials=50, results_root=None, manifest_path=None,
                 split_metadata=None, fixed_params_run=None, missing_policy="preserve",
                 recover_interrupted_trials=False, adopt_legacy_study=False,
                 source_provenance=None):
    """Full protocol for FT-Transformer baseline (strategy=none)."""
    _seed_everything(SPLIT_SEED)
    device = _get_device()

    print(f"\n{'=' * 60}")
    print(f"  FT-TRANSFORMER -- Strategy: None")
    print(f"{'=' * 60}")
    print(f"  Device: {device}")

    # ── Split DEV into train (80%) + val (20%) ─────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_FRACTION, stratify=y_train, random_state=SPLIT_SEED,
    )
    print(f"  DEV split: train={len(X_tr):,}, val={len(X_val):,}")

    # ── Preprocess ─────────────────────────────────────────────────
    (X_num_tr, X_cat_tr, X_num_val, X_cat_val,
     cat_cards, d_num, num_pipe, cat_enc, num_cols, cat_cols) = \
        preprocess_for_transformer(X_tr, X_val, missing_policy)

    val_loader = DataLoader(
        TabularDataset(X_num_val, X_cat_val, y_val.values),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    # ── [1/4] Optuna HPO ───────────────────────────────────────────
    if fixed_params_run is None:
        print(f"\n[1/4] Hyperparameter tuning ({n_trials} trials) ...")
    else:
        print("\n[1/4] Loading historical hyperparameters; no new HPO ...")
    t0 = time.time()

    def objective(trial):
        _seed_everything(SPLIT_SEED + trial.number)
        hp = suggest_hyperparams(trial)
        trial.set_user_attr("hp", hp)

        model = build_model(hp, d_num, cat_cards).to(device)

        train_loader = DataLoader(
            TabularDataset(X_num_tr, X_cat_tr, y_tr.values),
            batch_size=hp["batch_size"], shuffle=True, num_workers=NUM_WORKERS,
        )

        _, best_prauc, best_ep, _, _ = _train_model(
            model, train_loader, val_loader, hp, device, trial=trial,
            max_epochs=MAX_EPOCHS,
        )
        trial.set_user_attr("best_epoch", best_ep)
        return best_prauc

    tuning_trials, study_metadata = None, {}
    source_run = None
    if fixed_params_run is None:
        study, storage, study_metadata = open_study(
            dataset_name, dataset_hash, split_metadata, missing_policy, MAX_EPOCHS,
            results_root=results_root, recover_interrupted=recover_interrupted_trials,
            adopt_legacy=adopt_legacy_study, seed=SPLIT_SEED,
            validation_fraction=VAL_FRACTION, patience=PATIENCE)
        sampler_path = Path(study_metadata["optuna_storage"]).with_suffix(".sampler.joblib")
        try:
            remaining = max(0, n_trials - len(study.trials))
            print(f"  Optuna checkpoint: {len(study.trials)} existing, {remaining} remaining trials")
            study.optimize(objective, n_trials=remaining, gc_after_trial=True,
                           callbacks=[lambda study, trial: save_sampler_checkpoint(study, trial, sampler_path)])
            best_hp = copy.deepcopy(study.best_trial.user_attrs["hp"])
            best_epoch = study.best_trial.user_attrs.get("best_epoch", MAX_EPOCHS)
            tuning_value = study.best_value
            n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
            tuning_trials = optuna_trials_record(study)
        finally:
            storage.remove_session()
            storage.engine.dispose()
    else:
        best_hp, source_run, reference = load_reference(
            dataset_name, dataset_hash, split_metadata, missing_policy, MAX_EPOCHS,
            explicit_run=fixed_params_run, historical=True)
        best_epoch = reference.get("best_epoch", MAX_EPOCHS)
        tuning_value = None
        n_trials = 0
        n_pruned = 0
    tuning_time = time.time() - t0 if fixed_params_run is None else 0.0

    print(f"  HPO objective: {tuning_value}; fixed source: {fixed_params_run}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Pruned: {n_pruned}/{n_trials}")
    print(f"  Tuning time: {tuning_time:.1f}s")
    print(f"  Best HP: { {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best_hp.items()} }")

    # ── [2/4] Threshold selection on val set ───────────────────────
    print("\n[2/4] Threshold selection (max-F2 on val set) ...")

    _seed_everything(SPLIT_SEED)
    model = build_model(best_hp, d_num, cat_cards).to(device)
    train_loader = DataLoader(
        TabularDataset(X_num_tr, X_cat_tr, y_tr.values),
        batch_size=best_hp["batch_size"], shuffle=True, num_workers=NUM_WORKERS,
    )
    best_state, _, best_epoch, val_scores, val_true = _train_model(
        model, train_loader, val_loader, best_hp, device,
        max_epochs=MAX_EPOCHS,
    )
    model.load_state_dict(best_state)

    tau_final, best_f2_val = find_threshold_maximizing_f2(val_true, val_scores)
    print(f"  Threshold (max-F2): {tau_final:.6f}  (val F2={best_f2_val:.4f})")

    # ── [3/4] Final training on full DEV (train + val) ─────────────
    print("\n[3/4] Final training on full DEV set ...")
    t0_train = time.time()

    # Re-preprocess with all DEV data
    (X_num_dev, X_cat_dev, X_num_test, X_cat_test,
     cat_cards_final, d_num_final, num_pipe_final, cat_enc_final,
     num_cols_final, cat_cols_final) = \
        preprocess_for_transformer(X_train, X_test, missing_policy)

    _seed_everything(SPLIT_SEED)
    final_model = build_model(best_hp, d_num_final, cat_cards_final).to(device)
    dev_loader = DataLoader(
        TabularDataset(X_num_dev, X_cat_dev, y_train.values),
        batch_size=best_hp["batch_size"], shuffle=True, num_workers=NUM_WORKERS,
    )

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_hp["learning_rate"],
        weight_decay=best_hp["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-7,
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(best_epoch):
        train_one_epoch(final_model, dev_loader, optimizer, criterion, device)
        scheduler.step()

    train_time = time.time() - t0_train
    print(f"  Trained for {best_epoch} epochs in {train_time:.1f}s")

    # ── [4/4] Test evaluation ──────────────────────────────────────
    print("\n[4/4] Test evaluation ...")
    t0_infer = time.time()

    test_loader = DataLoader(
        TabularDataset(X_num_test, X_cat_test, y_test.values),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    y_true_test, y_test_scores = evaluate(final_model, test_loader, device)
    infer_time = time.time() - t0_infer

    metrics_test = compute_all_metrics(y_test, y_test_scores, tau_final)
    ci = bootstrap_ci(y_test, y_test_scores, tau_final, n_bootstrap=BOOTSTRAP_ITERATIONS) if BOOTSTRAP_ITERATIONS else None

    _print_results(metrics_test, ci)

    # ── Validation metrics (for metrics_cv.json compatibility) ─────
    val_metrics = compute_all_metrics(val_true, val_scores, tau_final)
    metrics_cv = {
        "validation": val_metrics,
        "note": "Single holdout validation split (not 5-fold CV)",
    }

    # ── Save checkpoint + artefacts ────────────────────────────────
    checkpoint = {
        "model_state_dict": final_model.cpu().state_dict(),
        "hyperparams": best_hp,
        "d_numerical": d_num_final,
        "cat_cardinalities": cat_cards_final,
        "num_cols": num_cols_final,
        "cat_cols": cat_cols_final,
        "best_epoch": best_epoch,
        "threshold": tau_final,
    }

    config = {
        "dataset": dataset_name,
        "dataset_file": dataset_file,
        "dataset_hash_sha256": dataset_hash,
        "split_seed": SPLIT_SEED,
        "split_ratio": "80/20 stratified",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "model": "fttransformer",
        "strategy": "none",
        "validation_split": f"{1 - VAL_FRACTION:.0%}/{VAL_FRACTION:.0%} within DEV",
        "scoring": "average_precision (PR-AUC)",
        "tuning": ("fixed historical hyperparameters; no new tuning" if fixed_params_run
                   else "Optuna (TPE sampler + MedianPruner)"),
        "n_trials": n_trials,
        "n_pruned": n_pruned,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "best_epoch": best_epoch,
        "threshold_rule": "maximise F2 on holdout validation set",
        "threshold_exact": tau_final,
        "sample_fraction": None if split_metadata is None else split_metadata.get("sample_fraction"),
        "best_params": best_hp,
        "tuning_time_s": round(tuning_time, 2),
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 4),
        "protocol_version": PROTOCOL_VERSION,
        "missing_policy": missing_policy,
        "scheduler_horizon": MAX_EPOCHS,
        "best_epoch_source": "seeded validation refit, maximum validation average precision",
        "best_params_source": str(fixed_params_run) if fixed_params_run else "Optuna on inner DEV holdout",
        "baseline_run": str(source_run) if source_run else None,
        "baseline_config_sha256": file_sha256(source_run / "config.json") if source_run else None,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        **study_metadata,
        **(source_provenance or {}),
        "software_versions": software_versions(),
    }

    # Save using existing save_run (model=checkpoint for torch)
    run_dir = save_run(
        model=checkpoint,
        metrics_cv=metrics_cv,
        metrics_test=metrics_test,
        y_test=np.asarray(y_test),
        y_test_scores=y_test_scores,
        config=config,
        model_name="fttransformer",
        strategy="none",
        dataset=dataset_name,
        bootstrap_ci=ci,
        model_type="torch",
        results_root=results_root,
        split_metadata=split_metadata,
        validation_evidence={"y_val": val_true, "y_val_scores": val_scores,
                             "row_indices": X_val.index.to_numpy()},
        tuning_trials=tuning_trials,
        manifest_path=manifest_path,
        defer_completion=True,
    )
    _save_validation_artefacts(run_dir, best_state, best_hp, d_num, cat_cards,
                              num_pipe, cat_enc, num_cols, cat_cols, num_pipe_final,
                              cat_enc_final, val_true, val_scores, best_epoch)
    finalise_run(run_dir, manifest_path, ("preprocessors.joblib", "validation_model.pt",
                                        "validation_preprocessors.joblib"))

    return final_model, metrics_test, run_dir


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy runs (resampling / class weights)
# ═══════════════════════════════════════════════════════════════════════════

def run_strategy(strategy, X_train, X_test, y_train, y_test,
                 dataset_name, dataset_file, dataset_hash, results_root=None,
                 manifest_path=None, split_metadata=None, baseline_run=None,
                 missing_policy="preserve", source_provenance=None):
    """FT-Transformer with an imbalance strategy (no re-tuning)."""
    _seed_everything(SPLIT_SEED)
    device = _get_device()

    print(f"\n{'=' * 60}")
    print(f"  FT-TRANSFORMER -- Strategy: {strategy}")
    print(f"{'=' * 60}")

    # Load baseline HP
    best_hp, best_epoch_baseline, selected_baseline = _load_baseline_params(
        dataset_name, results_root, manifest_path, baseline_run,
        dataset_hash=dataset_hash, split_metadata=split_metadata, missing_policy=missing_policy)
    diagnostics = []
    print(f"  Loaded baseline HP (best_epoch={best_epoch_baseline})")

    # ── Split DEV into train (80%) + val (20%) ─────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_FRACTION, stratify=y_train, random_state=SPLIT_SEED,
    )

    # ── Preprocess ─────────────────────────────────────────────────
    (X_num_tr, X_cat_tr, X_num_val, X_cat_val,
     cat_cards, d_num, num_pipe, cat_enc, num_cols, cat_cols) = \
        preprocess_for_transformer(X_tr, X_val, missing_policy)

    # ── Apply resampling strategy ──────────────────────────────────
    use_weights = (strategy == "weights")
    criterion = nn.BCEWithLogitsLoss()

    if strategy in RESAMPLING_STRATEGIES or strategy == "smotenc_control":
        print(f"  Applying {strategy} resampling ...")
        # Concatenate num + cat, resample, split back
        if X_cat_tr is not None:
            X_combined = np.hstack([X_num_tr, X_cat_tr])
        else:
            X_combined = X_num_tr

        X_num_tr, X_cat_tr, y_tr_values = _resample_arrays(
            strategy, X_num_tr, X_cat_tr, y_tr.values, d_num, "inner_training", diagnostics)
        print(f"  Resampled: {len(y_tr_values):,} samples ({int(y_tr_values.sum()):,} fraud)")
    elif use_weights:
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        y_tr_values = y_tr.values.astype(np.float32)
        print(f"  Using weighted loss: pos_weight={n_neg / n_pos:.2f}")
    else:
        y_tr_values = y_tr.values.astype(np.float32)

    # ── Train with early stopping on val ───────────────────────────
    print("\n[1/3] Training with early stopping ...")
    t0 = time.time()

    model = build_model(best_hp, d_num, cat_cards).to(device)
    train_loader = DataLoader(
        TabularDataset(X_num_tr, X_cat_tr, y_tr_values),
        batch_size=best_hp["batch_size"], shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        TabularDataset(X_num_val, X_cat_val, y_val.values),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    best_state, best_prauc, best_epoch, val_scores, val_true = _train_model(
        model, train_loader, val_loader, best_hp, device,
        criterion=criterion,
        max_epochs=MAX_EPOCHS,
    )
    model.load_state_dict(best_state)

    # ── [2/3] Threshold selection ──────────────────────────────────
    print("\n[2/3] Threshold selection (max-F2 on val set) ...")
    tau_final, best_f2_val = find_threshold_maximizing_f2(val_true, val_scores)
    print(f"  Threshold: {tau_final:.6f}  (val F2={best_f2_val:.4f})")

    # ── Final training on full DEV ─────────────────────────────────
    print("\n[3/3] Final training on full DEV + test evaluation ...")

    (X_num_dev, X_cat_dev, X_num_test, X_cat_test,
     cat_cards_f, d_num_f, num_pipe_f, cat_enc_f,
     num_cols_f, cat_cols_f) = \
        preprocess_for_transformer(X_train, X_test, missing_policy)

    # Apply resampling to full DEV if needed
    y_dev_values = y_train.values.astype(np.float32)
    final_criterion = nn.BCEWithLogitsLoss()

    if strategy in RESAMPLING_STRATEGIES or strategy == "smotenc_control":
        if X_cat_dev is not None:
            X_combined_dev = np.hstack([X_num_dev, X_cat_dev])
        else:
            X_combined_dev = X_num_dev
        X_num_dev, X_cat_dev, y_dev_values = _resample_arrays(
            strategy, X_num_dev, X_cat_dev, y_train.values, d_num_f, "full_dev", diagnostics)
    elif use_weights:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    _seed_everything(SPLIT_SEED)
    final_model = build_model(best_hp, d_num_f, cat_cards_f).to(device)
    dev_loader = DataLoader(
        TabularDataset(X_num_dev, X_cat_dev, y_dev_values),
        batch_size=best_hp["batch_size"], shuffle=True, num_workers=NUM_WORKERS,
    )

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_hp["learning_rate"],
        weight_decay=best_hp["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-7,
    )
    for epoch in range(best_epoch):
        train_one_epoch(final_model, dev_loader, optimizer, final_criterion, device)
        scheduler.step()

    train_time = time.time() - t0

    # ── Test evaluation ────────────────────────────────────────────
    t0_infer = time.time()
    test_loader = DataLoader(
        TabularDataset(X_num_test, X_cat_test, y_test.values),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    y_true_test, y_test_scores = evaluate(final_model, test_loader, device)
    infer_time = time.time() - t0_infer

    metrics_test = compute_all_metrics(y_test, y_test_scores, tau_final)
    ci = bootstrap_ci(y_test, y_test_scores, tau_final, n_bootstrap=BOOTSTRAP_ITERATIONS) if BOOTSTRAP_ITERATIONS else None

    _print_results(metrics_test, ci)

    val_metrics = compute_all_metrics(val_true, val_scores, tau_final)
    metrics_cv = {
        "validation": val_metrics,
        "note": f"Single holdout validation, strategy={strategy}",
    }

    checkpoint = {
        "model_state_dict": final_model.cpu().state_dict(),
        "hyperparams": best_hp,
        "d_numerical": d_num_f,
        "cat_cardinalities": cat_cards_f,
        "num_cols": num_cols_f,
        "cat_cols": cat_cols_f,
        "best_epoch": best_epoch,
        "threshold": tau_final,
    }

    config = {
        "dataset": dataset_name,
        "dataset_file": dataset_file,
        "dataset_hash_sha256": dataset_hash,
        "split_seed": SPLIT_SEED,
        "split_ratio": "80/20 stratified",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "model": "fttransformer",
        "strategy": strategy,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "validation_split": f"{1 - VAL_FRACTION:.0%}/{VAL_FRACTION:.0%} within DEV",
        "threshold_rule": "maximise F2 on holdout validation set",
        "threshold_exact": tau_final,
        "sample_fraction": None if split_metadata is None else split_metadata.get("sample_fraction"),
        "best_params": best_hp,
        "best_epoch": best_epoch,
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 4),
        "protocol_version": PROTOCOL_VERSION,
        "missing_policy": missing_policy,
        "scheduler_horizon": MAX_EPOCHS,
        "best_params_source": "loaded from the pinned baseline run; no re-tuning",
        "baseline_run": str(selected_baseline),
        "baseline_config_sha256": file_sha256(selected_baseline / "config.json"),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "software_versions": software_versions(),
        **(source_provenance or {}),
    }

    run_dir = save_run(
        model=checkpoint,
        metrics_cv=metrics_cv,
        metrics_test=metrics_test,
        y_test=np.asarray(y_test),
        y_test_scores=y_test_scores,
        config=config,
        model_name="fttransformer",
        strategy=strategy,
        dataset=dataset_name,
        bootstrap_ci=ci,
        model_type="torch",
        results_root=results_root,
        split_metadata=split_metadata,
        validation_evidence={"y_val": val_true, "y_val_scores": val_scores,
                             "row_indices": X_val.index.to_numpy()},
        sampler_diagnostics=diagnostics,
        manifest_path=manifest_path,
        defer_completion=True,
    )
    _save_validation_artefacts(run_dir, best_state, best_hp, d_num, cat_cards,
                              num_pipe, cat_enc, num_cols, cat_cols, num_pipe_f,
                              cat_enc_f, val_true, val_scores, best_epoch)
    finalise_run(run_dir, manifest_path, ("preprocessors.joblib", "validation_model.pt",
                                        "validation_preprocessors.joblib"))

    return final_model, metrics_test, run_dir


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="FT-Transformer — Fraud Detection Pipeline"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset: " + " | ".join(DATASET_REGISTRY.keys()),
    )
    parser.add_argument(
        "--strategy", type=str, nargs="+", default=["none"],
        choices=STRATEGY_NAMES + ["all", "smotenc_control"],
        help="Imbalance strategy (default: none = baseline)",
    )
    parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Optuna trials for baseline tuning (default: 50)",
    )
    parser.add_argument(
        "--sample", type=float, default=None,
        help="Stratified subsample fraction for smoke tests (e.g. 0.01)",
    )
    parser.add_argument("--results-root", type=Path, default=_PROJECT_ROOT / "results_revision" / "20261005")
    parser.add_argument("--run-manifest", type=Path, default=None)
    parser.add_argument("--baseline-run", type=Path, default=None)
    parser.add_argument("--fixed-params-run", type=Path, default=None)
    parser.add_argument("--missing-policy", choices=MISSING_POLICIES, default="preserve")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                        help="Use 200 for primary runs; shorter values are smoke tests only.")
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--recover-interrupted-trials", action="store_true")
    parser.add_argument("--adopt-legacy-study", action="store_true",
                        help="Adopt a reviewed legacy FT study lacking split/sample provenance.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Operational guard only: model mathematics, seeds and schedules are unchanged.
    args.results_root, args.run_manifest = validate_revision_destinations(args.results_root, args.run_manifest)
    source_provenance = capture_source_provenance(args.results_root)
    with baf_training_resource(args.dataset, args.results_root, "FT-Transformer BAF fitting"):
        _run_main(args, source_provenance)


def _run_main(args, source_provenance):
    global MAX_EPOCHS, BOOTSTRAP_ITERATIONS
    MAX_EPOCHS = args.max_epochs
    BOOTSTRAP_ITERATIONS = args.bootstrap_iterations
    if MAX_EPOCHS < 1 or BOOTSTRAP_ITERATIONS < 0 or args.n_trials < 1:
        raise ValueError("Epoch/trial budgets must be positive and bootstrap iterations non-negative.")
    torch.set_num_threads(4)

    # Resolve strategies
    strategies = args.strategy
    if "all" in strategies:
        strategies = [s for s in STRATEGY_NAMES if s != "none"]

    # Load data
    dataset_file, dataset_name = get_dataset_info(args.dataset)
    print(f"Loading {dataset_name} dataset ...")
    X_train, X_test, y_train, y_test, split_metadata = load_dataset(
        args.dataset, sample=args.sample, return_metadata=True)
    print(f"  Train : {len(X_train):,} samples ({int(y_train.sum())} fraud)")
    print(f"  Test  : {len(X_test):,} samples  ({int(y_test.sum())} fraud)")

    dataset_hash = _file_hash(dataset_file)
    device = _get_device()
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Run baseline first if "none" in strategies
    if "none" in strategies:
        run_baseline(
            X_train, X_test, y_train, y_test,
            dataset_name, dataset_file, dataset_hash,
            n_trials=args.n_trials,
            results_root=args.results_root, manifest_path=args.run_manifest,
            split_metadata=split_metadata, fixed_params_run=args.fixed_params_run,
            missing_policy=args.missing_policy,
            recover_interrupted_trials=args.recover_interrupted_trials,
            adopt_legacy_study=args.adopt_legacy_study,
            source_provenance=source_provenance,
        )
        strategies = [s for s in strategies if s != "none"]

    # Run other strategies
    for strategy in strategies:
        run_strategy(
            strategy, X_train, X_test, y_train, y_test,
            dataset_name, dataset_file, dataset_hash,
            results_root=args.results_root, manifest_path=args.run_manifest,
            split_metadata=split_metadata, baseline_run=args.baseline_run,
            missing_policy=args.missing_policy,
            source_provenance=source_provenance,
        )

    print(f"\n{'=' * 60}")
    print("  FT-Transformer runs complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
