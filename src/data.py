"""Deterministic loading with exact-profile deduplication before ULB splits.

BAF retains the historical random split and numerical absence sentinels.
Original CSV row positions are preserved in optional provenance metadata.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from experiment_protocol import array_sha256, file_sha256, PROTOCOL_VERSION

SPLIT_SEED = 42
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_REGISTRY = {}


def _register(name, csv_path, label):
    def decorator(fn):
        DATASET_REGISTRY[name] = {"loader": fn, "csv_path": csv_path, "label": label}
        return fn
    return decorator


def deduplicate_predictors(frame, target):
    """Keep the first exact predictor profile and reject conflicting labels."""
    predictors = [c for c in frame.columns if c != target]
    repeated = frame.duplicated(subset=predictors, keep=False)
    if repeated.any():
        conflicts = frame.loc[repeated].groupby(
            predictors, sort=False, dropna=False
        )[target].nunique(dropna=False)
        if (conflicts > 1).any():
            raise ValueError("Identical predictor profiles have conflicting labels.")
    kept = ~frame.duplicated(subset=predictors, keep="first")
    audit = {
        "policy": "exact_predictor_deduplication_before_all_splits",
        "raw_rows": int(len(frame)),
        "raw_positive_rows": int(frame[target].sum()),
        "retained_rows": int(kept.sum()),
        "retained_positive_rows": int(frame.loc[kept, target].sum()),
        "removed_rows": int((~kept).sum()),
        "removed_positive_rows": int(frame.loc[~kept, target].sum()),
        "predictor_columns": predictors,
        "conflicting_label_profiles": 0,
    }
    return frame.loc[kept].copy(), audit


def _split_frame(frame, target, *, deduplicate=False, sample=None):
    """Split eligible rows without fitting; retain original row positions."""
    if frame.isna().to_numpy().any():
        raise ValueError("The raw dataset contains unexpected NaN values.")
    if not frame.index.is_unique:
        raise ValueError("Raw CSV row positions must be unique.")
    if set(frame[target].unique()) != {0, 1}:
        raise ValueError("The target must contain binary classes 0 and 1.")
    if sample is not None and not 0 < sample <= 1:
        raise ValueError("sample must lie in (0, 1].")
    if deduplicate:
        frame, audit = deduplicate_predictors(frame, target)
    else:
        audit = {
            "policy": "no_row_removal_historical_baf_split",
            "raw_rows": len(frame), "retained_rows": len(frame),
            "raw_positive_rows": int(frame[target].sum()),
            "retained_positive_rows": int(frame[target].sum()),
            "removed_rows": 0, "removed_positive_rows": 0,
        }
    if sample is not None and sample < 1:
        minimum_size = int(np.ceil(10 / frame[target].mean()))
        n_sample = min(len(frame), max(100, int(len(frame) * sample), minimum_size))
        if n_sample < len(frame):
            frame, _ = train_test_split(
                frame, train_size=n_sample, stratify=frame[target],
                random_state=SPLIT_SEED,
            )
    X, y = frame.drop(columns=target), frame[target]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED,
    )
    dev_indices = X_dev.index.to_numpy(dtype=np.int64)
    test_indices = X_test.index.to_numpy(dtype=np.int64)
    if np.intersect1d(dev_indices, test_indices).size:
        raise AssertionError("DEV and TEST have overlapping original row positions.")
    metadata = {
        "protocol_version": PROTOCOL_VERSION, "split_seed": SPLIT_SEED,
        "split_ratio": "80/20 stratified", "sample_fraction": sample,
        "deduplication": audit, "eligible_rows": len(frame),
        "eligible_positive_rows": int(y.sum()), "feature_columns": list(X.columns),
        "dev_indices": dev_indices, "test_indices": test_indices,
        "dev_indices_sha256": array_sha256(dev_indices),
        "test_indices_sha256": array_sha256(test_indices),
    }
    return X_dev, X_test, y_dev, y_test, metadata


def _load_csv(csv_path, target, *, drop_columns=(), deduplicate=False,
              sample=None, return_metadata=False):
    frame = pd.read_csv(csv_path).drop(columns=list(drop_columns))
    output = _split_frame(frame, target, deduplicate=deduplicate, sample=sample)
    output[-1].update({
        "raw_file": str(Path(csv_path).resolve()),
        "raw_file_sha256": file_sha256(csv_path),
        "excluded_columns": list(drop_columns),
        "missing_value_policy": (
            "numeric_sentinels_preserved" if target == "fraud_bool"
            else "no_missing_values_observed"
        ),
    })
    return output if return_metadata else output[:4]


def load_dataset(name, sample=None, return_metadata=False):
    """Return four split objects, optionally followed by provenance metadata."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}; available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]["loader"](
        sample=sample, return_metadata=return_metadata,
    )


def get_dataset_info(name):
    entry = DATASET_REGISTRY[name]
    return entry["csv_path"], entry["label"]


@_register("ulb", str(_PROJECT_ROOT / "datasets/creditcard_2013.csv"), "ulb_2013")
def load_ulb_data(sample=None, return_metadata=False):
    """Remove exact predictor repetitions before all ULB split boundaries."""
    return _load_csv(
        _PROJECT_ROOT / "datasets/creditcard_2013.csv", "Class",
        deduplicate=True, sample=sample, return_metadata=return_metadata,
    )


@_register("baf_base", str(_PROJECT_ROOT / "datasets/Base.csv"), "baf_base")
def load_baf_base_data(sample=None, return_metadata=False):
    """Keep the historical pooled-month split, without month as a predictor."""
    return _load_csv(
        _PROJECT_ROOT / "datasets/Base.csv", "fraud_bool", drop_columns=("month",),
        sample=sample, return_metadata=return_metadata,
    )


def _load_baf_variant(csv_name, drop_extra_cols=False, sample=None,
                      return_metadata=False):
    """Project each Variant onto the shared Base predictor schema."""
    excluded = ("month", "x1", "x2") if drop_extra_cols else ("month",)
    return _load_csv(
        _PROJECT_ROOT / "datasets" / csv_name, "fraud_bool", drop_columns=excluded,
        sample=sample, return_metadata=return_metadata,
    )


@_register("baf_var1", str(_PROJECT_ROOT / "datasets/Variant I.csv"), "baf_var1")
def load_baf_var1(sample=None, return_metadata=False):
    return _load_baf_variant("Variant I.csv", sample=sample, return_metadata=return_metadata)


@_register("baf_var2", str(_PROJECT_ROOT / "datasets/Variant II.csv"), "baf_var2")
def load_baf_var2(sample=None, return_metadata=False):
    return _load_baf_variant("Variant II.csv", sample=sample, return_metadata=return_metadata)


@_register("baf_var3", str(_PROJECT_ROOT / "datasets/Variant III.csv"), "baf_var3")
def load_baf_var3(sample=None, return_metadata=False):
    return _load_baf_variant("Variant III.csv", True, sample=sample, return_metadata=return_metadata)


@_register("baf_var4", str(_PROJECT_ROOT / "datasets/Variant IV.csv"), "baf_var4")
def load_baf_var4(sample=None, return_metadata=False):
    return _load_baf_variant("Variant IV.csv", sample=sample, return_metadata=return_metadata)


@_register("baf_var5", str(_PROJECT_ROOT / "datasets/Variant V.csv"), "baf_var5")
def load_baf_var5(sample=None, return_metadata=False):
    return _load_baf_variant("Variant V.csv", True, sample=sample, return_metadata=return_metadata)
