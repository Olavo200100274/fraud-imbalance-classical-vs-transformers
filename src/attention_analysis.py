"""
Attention Map Analysis for FT-Transformer
==========================================
Extracts and visualises attention weights from the trained FT-Transformer
to diagnose how the [CLS] token distributes attention across numerical
vs categorical features.

No retraining required — loads the saved checkpoint and runs inference.

Usage:
    python src/attention_analysis.py --dataset baf_base --run-dir <pinned-run> \
        --output-dir <new-revision-directory>

Attention distributions are descriptive diagnostics, not feature attributions
or evidence identifying why predictive performance changed.
"""

import argparse
import json
import hashlib
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline

from data import load_dataset, get_dataset_info, DATASET_REGISTRY
from models.fttransformer import FTTransformer, TabularDataset, build_model

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = _PROJECT_ROOT / "results"
SPLIT_SEED = 42
VAL_FRACTION = 0.2
N_CASES = 3  # number of FP/FN cases to visualise


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def find_run_dir(dataset_name, strategy="none"):
    """Resolve the sole historical run; require an explicit path if ambiguous."""
    base = RESULTS_ROOT / dataset_name / "fttransformer" / strategy
    if not base.exists():
        raise FileNotFoundError(f"No runs found at {base}")
    runs = sorted(path for path in base.iterdir() if path.is_dir() and path.name.startswith("run_"))
    if not runs:
        raise FileNotFoundError(f"No runs found at {base}")
    if len(runs) > 1:
        raise ValueError(f"Multiple runs at {base}; provide --run-dir")
    return runs[0]


def load_checkpoint(run_dir, device):
    """Load the FT-Transformer checkpoint and rebuild the model."""
    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)

    hp = ckpt["hyperparams"]
    model = build_model(
        hp,
        d_numerical=ckpt["d_numerical"],
        cat_cardinalities=ckpt["cat_cardinalities"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt


def preprocess_data(X_train_df, X_other_df, num_cols, cat_cols, fitted_preprocessors=None):
    """Reproduce the same preprocessing as main_transformer.py."""
    if fitted_preprocessors is not None:
        for name, expected in (("num_cols", num_cols), ("cat_cols", cat_cols)):
            if name in fitted_preprocessors and list(fitted_preprocessors[name]) != list(expected):
                raise ValueError(f"Saved preprocessing and checkpoint token order disagree: {name}")
    if fitted_preprocessors is None:
        num_pipe = SkPipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
        num_pipe.fit(X_train_df[num_cols])
    else:
        num_pipe = fitted_preprocessors["num_preprocessor"]
    X_num_train = np.asarray(num_pipe.transform(X_train_df[num_cols]), dtype=np.float32)
    X_num_other = np.asarray(
        num_pipe.transform(X_other_df[num_cols]), dtype=np.float32
    )

    X_cat_train = None
    X_cat_other = None
    cat_cardinalities = []

    if cat_cols:
        if fitted_preprocessors is None:
            cat_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64,
            )
            cat_encoder.fit(X_train_df[cat_cols])
        else:
            cat_encoder = fitted_preprocessors["cat_encoder"]
        X_cat_train = np.asarray(cat_encoder.transform(X_train_df[cat_cols]), dtype=np.int64)
        X_cat_other = np.asarray(
            cat_encoder.transform(X_other_df[cat_cols]), dtype=np.int64
        )
        cat_cardinalities = [len(c) for c in cat_encoder.categories_]

    return X_num_train, X_cat_train, X_num_other, X_cat_other, cat_cardinalities


def classify_predictions(y_true, y_scores, threshold):
    """Classify each sample into TP/FP/FN/TN."""
    y_pred = (y_scores >= threshold).astype(int)
    labels = np.full(len(y_true), "", dtype=object)
    labels[(y_true == 1) & (y_pred == 1)] = "TP"
    labels[(y_true == 0) & (y_pred == 1)] = "FP"
    labels[(y_true == 1) & (y_pred == 0)] = "FN"
    labels[(y_true == 0) & (y_pred == 0)] = "TN"
    return labels, y_pred


def select_hard_cases(labels, y_scores, threshold, n=N_CASES):
    """Select the most borderline FP and FN cases."""
    cases = {}
    distance_to_threshold = np.abs(y_scores - threshold)

    for label in ["FN", "FP"]:
        mask = labels == label
        if mask.sum() == 0:
            print(f"  Warning: no {label} cases found.")
            continue
        indices = np.where(mask)[0]
        # Sort by distance to threshold (most borderline first)
        sorted_idx = indices[np.argsort(distance_to_threshold[indices])]
        cases[label] = sorted_idx[:n]

    return cases


def summarise_cls_attention(weights, numerical_count, categorical_count):
    """Describe normalised final-layer mean-head weights without causal inference."""
    weights = np.asarray(weights)
    if weights.ndim != 2 or weights.shape[1] != 1 + numerical_count + categorical_count:
        raise ValueError("CLS weight dimensions do not match the declared token schema")
    if not len(weights) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("CLS weights must be non-negative, finite and non-empty")
    if not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError("Every extracted CLS distribution must sum to one")
    means = weights.mean(axis=0, dtype=np.float64)
    numerical_mean = float(means[1:1 + numerical_count].mean()) if numerical_count else None
    categorical_mean = float(means[1 + numerical_count:].mean()) if categorical_count else None
    entropy_weights = np.asarray(weights, dtype=np.float64)
    entropy = -(entropy_weights * np.log(np.maximum(entropy_weights, np.finfo(np.float64).tiny))).sum(axis=1)
    maximum_entropy = float(np.log(weights.shape[1]))
    return {"cls_self_attention": float(means[0]),
            "mean_numerical_attention": numerical_mean,
            "mean_categorical_attention": categorical_mean,
            "ratio_cat_over_num": categorical_mean / numerical_mean
            if categorical_mean is not None and numerical_mean is not None and numerical_mean > 0 else None,
            "attention_entropy_mean": float(entropy.mean()),
            "attention_entropy_std": float(entropy.std()),
            "max_possible_entropy": maximum_entropy,
            "normalized_entropy": float(entropy.mean() / maximum_entropy)}


# ─────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────

def extract_attention(model, x_num, x_cat, device):
    """
    Extract attention weights for given samples.

    Returns
    -------
    attn_weights : list of np.ndarray
        One (n_samples, n_tokens, n_tokens) array per layer.
    """
    x_num_t = torch.tensor(x_num, dtype=torch.float32).to(device)
    x_cat_t = (
        torch.tensor(x_cat, dtype=torch.long).to(device) if x_cat is not None else None
    )

    logits, attn_layers = model.forward_with_attention(x_num_t, x_cat_t)
    return [w.cpu().numpy() for w in attn_layers], torch.sigmoid(logits).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────

def _plot_horizontal_cls_weights(weights, deviations, feature_names, num_cols, cat_cols,
                                 title, save_path, *, sort=False):
    """Draw all original token labels legibly at dissertation text width."""
    feat_weights = np.asarray(weights)[1:]
    feat_names = list(feature_names[1:])
    n_num, n_cat = len(num_cols), len(cat_cols)
    colours = np.array(["#4472C4"] * n_num + ["#ED7D31"] * n_cat)
    order = np.argsort(-feat_weights, kind="stable") if sort else np.arange(len(feat_names))
    errors = np.asarray(deviations)[1:][order] if deviations is not None else None
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    positions = np.arange(len(feat_names))
    ax.barh(positions, feat_weights[order], xerr=errors, color=colours[order],
            height=.72, edgecolor="white", linewidth=.3,
            capsize=1.5, error_kw={"linewidth": .55, "alpha": .7})
    ax.set_yticks(positions, [feat_names[index].replace("_", " ") for index in order],
                  fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel("Mean attention weight ± across-row SD" if errors is not None else "Attention weight",
                  fontsize=8.5)
    ax.set_title(title, fontsize=9, pad=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    mean_num = float(feat_weights[:n_num].mean())
    ax.axvline(mean_num, color="#4472C4", linestyle="--", linewidth=.9, alpha=.75)
    mean_cat = float(feat_weights[n_num:].mean()) if n_cat else None
    if mean_cat is not None:
        ax.axvline(mean_cat, color="#ED7D31", linestyle="--", linewidth=.9, alpha=.75)
    uniform = 1.0 / len(feature_names)
    ax.axvline(uniform, color="#444444", linestyle=":", linewidth=1.0)
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color="#4472C4", linestyle="--",
                     label=f"Numerical mean: {mean_num:.4f}")]
    if mean_cat is not None:
        legend.append(Line2D([0], [0], color="#ED7D31", linestyle="--",
                             label=f"Categorical mean: {mean_cat:.4f}"))
    legend.append(Line2D([0], [0], color="#444444", linestyle=":",
                         label=f"Uniform 1/{len(feature_names)}: {uniform:.4f}"))
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.66, .025),
               ncol=len(legend), fontsize=7.5, frameon=False, handlelength=1.5,
               columnspacing=.9, handletextpad=.35)
    fig.subplots_adjust(left=.405, right=.985, top=.905, bottom=.17)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cls_attention_bar(
    cls_weights, feature_names, num_cols, cat_cols, title, save_path,
    score=None, true_label=None,
):
    """Display one final-layer, head-averaged CLS distribution, including its reference."""
    subtitle = f"CLS self-attention: {cls_weights[0]:.4f}"
    if score is not None:
        subtitle += f" | P(fraud): {score:.6f}"
    if true_label is not None:
        subtitle += f" | {'Fraud' if true_label == 1 else 'Legitimate'}"
    _plot_horizontal_cls_weights(cls_weights, None, feature_names, num_cols, cat_cols,
                                 title + "\n" + subtitle, save_path)


def plot_aggregate_attention(
    all_cls_weights, feature_names, num_cols, cat_cols, save_path,
):
    """Display descriptive mean and across-row SD, not uncertainty intervals."""
    mean_weights = all_cls_weights.mean(axis=0, dtype=np.float64)
    std_weights = all_cls_weights.std(axis=0, dtype=np.float64)
    title = (f"Final-layer, head-averaged CLS attention (N={len(all_cls_weights):,})\n"
             f"Mean CLS self-attention: {mean_weights[0]:.4f}")
    _plot_horizontal_cls_weights(mean_weights, std_weights, feature_names, num_cols, cat_cols,
                                 title, save_path, sort=True)


def plot_attention_heatmap(attn_weights, feature_names, title, save_path):
    """Show the selected full token matrix with readable row labels and matching key indices."""
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    im = ax.imshow(attn_weights, cmap="Blues", aspect="auto")
    indices = np.arange(len(feature_names))
    ax.set_xticks(indices, [str(index + 1) for index in indices], rotation=90, fontsize=7.5)
    ax.set_yticks(indices, [f"{index + 1}  {name.replace('_', ' ')}"
                           for index, name in enumerate(feature_names)], fontsize=8.0)
    ax.set_xlabel("Key token (index matches the corresponding row label)", fontsize=8.5)
    ax.set_ylabel("Query token", fontsize=8.5)
    ax.set_title(title, fontsize=9, pad=9)
    fig.colorbar(im, ax=ax, fraction=.04, pad=.02).ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.37, right=.965, top=.925, bottom=.12)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Attention Map Analysis for FT-Transformer"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_REGISTRY.keys()),
    )
    parser.add_argument(
        "--batch_size", type=int, default=2048,
        help="Batch size for aggregate extraction (default: 2048)",
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Explicit saved FT-Transformer run")
    parser.add_argument("--output-dir", type=Path, required=True, help="New revision output directory")
    parser.add_argument("--allow-legacy-preprocessing", action="store_true",
                        help="Reconstruct preprocessing only for historical runs without saved preprocessors")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--threads", type=int, default=2, help="Bound CPU inference threads")
    args = parser.parse_args()
    from generate_revision_interpretability import validate_output_directory
    args.output_dir = validate_output_directory(args.output_dir)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    from revision_resources import baf_training_resource
    with baf_training_resource(args.dataset, args.output_dir,
                               f"Attention extraction: {args.run_dir.resolve()}"):
        run_analysis(args)


def run_analysis(args):
    """Extract the explicitly selected run; the CLI coordinates shared BAF RAM."""
    torch.set_num_threads(args.threads)
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    print(f"Device: {device}")

    # ── Load run artifacts ────────────────────────────────────────────
    _, dataset_label = get_dataset_info(args.dataset)
    run_dir = args.run_dir.resolve()
    print(f"Run dir: {run_dir}")
    protected_sources = ("config.json", "model.pt", "preprocessors.joblib", "y_test.npy",
                         "y_test_scores.npy", "test_row_indices.npy")
    initial_source_hashes = {
        name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        for name in protected_sources if (run_dir / name).is_file()
    }
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("dataset") != dataset_label or config.get("model") != "fttransformer":
        raise ValueError("The selected run has the wrong dataset or model")

    model, ckpt = load_checkpoint(run_dir, device)
    num_cols = ckpt["num_cols"]
    cat_cols = ckpt["cat_cols"]
    threshold = ckpt["threshold"]
    if "threshold_exact" in config and float(threshold) != float(config["threshold_exact"]):
        raise ValueError("Checkpoint and configuration disagree about the selected exact threshold")
    feature_names = ["[CLS]"] + num_cols + cat_cols

    print(f"Model: d_token={ckpt['hyperparams']['d_token']}, "
          f"n_blocks={ckpt['hyperparams']['n_blocks']}, "
          f"n_heads={ckpt['hyperparams']['attention_n_heads']}")
    print(f"Features: {len(num_cols)} numerical + {len(cat_cols)} categorical")
    print(f"Threshold: {threshold:.6f}")

    # ── Load dataset and preprocess ───────────────────────────────────
    print(f"\nLoading {args.dataset} dataset ...")
    X_train, X_test, y_train, y_test, metadata = load_dataset(args.dataset, return_metadata=True)
    indices_path = run_dir / "test_row_indices.npy"
    if indices_path.exists() and not np.array_equal(np.load(indices_path), metadata["test_indices"]):
        raise ValueError("Dataset TEST membership differs from the selected saved run")

    preprocessors_path = run_dir / "preprocessors.joblib"
    fitted_preprocessors = joblib.load(preprocessors_path) if preprocessors_path.exists() else None
    if args.run_dir and fitted_preprocessors is None and not args.allow_legacy_preprocessing:
        raise FileNotFoundError("Explicit revision run requires preprocessors.joblib; "
                                "use --allow-legacy-preprocessing only for preserved historical runs")
    X_num_train, X_cat_train, X_num_test, X_cat_test, _ = preprocess_data(
        X_train, X_test, num_cols, cat_cols, fitted_preprocessors,
    )
    print(f"Test set: {len(X_num_test):,} samples")

    # ── Load saved scores for case selection ──────────────────────────
    y_test_arr = np.load(run_dir / "y_test.npy")
    y_scores_arr = np.load(run_dir / "y_test_scores.npy")
    if not np.array_equal(y_test_arr, np.asarray(y_test)) or len(y_scores_arr) != len(X_test):
        raise ValueError("Dataset TEST ordering does not match the selected saved run")

    labels, y_pred = classify_predictions(y_test_arr, y_scores_arr, threshold)
    print(f"\nConfusion matrix at threshold={threshold:.4f}:")
    for lbl in ["TP", "FP", "FN", "TN"]:
        print(f"  {lbl}: {(labels == lbl).sum():,}")

    # ── Output directory ──────────────────────────────────────────────
    from generate_revision_interpretability import validate_output_directory
    out_dir = validate_output_directory(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput dir: {out_dir}")

    # ── Select hard cases ─────────────────────────────────────────────
    hard_cases = select_hard_cases(labels, y_scores_arr, threshold)

    # ── Individual case visualisations ────────────────────────────────
    print("\n--- Individual Case Analysis ---")
    case_details = {}

    for case_type, indices in hard_cases.items():
        for rank, idx in enumerate(indices):
            x_num_i = X_num_test[idx : idx + 1]
            x_cat_i = X_cat_test[idx : idx + 1] if X_cat_test is not None else None

            attn_layers, scores = extract_attention(model, x_num_i, x_cat_i, device)
            if abs(float(scores[0]) - float(y_scores_arr[idx])) > 1e-5:
                raise ValueError(f"Attention forward differs from saved score for TEST position {idx}")
            if bool(scores[0] >= threshold) != bool(y_scores_arr[idx] >= threshold):
                raise ValueError(f"Attention numerical drift changes the selected {case_type} label at {idx}")

            # Use the last layer's attention
            last_layer_attn = attn_layers[-1][0]  # (n_tokens, n_tokens)
            if not np.isfinite(last_layer_attn).all() or not np.allclose(last_layer_attn.sum(axis=1), 1, rtol=0, atol=1e-5):
                raise ValueError("Selected case attention contains non-finite or non-normalised query rows")
            cls_weights = last_layer_attn[0]  # CLS row → weights over all tokens

            # Sanity check
            assert abs(cls_weights.sum() - 1.0) < 1e-4, \
                f"CLS weights don't sum to 1: {cls_weights.sum()}"

            title = f"{case_type} Case #{rank + 1} (sample idx={idx})"
            save_name = f"attention_bar_{case_type}_{rank + 1}.png"

            plot_cls_attention_bar(
                cls_weights, feature_names, num_cols, cat_cols,
                title=title,
                save_path=out_dir / save_name,
                score=y_scores_arr[idx],
                true_label=y_test_arr[idx],
            )

            # Heatmap for one case per type
            if rank == 0:
                plot_attention_heatmap(
                    last_layer_attn, feature_names,
                    title=f"Full Attention Matrix — {case_type} Case #{rank + 1}",
                    save_path=out_dir / f"attention_heatmap_{case_type}.png",
                )

            case_details[f"{case_type}_{rank + 1}"] = {
                "sample_idx": int(idx),
                "score": float(y_scores_arr[idx]),
                "distance_to_threshold": abs(float(y_scores_arr[idx]) - float(threshold)),
                "attention_forward_score": float(scores[0]),
                "true_label": int(y_test_arr[idx]),
                "cls_self_attention": float(cls_weights[0]),
                "mean_numerical_attention": float(cls_weights[1 : 1 + len(num_cols)].mean()),
                "mean_categorical_attention": float(cls_weights[1 + len(num_cols) :].mean()) if cat_cols else None,
            }
            matrix_name = f"attention_matrix_{case_type}_{rank + 1}.npy"
            np.save(out_dir / matrix_name, last_layer_attn.astype(np.float32), allow_pickle=False)
            case_details[f"{case_type}_{rank + 1}"]["matrix_file"] = matrix_name

    # ── Aggregate analysis over full test set ─────────────────────────
    print("\n--- Aggregate Analysis (full test set) ---")
    all_cls_weights = []
    max_score_error = 0.0
    threshold_disagreements = 0

    n_test = len(X_num_test)
    bs = args.batch_size
    for start in range(0, n_test, bs):
        end = min(start + bs, n_test)
        x_num_batch = X_num_test[start:end]
        x_cat_batch = X_cat_test[start:end] if X_cat_test is not None else None

        attn_layers, attention_scores = extract_attention(model, x_num_batch, x_cat_batch, device)
        score_error = float(np.max(np.abs(attention_scores - y_scores_arr[start:end])))
        max_score_error = max(max_score_error, score_error)
        threshold_disagreements += int(np.count_nonzero(
            (attention_scores >= threshold) != (y_scores_arr[start:end] >= threshold)))
        # Last layer, CLS row
        cls_w = attn_layers[-1][:, 0, :]  # (batch, n_tokens)
        all_cls_weights.append(cls_w)

        if (start // bs) % 10 == 0:
            print(f"  Processed {end:,}/{n_test:,} samples ...")

    all_cls_weights = np.concatenate(all_cls_weights, axis=0)  # (N, n_tokens)
    if max_score_error > 1e-5 or threshold_disagreements:
        raise ValueError(f"Attention forward and saved TEST predictions differ: max error={max_score_error:g}, "
                         f"threshold decisions changed={threshold_disagreements}")
    if not np.allclose(all_cls_weights.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Extracted CLS attention does not sum to one")
    print(f"  Total: {all_cls_weights.shape[0]:,} samples")

    plot_aggregate_attention(
        all_cls_weights, feature_names, num_cols, cat_cols,
        save_path=out_dir / "attention_aggregate_cls.png",
    )

    # ── Compute summary statistics ────────────────────────────────────
    mean_w = all_cls_weights.mean(axis=0, dtype=np.float64)
    statistics = summarise_cls_attention(all_cls_weights, len(num_cols), len(cat_cols))
    cls_self = statistics["cls_self_attention"]
    num_mean = statistics["mean_numerical_attention"]
    cat_mean = statistics["mean_categorical_attention"]

    # Per-class analysis
    fraud_mask = y_test_arr == 1
    legit_mask = y_test_arr == 0

    summary = {
        "dataset": args.dataset,
        "strategy": config.get("strategy"),
        "n_test_samples": int(n_test),
        "n_numerical_features": len(num_cols),
        "n_categorical_features": len(cat_cols),
        "feature_token_names": feature_names,
        "attention_storage_dtype": "float32",
        "summary_accumulation_dtype": "float64",
        "threshold": float(threshold),
        "source_run": str(run_dir.resolve()),
        "preprocessing_source": str(preprocessors_path) if fitted_preprocessors is not None else "reconstructed from historical DEV",
        "aggregation_scope": "Final layer; attention averaged across heads; all tokens including CLS retained in entropy",
        "extraction_forward_policy": "Inspect pre-normalised attention weights independently; propagate activations through the unchanged native encoder layers",
        "interpretation_boundary": "Descriptive allocation, not causal feature importance or an explanation of performance differences",
        "case_selection_rule": "Three FP and three FN TEST cases nearest to the frozen baseline threshold",
        "score_verification": {"maximum_absolute_error": max_score_error,
                               "threshold_disagreements": threshold_disagreements,
                               "absolute_tolerance": 1e-5, "compared_test_rows": int(n_test)},
        "attention_summary": statistics,
        "per_class": {
            "fraud_cls_self": float(all_cls_weights[fraud_mask, 0].mean(dtype=np.float64)) if fraud_mask.any() else None,
            "fraud_num_mean": float(all_cls_weights[fraud_mask, 1:1+len(num_cols)].mean(dtype=np.float64)) if fraud_mask.any() else None,
            "fraud_cat_mean": float(all_cls_weights[fraud_mask, 1+len(num_cols):].mean(dtype=np.float64)) if fraud_mask.any() and cat_cols else None,
            "legit_cls_self": float(all_cls_weights[legit_mask, 0].mean(dtype=np.float64)) if legit_mask.any() else None,
            "legit_num_mean": float(all_cls_weights[legit_mask, 1:1+len(num_cols)].mean(dtype=np.float64)) if legit_mask.any() else None,
            "legit_cat_mean": float(all_cls_weights[legit_mask, 1+len(num_cols):].mean(dtype=np.float64)) if legit_mask.any() and cat_cols else None,
        },
        "per_feature_mean_attention": {
            name: float(mean_w[i]) for i, name in enumerate(feature_names)
        },
        "individual_cases": case_details,
    }

    # ── Bounded descriptive observations ──────────────────────────────
    ratio = statistics["ratio_cat_over_num"]
    norm_entropy = statistics["normalized_entropy"]

    diagnosis = []
    if cls_self > 0.3:
        diagnosis.append(f"The mean CLS self-attention weight is {cls_self:.1%}; "
                         "this does not by itself identify a predictive limitation.")
    if ratio is not None and ratio > 2.0:
        diagnosis.append(f"Mean per-token categorical attention is {ratio:.1f} times "
                         "mean numerical attention; these weights are not causal feature importance.")
    if norm_entropy > 0.95:
        diagnosis.append(f"The final-layer mean-head CLS distribution has high normalised entropy ({norm_entropy:.3f}); "
                         "near-uniform weights do not establish the cause of model performance.")
    # Check tunnel vision: any single feature > 30% of total non-CLS attention
    feat_w = mean_w[1:]
    max_feat_pct = feat_w.max() / feat_w.sum()
    if max_feat_pct > 0.30:
        top_feat = feature_names[1 + feat_w.argmax()]
        diagnosis.append(f"Token '{top_feat}' receives {max_feat_pct:.1%} of mean non-CLS attention; "
                         "attention allocation is distinct from contribution to the prediction.")
    if not diagnosis:
        diagnosis.append("No declared descriptive concentration or high-entropy flag was triggered; "
                         "this is not a validation of model performance or explanation faithfulness.")

    summary["diagnosis"] = diagnosis
    final_source_hashes = {
        name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        for name in protected_sources
        if (run_dir / name).is_file()
    }
    if final_source_hashes != initial_source_hashes:
        raise ValueError("A protected source artefact changed during attention extraction")
    summary["source_sha256"] = initial_source_hashes
    summary["extractor_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["model_implementation_sha256"] = hashlib.sha256(
        (_PROJECT_ROOT / "src/models/fttransformer.py").read_bytes()).hexdigest()

    # Save summary
    np.save(out_dir / "attention_cls_weights.npy", all_cls_weights.astype(np.float32), allow_pickle=False)
    summary["derived_artefacts_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out_dir.iterdir()) if path.suffix in (".png", ".npy")
    }
    summary_path = out_dir / "attention_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"\n  Saved: {summary_path}")

    # ── Print diagnosis ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  ATTENTION ANALYSIS DIAGNOSIS")
    print(f"{'=' * 60}")
    print(f"  CLS → CLS (self):      {cls_self:.4f} ({cls_self:.1%})")
    print(f"  Mean numerical attn:    {num_mean:.4f}")
    if cat_mean is not None:
        print(f"  Mean categorical attn:  {cat_mean:.4f}")
    if ratio is not None:
        print(f"  Ratio cat/num:          {ratio:.2f}x")
    print(f"  Normalised entropy:     {norm_entropy:.3f} (1.0 = perfectly uniform)")
    print()
    for d in diagnosis:
        print(f"  >> {d}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
