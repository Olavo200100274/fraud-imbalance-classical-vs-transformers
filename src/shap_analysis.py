"""
SHAP Interpretability Analysis — Fraud Detection Pipeline
==========================================================
Historical explainer implementations are retained for traceability. The current
CLI reads explicitly pinned compatible saved matrices and writes revised figures
and numerical QA only in an isolated output directory; archives are immutable.

Analyses:
  1. Global feature importance (mean |SHAP|, top-K)
  2. Cross-model consistency (Jaccard similarity of top-K features)
  3. Cross-variant stability (LGBM SHAP on BAF Variants I–V)
  4. Local explanations (selected high-score TP/FP and lowest-score FN cases)

Models:
  - LogReg     → shap.LinearExplainer (fast, exact)
  - LGBM       → shap.TreeExplainer  (fast, exact)
  - CatBoost   → shap.TreeExplainer  (fast, exact)
FT-Transformer GradientExplainer artefacts cannot reliably attribute categorical
embedding indices and are excluded from the current primary comparison.

Usage:
    python src/shap_analysis.py --manifest <revision-manifest.json> \
        --output-dir <isolated-revision-interpretability-directory>

Use revision_variant_shap.py separately for corrected, explicitly prepared
Base-DEV-profile-disjoint Variant populations. Neither CLI re-fits a model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from data import load_dataset

# ── Constants ────────────────────────────────────────────────────────────
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
DATASET_KEY = "baf_base"
DATASET_LABEL = "baf_base"
TOP_K = 15
FT_BACKGROUND_N = 500       # background samples for GradientExplainer
FT_EXPLAIN_N = 2000         # test samples to explain for FT-Transformer
SHAP_MODELS = ["logreg", "lgbm", "catboost"]

TARGET_VARIANTS = {
    "baf_var1": "Variant I",
    "baf_var2": "Variant II",
    "baf_var3": "Variant III",
    "baf_var4": "Variant IV",
    "baf_var5": "Variant V",
}


# ── Helpers ──────────────────────────────────────────────────────────────

def find_latest_run(dataset_label, model_name, strategy="none"):
    base = RESULTS_ROOT / dataset_label / model_name / strategy
    if not base.exists():
        return None
    runs = sorted(path for path in base.iterdir() if path.is_dir() and path.name.startswith("run_"))
    if len(runs) > 1:
        raise ValueError(f"Multiple runs at {base}; select the source explicitly")
    return runs[0] if runs else None


def save_json(obj, path):
    """Save dict as JSON, converting numpy types."""
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert(v) for v in o]
        return o
    with open(path, "w") as f:
        json.dump(convert(obj), f, indent=2)


def aggregate_onehot_shap(shap_values, feature_names, original_cat_features):
    """
    Aggregate one-hot SHAP values back to original categorical features.

    E.g., payment_type_AA, payment_type_AB, ... → payment_type (sum of |SHAP|).
    Returns (aggregated_shap_values, aggregated_feature_names).
    """
    # Map each one-hot column to its parent feature
    col_to_parent = {}
    for i, fname in enumerate(feature_names):
        matched = False
        for cat_feat in original_cat_features:
            if fname.startswith(cat_feat + "_"):
                col_to_parent[i] = cat_feat
                matched = True
                break
        if not matched:
            col_to_parent[i] = fname  # numeric feature, keep as-is

    # Get unique parent features in order
    seen = set()
    parent_order = []
    for i in range(len(feature_names)):
        p = col_to_parent[i]
        if p not in seen:
            seen.add(p)
            parent_order.append(p)

    # Aggregate: for each parent, sum absolute SHAP values of its children
    agg_shap = np.zeros((shap_values.shape[0], len(parent_order)))
    for i, fname in enumerate(feature_names):
        parent = col_to_parent[i]
        parent_idx = parent_order.index(parent)
        agg_shap[:, parent_idx] += np.abs(shap_values[:, i])

    return agg_shap, parent_order


def extract_global_importance(shap_abs, feature_names, top_k=TOP_K):
    """Return sorted dict of {feature: mean_abs_shap} for top-K features."""
    mean_abs = np.mean(shap_abs, axis=0)
    indices = np.argsort(mean_abs)[::-1][:top_k]
    return {feature_names[i]: float(mean_abs[i]) for i in indices}


# ── Categorical feature names (BAF) ─────────────────────────────────────
BAF_CAT_FEATURES = [
    "payment_type", "employment_status", "housing_status",
    "source", "device_os",
]


# ── SHAP for sklearn pipeline models ────────────────────────────────────

def compute_shap_sklearn(model_name, pipeline, X_test):
    """
    Compute SHAP values for a sklearn Pipeline model.

    Returns (shap_values, feature_names_post_transform).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform test data through preprocessor
    X_transformed = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    # Convert to numpy if DataFrame (due to set_config(transform_output="pandas"))
    if hasattr(X_transformed, "values"):
        X_transformed = X_transformed.values

    if model_name == "logreg":
        explainer = shap.LinearExplainer(classifier, X_transformed)
    else:  # lgbm, catboost
        explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X_transformed)

    # TreeExplainer may return list for binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    return shap_values, feature_names


# ── SHAP for FT-Transformer ─────────────────────────────────────────────

def compute_shap_fttransformer(run_dir, X_test, background_n=FT_BACKGROUND_N,
                                explain_n=FT_EXPLAIN_N):
    """
    Compute SHAP values for the FT-Transformer using GradientExplainer.

    Uses a thin wrapper to present the model as a single-input function.
    Returns (shap_values, feature_names).
    """
    import torch
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipeline
    from models.fttransformer import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu",
                            weights_only=False)
    hp = checkpoint["hyperparams"]
    d_num = checkpoint["d_numerical"]
    cat_cards = checkpoint["cat_cardinalities"]
    num_cols = checkpoint["num_cols"]
    cat_cols = checkpoint["cat_cols"]

    # Fit preprocessors on BAF Base train
    X_base_train, _, _, _ = load_dataset("baf_base")

    num_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    num_pipe.fit(X_base_train[num_cols])
    X_num_test = np.asarray(
        num_pipe.transform(X_test[num_cols]), dtype=np.float32
    )

    X_cat_test = None
    cat_enc = None
    if cat_cols:
        cat_enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
        )
        cat_enc.fit(X_base_train[cat_cols])
        X_cat_test = np.asarray(cat_enc.transform(X_test[cat_cols]), dtype=np.int64)

    # Build and load model
    model = build_model(hp, d_num, cat_cards).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Feature names: numeric cols + categorical cols (ordinal, not one-hot)
    feature_names = list(num_cols) + list(cat_cols)

    # Concatenate num + cat into a single array for SHAP
    if X_cat_test is not None:
        X_combined = np.hstack([X_num_test, X_cat_test.astype(np.float32)])
    else:
        X_combined = X_num_test

    n_num = len(num_cols)
    n_cat = len(cat_cols)

    # Thin wrapper: single input tensor → sigmoid probability
    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model, n_num, n_cat):
            super().__init__()
            self.base_model = base_model
            self.n_num = n_num
            self.n_cat = n_cat

        def forward(self, x):
            x_num = x[:, :self.n_num]
            x_cat = None
            if self.n_cat > 0:
                x_cat = x[:, self.n_num:].long()
            logits = self.base_model(x_num, x_cat)
            return torch.sigmoid(logits).unsqueeze(-1)

    wrapper = ModelWrapper(model, n_num, n_cat).to(device)
    wrapper.eval()

    # Uniform random background and explanation samples (not stratified).
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_combined), size=min(background_n, len(X_combined)),
                        replace=False)
    explain_idx = rng.choice(len(X_combined), size=min(explain_n, len(X_combined)),
                             replace=False)

    bg_tensor = torch.tensor(X_combined[bg_idx], dtype=torch.float32).to(device)
    explain_tensor = torch.tensor(X_combined[explain_idx],
                                  dtype=torch.float32).to(device)

    explainer = shap.GradientExplainer(wrapper, bg_tensor)
    shap_values = explainer.shap_values(explain_tensor)

    # GradientExplainer may return list
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Convert to numpy
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().numpy()

    # GradientExplainer may return shape (n, features, 1) — squeeze last dim
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze(-1)

    # For FT-Transformer, no one-hot aggregation needed (ordinal encoding)
    return shap_values, feature_names, explain_idx


# ── Local explanations ───────────────────────────────────────────────────

def find_representative_cases(y_test, y_scores, threshold, shap_values,
                               feature_names, sample_indices=None):
    """
    Find representative TP, FP, FN cases for waterfall plots.

    Returns dict with SHAP values and metadata for each case type.
    """
    y_pred = (y_scores >= threshold).astype(int)
    y_true = y_test.values if hasattr(y_test, "values") else y_test

    # If we only explained a subset, map back
    if sample_indices is not None:
        y_true_sub = y_true[sample_indices]
        y_pred_sub = y_pred[sample_indices]
        y_scores_sub = y_scores[sample_indices]
    else:
        y_true_sub = y_true
        y_pred_sub = y_pred
        y_scores_sub = y_scores

    tp_mask = (y_true_sub == 1) & (y_pred_sub == 1)
    fp_mask = (y_true_sub == 0) & (y_pred_sub == 1)
    fn_mask = (y_true_sub == 1) & (y_pred_sub == 0)

    cases = {}
    for label, mask in [("TP", tp_mask), ("FP", fp_mask), ("FN", fn_mask)]:
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        if label == "FN":
            # Highest-confidence missed fraud (lowest score among FN)
            best = idxs[np.argmin(y_scores_sub[idxs])]
        else:
            # Highest-confidence case
            best = idxs[np.argmax(y_scores_sub[idxs])]

        cases[label] = {
            "score": float(y_scores_sub[best]),
            "true_label": int(y_true_sub[best]),
            "pred_label": int(y_pred_sub[best]),
            "shap_values": {feature_names[i]: float(shap_values[best, i])
                            for i in range(len(feature_names))},
        }

    return cases


# ── Cross-variant stability ─────────────────────────────────────────────

def compute_variant_shap(lgbm_pipeline, top_k=10, variant_test_frames=None):
    """
    Compute SHAP feature rankings for LGBM on each BAF Variant test set.

    Uses the Base-trained model (no retraining) — same cross-domain setup.
    Returns dict[variant_key] = list of top-K feature names.
    """
    preprocessor = lgbm_pipeline.named_steps["preprocessor"]
    classifier = lgbm_pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)

    variant_rankings = {}

    # Base reference
    if variant_test_frames is None:
        _, X_test_base, _, _ = load_dataset("baf_base")
    else:
        expected = {"baf_base", *TARGET_VARIANTS}
        if set(variant_test_frames) != expected:
            raise ValueError("Explicit SHAP frames must include Base and all five BAF Variants")
        X_test_base = variant_test_frames["baf_base"]
    X_base_transformed = preprocessor.transform(X_test_base)
    if hasattr(X_base_transformed, "values"):
        X_base_transformed = X_base_transformed.values
    feature_names = list(preprocessor.get_feature_names_out())

    sv_base = explainer.shap_values(X_base_transformed)
    if isinstance(sv_base, list):
        sv_base = sv_base[1]

    sv_agg, agg_names = aggregate_onehot_shap(sv_base, feature_names,
                                               BAF_CAT_FEATURES)
    mean_abs = np.mean(sv_agg, axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    variant_rankings["baf_base"] = [agg_names[i] for i in top_idx]

    # Variants
    for var_key, var_label in TARGET_VARIANTS.items():
        print(f"    {var_label} ...")
        if variant_test_frames is None:
            _, X_test_var, _, _ = load_dataset(var_key)
        else:
            X_test_var = variant_test_frames[var_key]
        X_var_transformed = preprocessor.transform(X_test_var)
        if hasattr(X_var_transformed, "values"):
            X_var_transformed = X_var_transformed.values

        sv_var = explainer.shap_values(X_var_transformed)
        if isinstance(sv_var, list):
            sv_var = sv_var[1]

        sv_agg_var, _ = aggregate_onehot_shap(sv_var, feature_names,
                                               BAF_CAT_FEATURES)
        mean_abs_var = np.mean(sv_agg_var, axis=0)
        top_idx_var = np.argsort(mean_abs_var)[::-1][:top_k]
        variant_rankings[var_key] = [agg_names[i] for i in top_idx_var]

    return variant_rankings


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ── Main ─────────────────────────────────────────────────────────────────

def _historical_main():
    """Retain the old workflow as evidence, not as an archive-writing entry point."""
    raise RuntimeError("The historical in-place SHAP workflow is disabled; use the explicit revision CLI")
    parser = argparse.ArgumentParser(description="SHAP Interpretability Analysis")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to analyse (default: all)")
    parser.add_argument("--skip-transformer", action="store_true",
                        help="Skip FT-Transformer (slow)")
    parser.add_argument("--skip-variants", action="store_true",
                        help="Skip cross-variant stability analysis")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Number of top features (default: {TOP_K})")
    parser.add_argument("--allow-exploratory-ft", action="store_true",
                        help="Explicitly allow gradient artefacts that cannot explain categorical indices")
    args = parser.parse_args()

    models = args.models or [m for m in SHAP_MODELS
                             if not (args.skip_transformer and m == "fttransformer")]
    if "fttransformer" in models and not args.allow_exploratory_ft:
        parser.error("FT GradientExplainer cannot attribute categorical indices; "
                     "it is excluded by default and requires --allow-exploratory-ft")
    top_k = args.top_k

    print("=" * 60)
    print("  SHAP Interpretability Analysis (BAF Base)")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    print("\nLoading baf_base dataset ...")
    X_train, X_test, y_train, y_test = load_dataset("baf_base")
    print(f"  Train: {len(X_train):,} samples ({y_train.sum():,} fraud)")
    print(f"  Test : {len(X_test):,} samples ({y_test.sum():,} fraud)")

    # ── Compute SHAP per model ───────────────────────────────────
    all_global = {}        # model → {feature: importance}
    all_rankings = {}      # model → [top-K features]
    all_local = {}         # model → {TP/FP/FN → case data}
    all_raw_shap = {}      # model → (shap_values, feature_names)

    for model_name in models:
        run_dir = find_latest_run(DATASET_LABEL, model_name)
        if run_dir is None:
            print(f"\n  [SKIP] {model_name} — no baseline run found")
            continue

        print(f"\n{'─' * 60}")
        print(f"  {model_name.upper()} — SHAP Analysis")
        print(f"{'─' * 60}")
        print(f"  Run dir: {run_dir}")

        t0 = time.time()

        if model_name == "fttransformer":
            shap_values, feature_names, explain_idx = compute_shap_fttransformer(
                run_dir, X_test,
                background_n=FT_BACKGROUND_N, explain_n=FT_EXPLAIN_N,
            )
            # For FT-Transformer, no one-hot aggregation needed
            shap_abs = np.abs(shap_values)
            agg_names = feature_names

            # Load threshold for local explanations
            import torch
            checkpoint = torch.load(run_dir / "model.pt", map_location="cpu",
                                    weights_only=False)
            metrics = json.load(open(run_dir / "metrics_test.json"))
            threshold = metrics["threshold"]
            y_scores = np.load(run_dir / "y_test_scores.npy")

            # Local cases (on explained subset)
            local_cases = find_representative_cases(
                y_test, y_scores, threshold, shap_values,
                feature_names, sample_indices=explain_idx
            )

        else:
            # sklearn pipeline model
            pipeline = joblib.load(run_dir / "model.joblib")
            shap_values, feature_names = compute_shap_sklearn(
                model_name, pipeline, X_test
            )

            # Aggregate one-hot SHAP to original features
            shap_abs, agg_names = aggregate_onehot_shap(
                shap_values, feature_names, BAF_CAT_FEATURES
            )

            # Load threshold and scores for local explanations
            metrics = json.load(open(run_dir / "metrics_test.json"))
            threshold = metrics["threshold"]
            y_scores = np.load(run_dir / "y_test_scores.npy")

            local_cases = find_representative_cases(
                y_test, y_scores, threshold, shap_values,
                feature_names, sample_indices=None
            )

        elapsed = time.time() - t0
        print(f"  SHAP computed in {elapsed:.1f}s")

        # Global importance
        global_imp = extract_global_importance(shap_abs, agg_names, top_k)
        all_global[model_name] = global_imp
        all_rankings[model_name] = list(global_imp.keys())[:10]  # top-10 for Jaccard
        all_local[model_name] = local_cases
        all_raw_shap[model_name] = (shap_values, feature_names, agg_names)

        print(f"  Top-5 features: {list(global_imp.keys())[:5]}")

        # Save per-model artefacts
        save_json(global_imp, run_dir / "shap_global.json")
        save_json(local_cases, run_dir / "shap_local_cases.json")
        np.save(run_dir / "shap_values.npy", shap_values)
        print(f"  → {run_dir / 'shap_global.json'}")

    # ── Cross-model consistency (Jaccard) ────────────────────────
    if len(all_rankings) >= 2:
        print(f"\n{'─' * 60}")
        print("  Cross-Model Feature Consistency (Jaccard, top-10)")
        print(f"{'─' * 60}")

        consistency = {}
        model_list = [model for model in all_rankings if model in SHAP_MODELS]
        for i, m1 in enumerate(model_list):
            for m2 in model_list[i+1:]:
                j = jaccard(all_rankings[m1], all_rankings[m2])
                key = f"{m1}_vs_{m2}"
                consistency[key] = round(j, 3)
                print(f"  {m1:<15s} vs {m2:<15s} : {j:.3f}")

        # Save full Jaccard matrix
        jaccard_matrix = {}
        for m1 in model_list:
            jaccard_matrix[m1] = {}
            for m2 in model_list:
                if m1 == m2:
                    jaccard_matrix[m1][m2] = 1.0
                else:
                    jaccard_matrix[m1][m2] = jaccard(all_rankings[m1],
                                                     all_rankings[m2])

        # Save to LGBM run dir (or first available)
        ref_dir = find_latest_run(DATASET_LABEL, "lgbm") or \
                  find_latest_run(DATASET_LABEL, model_list[0])
        save_json({
            "models": model_list,
            "top_k": 10,
            "rankings": all_rankings,
            "jaccard_matrix": jaccard_matrix,
        }, ref_dir / "shap_consistency.json")
        print(f"  → {ref_dir / 'shap_consistency.json'}")

    # ── Cross-variant stability ──────────────────────────────────
    if not args.skip_variants and "lgbm" in models:
        print(f"\n{'─' * 60}")
        print("  Cross-Variant Feature Stability (LGBM, top-10)")
        print(f"{'─' * 60}")

        lgbm_dir = find_latest_run(DATASET_LABEL, "lgbm")
        lgbm_pipeline = joblib.load(lgbm_dir / "model.joblib")

        t0 = time.time()
        variant_rankings = compute_variant_shap(lgbm_pipeline, top_k=10)
        elapsed = time.time() - t0
        print(f"  Computed in {elapsed:.1f}s")

        # Jaccard matrix
        var_keys = list(variant_rankings.keys())
        var_jaccard = {}
        for v1 in var_keys:
            var_jaccard[v1] = {}
            for v2 in var_keys:
                if v1 == v2:
                    var_jaccard[v1][v2] = 1.0
                else:
                    var_jaccard[v1][v2] = jaccard(variant_rankings[v1],
                                                  variant_rankings[v2])

        print("\n  Jaccard matrix:")
        labels = {"baf_base": "Base", **TARGET_VARIANTS}
        for v1 in var_keys:
            row = "  " + f"{labels.get(v1, v1):<10s}"
            for v2 in var_keys:
                row += f" {var_jaccard[v1][v2]:.2f}"
            print(row)

        save_json({
            "variant_keys": var_keys,
            "top_k": 10,
            "rankings": variant_rankings,
            "jaccard_matrix": var_jaccard,
        }, lgbm_dir / "shap_variant_stability.json")
        print(f"  → {lgbm_dir / 'shap_variant_stability.json'}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  SHAP Analysis Complete!")
    print(f"{'=' * 60}")


def main():
    """Generate compatible figures from pinned saved explanations only."""
    parser = argparse.ArgumentParser(description="Revision-safe compatible SHAP figure generation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from generate_revision_interpretability import generate
    generate(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
