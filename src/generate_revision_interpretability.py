"""Generate audited BAF interpretability figures from explicitly selected runs.

This module never trains a model or modifies the selected source runs. Outputs
must be written outside the preserved results, results_thesis and Overleaf
directories. The manifest must contain a ``baseline_runs`` mapping with the
keys ``logreg``, ``lgbm`` and ``catboost``; paths are relative to the repository
root unless absolute. Only architecture-compatible SHAP artefacts are used.

Example:
    python src/generate_revision_interpretability.py \
        --manifest <revision-manifest.json> --output-dir <revision-output-dir>

Categorical global importance is the mean of the sum of absolute one-hot
contributions. Local explanations instead sum signed one-hot contributions,
which preserves additivity at the original-feature level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MODELS = ("logreg", "lgbm", "catboost")
LABELS = {"logreg": "Logistic Regression", "lgbm": "LightGBM", "catboost": "CatBoost"}
COLOURS = {"logreg": "#7561a8", "lgbm": "#2677a6", "catboost": "#b73d46"}
CATEGORICAL = ("payment_type", "employment_status", "housing_status", "source", "device_os")


@dataclass
class ShapArtefacts:
    model: str
    run_dir: Path
    feature_names: list[str]
    shap_values: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    threshold: float
    original_features: list[str]
    global_importance: np.ndarray
    expected_logit: float
    max_additivity_error: float
    threshold_source_run: Path | None = None


def original_feature_name(name: str) -> str:
    """Return the source feature for an encoded column, without prefix guesses."""
    return next((feature for feature in CATEGORICAL if name.startswith(feature + "_")), name)


def group_contributions(values: np.ndarray, names: list[str], *, absolute: bool) -> tuple[np.ndarray, list[str]]:
    """Group one-hot columns; choose absolute importance or signed attribution."""
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("SHAP matrix and feature-name dimensions do not match")
    parents = [original_feature_name(name) for name in names]
    features = list(dict.fromkeys(parents))
    indices = {name: index for index, name in enumerate(features)}
    grouped = np.zeros((values.shape[0], len(features)), dtype=np.float64)
    for column, parent in enumerate(parents):
        contribution = values[:, column]
        grouped[:, indices[parent]] += np.abs(contribution) if absolute else contribution
    return grouped, features


def read_manifest(path: Path) -> dict[str, Path]:
    """Resolve exactly the runs named by the manifest; never select the latest."""
    document = json.loads(path.read_text(encoding="utf-8"))
    mapping = document.get("interpretability", {}).get("baseline_runs")
    if mapping is None:
        mapping = document.get("baseline_runs", {})
        if "baf_base" in mapping:
            mapping = mapping["baf_base"]
    if not isinstance(mapping, dict) or not all(model in mapping for model in MODELS):
        raise ValueError("Manifest requires baseline_runs for logreg, lgbm and catboost")
    resolved = {}
    for model in MODELS:
        entry = mapping[model]
        run = Path(entry["run_dir"] if isinstance(entry, dict) else entry)
        run = (ROOT / run).resolve() if not run.is_absolute() else run.resolve()
        if not run.is_dir():
            raise FileNotFoundError(run)
        if isinstance(entry, dict) and entry.get("config_sha256") and file_digest(run / "config.json") != entry["config_sha256"]:
            raise ValueError(f"Manifest-selected SHAP configuration changed: {run}")
        resolved[model] = run
    return resolved


def validate_output_directory(path: Path) -> Path:
    """Protect preserved scientific artefacts from accidental replacement."""
    destination = path.resolve()
    for name in ("results", "results_thesis", "Overleaf"):
        protected = (ROOT / name).resolve()
        if destination == protected or protected in destination.parents:
            raise ValueError(f"Revision outputs must not replace preserved {name} artefacts")
    return destination


def load_artefacts(model: str, run_dir: Path) -> ShapArtefacts:
    """Read saved data and verify log-odds additivity against saved predictions."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("dataset") != "baf_base" or config.get("strategy") != "none":
        raise ValueError(f"Expected a BAF Base baseline run: {run_dir}")
    pipeline = joblib.load(run_dir / "model.joblib")
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    values = np.load(run_dir / "shap_values.npy", mmap_mode="r")
    labels = np.load(run_dir / "y_test.npy")
    scores = np.load(run_dir / "y_test_scores.npy")
    metrics = json.loads((run_dir / "metrics_test.json").read_text(encoding="utf-8"))
    if values.shape != (len(scores), len(names)) or len(labels) != len(scores):
        raise ValueError(f"SHAP and TEST dimensions differ: {run_dir}")
    if not np.isfinite(values).all() or not np.isfinite(scores).all():
        raise ValueError(f"Non-finite SHAP values or predictions: {run_dir}")
    if not np.all((scores > 0) & (scores < 1)):
        raise ValueError("Cannot infer the raw-logit baseline from saturated probabilities")
    raw_logits = np.log(scores) - np.log1p(-scores)
    baseline_by_row = raw_logits - values.sum(axis=1)
    expected = float(np.median(baseline_by_row))
    error = float(np.max(np.abs(baseline_by_row - expected)))
    if error > 1e-7:
        raise ValueError(f"Saved SHAP values fail log-odds additivity: {model}, error={error:g}")
    grouped, features = group_contributions(values, names, absolute=True)
    return ShapArtefacts(model, run_dir, names, values, labels, scores,
                         float(metrics["threshold"]), features, grouped.mean(axis=0), expected, error)


def apply_operating_baseline(artefact: ShapArtefacts, selected_run: Path) -> dict:
    """Use a recovered exact primary threshold only after verifying frozen scores."""
    selected_run = selected_run.resolve()
    config = json.loads((selected_run / "config.json").read_text(encoding="utf-8"))
    if (config.get("dataset"), config.get("model"), config.get("strategy")) != ("baf_base", artefact.model, "none"):
        raise ValueError("The operating baseline has a different dataset, model or intervention")
    source_model_digest = config.get("source_final_model_sha256")
    if source_model_digest and file_digest(artefact.run_dir / "model.joblib") != source_model_digest:
        raise ValueError("Saved SHAP explains a different historical final model from the recovered primary baseline")
    if "threshold_exact" not in config or not np.isfinite(config["threshold_exact"]):
        raise ValueError("The operating baseline lacks its recovered exact threshold")
    labels = np.load(selected_run / "y_test.npy", allow_pickle=False)
    scores = np.load(selected_run / "y_test_scores.npy", allow_pickle=False)
    if not np.array_equal(labels, artefact.labels) or scores.shape != artefact.scores.shape:
        raise ValueError("Historical compatible SHAP cannot explain changed primary TEST labels or score dimensions")
    maximum_score_difference = float(np.max(np.abs(scores - artefact.scores)))
    if not np.isfinite(scores).all() or not np.allclose(scores, artefact.scores, rtol=0, atol=1e-12):
        raise ValueError("Historical compatible SHAP cannot explain materially changed primary TEST scores")
    exact_threshold = float(config["threshold_exact"])
    decisions_changed = int(np.sum((scores >= exact_threshold) != (artefact.scores >= exact_threshold)))
    if decisions_changed:
        raise ValueError("Numerically equivalent scores change primary threshold decisions")
    historical_cases = select_local_cases(artefact)
    historical_threshold = artefact.threshold
    artefact.threshold = float(config["threshold_exact"])
    artefact.threshold_source_run = selected_run
    predictions = artefact.scores >= artefact.threshold
    unchanged_cases = {case: bool(predictions[index]) == (case != "FN")
                       for case, index in historical_cases.items()}
    if not all(unchanged_cases.values()):
        raise ValueError("A saved selected local case changes error class under the recovered exact primary threshold")
    return {"source_run": str(selected_run), "threshold_exact": artefact.threshold,
            "historical_rounded_threshold": historical_threshold,
            "maximum_absolute_score_difference": maximum_score_difference,
            "score_comparison_absolute_tolerance": 1e-12,
            "scores_bitwise_equal": bool(np.array_equal(scores, artefact.scores)),
            "threshold_decisions_changed": decisions_changed,
            "selected_case_classes_unchanged": unchanged_cases,
            "source_sha256": {name: file_digest(selected_run / name) for name in
                              ("config.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy")}}


def operating_run_from_manifest(document: dict, model: str) -> Path | None:
    entry = document.get("baseline_runs", {}).get(f"baf_base/{model}")
    if entry is None:
        return None
    path = Path(entry["run_dir"] if isinstance(entry, dict) else entry)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if isinstance(entry, dict) and entry.get("config_sha256") and file_digest(path / "config.json") != entry["config_sha256"]:
        raise ValueError("The pinned operating baseline configuration changed")
    return path


def jaccard_matrix(artefacts: dict[str, ShapArtefacts], top_k: int = 10) -> dict:
    """Compare complete source-feature importance rankings before truncation."""
    rankings = {}
    for model, artefact in artefacts.items():
        order = np.argsort(-artefact.global_importance, kind="stable")[:top_k]
        rankings[model] = [artefact.original_features[index] for index in order]
    matrix = {}
    for first in MODELS:
        matrix[first] = {}
        for second in MODELS:
            left, right = set(rankings[first]), set(rankings[second])
            matrix[first][second] = len(left & right) / len(left | right)
    return {"top_k": top_k, "rankings": rankings, "jaccard_matrix": matrix}


def select_local_cases(artefact: ShapArtefacts) -> dict[str, int]:
    """Select high-score TP/FP and the lowest-score FN, documenting selection."""
    predictions = artefact.scores >= artefact.threshold
    masks = {"TP": (artefact.labels == 1) & predictions,
             "FP": (artefact.labels == 0) & predictions,
             "FN": (artefact.labels == 1) & ~predictions}
    selected = {}
    for case, mask in masks.items():
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            raise ValueError(f"No {case} case at the selected LGBM threshold")
        index = np.argmin(artefact.scores[candidates]) if case == "FN" else np.argmax(artefact.scores[candidates])
        selected[case] = int(candidates[index])
    return selected


def plot_global(artefacts: dict[str, ShapArtefacts], destination: Path) -> None:
    """Draw a separate, correctly ranked top-15 panel for each compatible model."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 8.0))
    for ax, model in zip(axes, MODELS):
        artefact = artefacts[model]
        order = np.argsort(-artefact.global_importance, kind="stable")[:15]
        values = artefact.global_importance[order]
        names = [artefact.original_features[index].replace("_", " ") for index in order]
        ax.barh(np.arange(len(order)), values, color=COLOURS[model], height=.72)
        ax.set_yticks(np.arange(len(order)), names, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, float(values.max()) * 1.18)
        ax.set_title(LABELS[model], fontsize=10, pad=8, fontweight="semibold")
        ax.tick_params(axis="x", labelsize=8.5)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.40, .015, "Mean absolute SHAP contribution (log-odds)\n"
             "Categorical importance: sum of absolute one-hot contributions", fontsize=8.5)
    fig.subplots_adjust(left=.40, right=.98, bottom=.075, top=.965, hspace=.40)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def plot_local(artefact: ShapArtefacts, selected: dict[str, int], destination: Path) -> dict:
    """Draw complete signed-feature waterfalls, including unshown contributions."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 8.0))
    details = {}
    titles = {"TP": "True positive", "FP": "False positive", "FN": "False negative"}
    for ax, case in zip(axes, ("TP", "FP", "FN")):
        index = selected[case]
        grouped, names = group_contributions(artefact.shap_values[index:index + 1],
                                             artefact.feature_names, absolute=False)
        contributions = grouped[0]
        order = np.argsort(-np.abs(contributions), kind="stable")
        displayed = order[:10]
        remaining = float(contributions[order[10:]].sum())
        sequence = [float(contributions[column]) for column in displayed] + [remaining]
        feature_names = [names[column].replace("_", " ") for column in displayed] + ["Remaining features"]
        starts = artefact.expected_logit + np.concatenate(([0.0], np.cumsum(sequence[:-1])))
        finishes = starts + sequence
        rows = np.arange(len(sequence))
        for row, start, finish, value in zip(rows, starts, finishes, sequence):
            ax.barh(row, abs(value), left=min(start, finish), height=.65,
                    color="#bd4954" if value >= 0 else "#2878ad")
            ax.text(max(start, finish) + .035, row, f"{value:+.3f}", va="center", fontsize=8.5)
            if row < len(rows) - 1:
                ax.plot([finish, finish], [row + .34, row + .66], color="#9c9c9c", linewidth=.65)
        output_logit = float(np.log(artefact.scores[index]) - np.log1p(-artefact.scores[index]))
        error = abs(float(finishes[-1]) - output_logit)
        if error > 1e-7:
            raise ValueError(f"Waterfall does not reconstruct output for {case}")
        ax.axvline(artefact.expected_logit, color="#777777", linestyle=":", linewidth=.9)
        ax.axvline(output_logit, color="#333333", linestyle="--", linewidth=.9)
        ax.set_yticks(rows, feature_names, fontsize=8.5)
        ax.invert_yaxis()
        extrema = [*starts, *finishes, artefact.expected_logit, output_logit]
        span = max(max(extrema) - min(extrema), 1.0)
        ax.set_xlim(min(extrema) - .09 * span, max(extrema) + .30 * span)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_title(f"{titles[case]}: P(fraud) = {artefact.scores[index]:.4f}; "
                     f"logit = {output_logit:.3f}", fontsize=9.5, pad=9)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        details[case] = {"test_position": index, "true_label": int(artefact.labels[index]),
                         "score": float(artefact.scores[index]), "expected_logit": artefact.expected_logit,
                         "output_logit": output_logit, "remaining_signed_contribution": remaining,
                         "additivity_error": error,
                         "displayed_signed_features": {names[column]: float(contributions[column]) for column in displayed}}
    fig.text(.40, .012, "Raw model output (log-odds)\n"
             f"Dotted: expected logit = {artefact.expected_logit:.3f}; dashed: output logit\n"
             f"Fraud-probability threshold = {artefact.threshold:.5f}", fontsize=8.5)
    fig.subplots_adjust(left=.40, right=.985, bottom=.085, top=.95, hspace=.45)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return details


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(manifest: Path, output_dir: Path) -> dict:
    """Generate figures and a machine-readable numeric/provenance QA record."""
    import matplotlib
    matplotlib.use("Agg")
    output_dir = validate_output_directory(output_dir)
    runs = read_manifest(manifest)
    artefacts = {model: load_artefacts(model, run) for model, run in runs.items()}
    document = json.loads(manifest.read_text(encoding="utf-8"))
    operating_baselines = {}
    for model, artefact in artefacts.items():
        selected = operating_run_from_manifest(document, model)
        if selected is not None:
            operating_baselines[model] = apply_operating_baseline(artefact, selected)
    reference_labels = artefacts["lgbm"].labels
    if any(not np.array_equal(item.labels, reference_labels) for item in artefacts.values()):
        raise ValueError("Compatible model runs do not use identical BAF TEST labels")
    output_dir.mkdir(parents=True, exist_ok=True)
    global_path = output_dir / "shap_global_compatible.pdf"
    local_path = output_dir / "shap_local_waterfalls.pdf"
    plot_global(artefacts, global_path)
    local = plot_local(artefacts["lgbm"], select_local_cases(artefacts["lgbm"]), local_path)
    provenance = {}
    for model, artefact in artefacts.items():
        provenance[model] = {"run_directory": str(artefact.run_dir), "test_samples": len(artefact.labels),
                             "fraud_samples": int(artefact.labels.sum()),
                             "expected_logit": artefact.expected_logit,
                             "maximum_logit_additivity_error": artefact.max_additivity_error,
                             "complete_global_importance": dict(zip(artefact.original_features, artefact.global_importance.tolist())),
                             "source_sha256": {name: file_digest(artefact.run_dir / name) for name in
                                               ("config.json", "model.joblib", "shap_values.npy", "y_test.npy", "y_test_scores.npy")}}
    report = {"manifest_path": str(manifest.resolve()), "global_units": "log-odds",
              "global_grouping": "mean(sum(abs(one-hot SHAP contributions)))",
              "local_grouping": "sum(signed one-hot SHAP contributions)",
              "local_selection": "highest-score TP and FP; lowest-score FN; selected for illustration only",
              "operating_baselines": operating_baselines,
              "models": provenance, "consistency": jaccard_matrix(artefacts), "local_cases": local,
              "outputs": {path.name: file_digest(path) for path in (global_path, local_path)}}
    (output_dir / "interpretability_qa.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.manifest, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "models": list(report["models"]), "local_cases": report["local_cases"]}, indent=2))


if __name__ == "__main__":
    main()
