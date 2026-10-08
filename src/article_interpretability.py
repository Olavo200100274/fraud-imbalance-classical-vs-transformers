"""Portable, saved-evidence replay of Article 1 interpretability figures.

The author-only export reads the frozen LGBM preprocessing pipeline and full
archived attributions. It transforms the documented TEST features but never
fits a model, runs model inference or calculates new SHAP values. Only the
fifteen rows actually displayed by the published beeswarm are exported: the
fourteen leading encoded contributions and the signed sum of the remainder.
All 200,000 TEST observations are retained, without sampling or rounding.

Replay uses these plot-ready inputs and the public explanation summaries. It
does not require raw datasets or model weights. Regenerated visual encodings
represent the same saved evidence; PDF/PNG bytes are not promised identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MODELS = ("logreg", "lgbm", "catboost")
LABELS = {"logreg": "Logistic Regression", "lgbm": "LightGBM", "catboost": "CatBoost"}
COLOURS = {"logreg": "#7561a8", "lgbm": "#2677a6", "catboost": "#b73d46"}
INPUT_DIRECTORY = Path("results/interpretability/plot_inputs")


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _source(root: Path, value: str) -> Path:
    candidate = Path(value)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("An interpretability source escapes the author archive")
    return candidate


def _checked(path: Path, expected: str) -> None:
    if _digest(path) != expected:
        raise ValueError(f"Pinned interpretability source changed: {path.name}")


def _prepare_beeswarm_arrays(values, features, names):
    """Match SHAP 0.51's max_display=15, no-clustering grouping exactly.

    SHAP orders by mean absolute encoded attribution, then replaces the
    fifteenth displayed column with the signed sum of all remaining columns.
    Its colour data for that aggregate stays the original fifteenth encoded
    feature's values; retaining those values reproduces that plotting policy,
    not a meaningful feature-value scale for the combined contribution.
    """
    values, features = np.asarray(values), np.asarray(features)
    names = list(names)
    if (values.ndim != 2 or values.shape != features.shape
            or values.shape[1] != len(names) or len(names) < 15
            or not np.isfinite(values).all() or not np.isfinite(features).all()):
        raise ValueError("Beeswarm attribution/features must be finite aligned matrices with at least 15 columns")
    # This is Explanation.abs.mean(0).argsort.flip, not a ranking of original
    # categorical groups and not a ranking after residual aggregation.
    order = np.argsort(np.abs(values).mean(axis=0))[::-1]
    displayed = np.column_stack((values[:, order[:14]],
                                 np.sum([values[:, index] for index in order[14:]], axis=0)))
    colour_data = features[:, order[:15]].copy()
    display_names = [names[index] for index in order[:14]] + [f"Sum of {len(order) - 14} other features"]
    return {"shap_values": displayed, "feature_values": colour_data,
            "feature_names": np.asarray(display_names, dtype=str),
            "encoded_feature_order": order.astype(np.int64),
            "original_feature_names": np.asarray(names, dtype=str)}


def export_plot_inputs(author_root: Path, destination: Path) -> list[Path]:
    """Export complete displayed plot data into a dedicated repository root.

    Sources are verified against the existing frozen reporting/attention pins.
    Existing destination files are never overwritten, nor are author sources
    modified. The destination may be a checkout inside an ignored staging
    directory, but cannot be the author repository itself or a source archive.
    """
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split

    author_root, destination = author_root.resolve(), destination.resolve()
    if destination == author_root or any(destination == author_root / name
                                        or (author_root / name) in destination.parents
                                        for name in ("results", "results_revision", "Overleaf", "results_thesis")):
        raise ValueError("Plot export must not overwrite the retained author evidence")
    folder = destination / INPUT_DIRECTORY
    targets = [folder / name for name in ("beeswarm.npz", "attention.npz", "source_manifest.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Use a fresh plot-input destination; existing files are not overwritten")
    revision = author_root / "results_revision/20261005"
    qa_path = revision / "derived/reporting/generation_qa.json"
    qa = _read(qa_path)
    manifest_path = revision / "revision_manifest.json"
    _checked(manifest_path, qa["manifest_sha256"])
    manifest = _read(manifest_path)
    pins = qa["compatible_shap_sources"]["lgbm"]["artefacts"]
    sources = {}
    for name in ("config.json", "model.joblib", "shap_values.npy", "y_test.npy"):
        path = _source(author_root, pins[name]["path"])
        _checked(path, pins[name]["sha256"])
        sources[path.relative_to(author_root).as_posix()] = pins[name]["sha256"]
    run = _source(author_root, qa["compatible_shap_sources"]["lgbm"]["run_dir"])
    config = _read(run / "config.json")
    raw = author_root / "datasets/Base.csv"
    raw_hash = manifest["datasets"]["baf_base"]["raw_file_sha256"]
    _checked(raw, raw_hash)
    if config["dataset_hash_sha256"] != raw_hash or config["split_seed"] != 42:
        raise ValueError("Beeswarm must use the documented frozen BAF data and split")
    frame = pd.read_csv(raw).drop(columns="month")
    _, x_test, _, y_test = train_test_split(frame.drop(columns="fraud_bool"), frame["fraud_bool"],
                                         test_size=0.2, stratify=frame["fraud_bool"], random_state=42)
    if not np.array_equal(y_test.to_numpy(), np.load(run / "y_test.npy", allow_pickle=False)):
        raise ValueError("Raw-file TEST labels do not match the pinned LGBM attribution order")
    pipeline = joblib.load(run / "model.joblib")
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(x_test)
    if hasattr(transformed, "to_numpy"):
        transformed = transformed.to_numpy()
    elif hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    beeswarm = _prepare_beeswarm_arrays(np.load(run / "shap_values.npy", allow_pickle=False),
                                       transformed, preprocessor.get_feature_names_out())
    if beeswarm["shap_values"].shape != (200000, 15):
        raise ValueError("The full BAF TEST beeswarm population is required")
    sources["datasets/Base.csv"] = raw_hash
    sources[qa_path.relative_to(author_root).as_posix()] = _digest(qa_path)
    sources[manifest_path.relative_to(author_root).as_posix()] = qa["manifest_sha256"]

    attention_folder = revision / "derived/interpretability/attention"
    attention_path = attention_folder / "attention_summary.json"
    attention_summary = _read(attention_path)
    attention = {}
    source_names = ["attention_cls_weights.npy"] + [f"attention_matrix_{kind}_{rank}.npy"
                                                   for kind in ("FP", "FN") for rank in (1, 2, 3)]
    for name in source_names:
        path = attention_folder / name
        expected = attention_summary["derived_artefacts_sha256"][name]
        _checked(path, expected)
        array = np.load(path, allow_pickle=False)
        expected_shape = (200000, 31) if name == "attention_cls_weights.npy" else (31, 31)
        if array.shape != expected_shape or not np.isfinite(array).all():
            raise ValueError("Saved attention array is malformed")
        attention[Path(name).stem] = array
        sources[path.relative_to(author_root).as_posix()] = expected
    attention["feature_token_names"] = np.asarray(attention_summary["feature_token_names"], dtype=str)
    sources[attention_path.relative_to(author_root).as_posix()] = _digest(attention_path)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(targets[0], **beeswarm)
    np.savez_compressed(targets[1], **attention)
    record = {"schema_version": 1, "article": 1, "model_fitting": False, "model_inference": False,
              "new_shap_computation": False, "test_row_sampling": False,
              "beeswarm": {"rows": 200000, "original_encoded_columns": len(beeswarm["original_feature_names"]),
                           "displayed_columns": 15, "individual_columns": 14,
                           "remaining_signed_columns": len(beeswarm["original_feature_names"]) - 14,
                           "colour_policy": "Exact original SHAP 0.51 policy: residual colour is the fifteenth ranked encoded feature, not an aggregate feature value",
                           "order_policy": "Descending mean absolute original encoded attribution before residual aggregation",
                           "dtype_preserved": True, "jitter_seed": 42},
              "attention": {"scope": attention_summary["aggregation_scope"], "dtype_preserved": True,
                            "storage_precision_note": "Archived float32 weights replay plots; authoritative float64-accumulated summary is retained separately"},
              "source_sha256": sources,
              "archives": {path.name: {"sha256": _digest(path), "bytes": path.stat().st_size}
                           for path in targets[:2]},
              "replay_scope": "Same complete displayed saved evidence; not fresh inference, not causal attribution, and not bitwise PDF/PNG reproduction"}
    targets[2].write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return targets


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    return plt


def _plot_global(summary, destination, plt):
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 8.0))
    for axis, model in zip(axes, MODELS):
        importance = summary["models"][model]["complete_global_importance"]
        names = sorted(importance, key=lambda name: -importance[name])[:15]
        values = [importance[name] for name in names]
        axis.barh(np.arange(15), values, color=COLOURS[model], height=.72)
        axis.set_yticks(np.arange(15), [name.replace("_", " ") for name in names], fontsize=8.5)
        axis.invert_yaxis()
        axis.set_xlim(0, max(values) * 1.18)
        axis.set_title(LABELS[model], fontsize=10, fontweight="semibold")
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.40, .015, "Mean absolute SHAP contribution (log-odds)\nCategorical importance: sum of absolute one-hot contributions", fontsize=8.5)
    fig.subplots_adjust(left=.40, right=.98, bottom=.075, top=.965, hspace=.40)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def _waterfall_sequence(record):
    contributions = record["displayed_signed_features"]
    names = sorted(contributions, key=lambda name: -abs(contributions[name]))
    values = np.asarray([contributions[name] for name in names] + [record["remaining_signed_contribution"]])
    if not np.isfinite(values).all() or abs(record["expected_logit"] + values.sum() - record["output_logit"]) > 1e-7:
        raise ValueError("Published waterfall contributions do not reconstruct the recorded output")
    return [name.replace("_", " ") for name in names] + ["Remaining features"], values


def _plot_waterfalls(summary, destination, plt):
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 8.0))
    titles = {"TP": "True positive", "FP": "False positive", "FN": "False negative"}
    for axis, case in zip(axes, ("TP", "FP", "FN")):
        record = summary["local_cases"][case]
        names, sequence = _waterfall_sequence(record)
        starts = record["expected_logit"] + np.concatenate(([0.0], np.cumsum(sequence[:-1])))
        finishes = starts + sequence
        rows = np.arange(len(sequence))
        for row, start, finish, value in zip(rows, starts, finishes, sequence):
            axis.barh(row, abs(value), left=min(start, finish), height=.65,
                      color="#bd4954" if value >= 0 else "#2878ad")
            axis.text(max(start, finish) + .035, row, f"{value:+.3f}", va="center", fontsize=8.5)
            if row < len(rows) - 1:
                axis.plot([finish, finish], [row + .34, row + .66], color="#9c9c9c", linewidth=.65)
        axis.axvline(record["expected_logit"], color="#777777", linestyle=":", linewidth=.9)
        axis.axvline(record["output_logit"], color="#333333", linestyle="--", linewidth=.9)
        axis.set_yticks(rows, names, fontsize=8.5)
        axis.invert_yaxis()
        extrema = [*starts, *finishes, record["expected_logit"], record["output_logit"]]
        span = max(max(extrema) - min(extrema), 1.0)
        axis.set_xlim(min(extrema) - .09 * span, max(extrema) + .30 * span)
        axis.set_title(f"{titles[case]}: P(fraud) = {record['score']:.4f}; logit = {record['output_logit']:.3f}", fontsize=9.5)
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.40, .012, "Raw model output (log-odds)\nDotted: expected logit; dashed: case output\nSelected illustrative extremes; not representative error samples", fontsize=8.5)
    fig.subplots_adjust(left=.40, right=.985, bottom=.085, top=.95, hspace=.45)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def _plot_beeswarm(arrays, destination, plt):
    import shap
    values, features = arrays["shap_values"], arrays["feature_values"]
    names = arrays["feature_names"].tolist()
    if values.shape != features.shape or values.shape != (200000, 15):
        raise ValueError("The complete published beeswarm population is missing")
    explanation = shap.Explanation(values=values, data=features, feature_names=names)
    fig, axis = plt.subplots(figsize=(6.6, 6.5))
    state = np.random.get_state()
    try:
        np.random.seed(42)  # The existing figure's cosmetic jitter, not row selection.
        shap.plots.beeswarm(explanation, max_display=15, order=np.arange(15),
                            group_remaining_features=False, show=False, ax=axis, plot_size=None)
    finally:
        np.random.set_state(state)
    axis.set_title("SHAP Beeswarm — LGBM on BAF Base", fontsize=10.5)
    axis.tick_params(axis="y", labelsize=11)
    axis.set_xlabel("SHAP contribution (log-odds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_attention(arrays, summary, aggregate_path, heatmap_path, plt):
    names = arrays["feature_token_names"].tolist()
    weights = arrays["attention_cls_weights"]
    if weights.shape != (200000, 31) or len(names) != 31 or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("The complete 31-token attention population is malformed")
    means, deviations = weights.mean(axis=0, dtype=np.float64), weights.std(axis=0, dtype=np.float64)
    order = np.argsort(-means[1:], kind="stable")
    n_num = int(summary["n_numerical_features"])
    colours = np.asarray(["#4472C4"] * n_num + ["#ED7D31"] * int(summary["n_categorical_features"]))
    fig, axis = plt.subplots(figsize=(6.6, 5.0))
    axis.barh(np.arange(30), means[1:][order], xerr=deviations[1:][order], color=colours[order],
              height=.72, capsize=1.5, error_kw={"linewidth": .55, "alpha": .7})
    axis.set_yticks(np.arange(30), [names[index + 1].replace("_", " ") for index in order], fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("Mean attention weight ± across-row SD", fontsize=8.5)
    axis.set_title(f"Final-layer, head-averaged CLS attention (N={len(weights):,})\nMean CLS self-attention: {means[0]:.4f}", fontsize=9)
    axis.axvline(1 / 31, color="#444444", linestyle=":", linewidth=1, label="Uniform 1/31")
    axis.axvline(means[1:1 + n_num].mean(), color="#4472C4", linestyle="--", linewidth=.9, label="Numerical mean")
    axis.axvline(means[1 + n_num:].mean(), color="#ED7D31", linestyle="--", linewidth=.9, label="Categorical mean")
    axis.grid(axis="x", alpha=.2)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(.66, .025))
    fig.subplots_adjust(left=.405, right=.985, top=.905, bottom=.17)
    fig.savefig(aggregate_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    matrix = arrays["attention_matrix_FP_1"]
    if matrix.shape != (31, 31) or not np.isfinite(matrix).all():
        raise ValueError("The selected false-positive attention matrix is malformed")
    fig, axis = plt.subplots(figsize=(6.6, 6.2))
    im = axis.imshow(matrix, cmap="Blues", aspect="auto")
    axis.set_xticks(np.arange(31), [str(index + 1) for index in range(31)], rotation=90, fontsize=7.5)
    axis.set_yticks(np.arange(31), [f"{index + 1}  {name.replace('_', ' ')}" for index, name in enumerate(names)], fontsize=8)
    axis.set_xlabel("Key token (index matches the corresponding row label)", fontsize=8.5)
    axis.set_ylabel("Query token", fontsize=8.5)
    axis.set_title("Full Attention Matrix — FP Case #1", fontsize=9)
    fig.colorbar(im, ax=axis, fraction=.04, pad=.02).ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.37, right=.965, top=.925, bottom=.12)
    fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def regenerate(repository: Path, output: Path) -> list[Path]:
    """Regenerate five Article 1 plots without fitting, inference or raw data."""
    repository, output = repository.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; published figures are not overwritten")
    for protected in (repository / "results", repository / "publications", repository / "manuscript"):
        if output == protected or protected in output.parents:
            raise ValueError("Replay output must be separate from published evidence")
    folder = repository / INPUT_DIRECTORY
    record = _read(folder / "source_manifest.json")
    if record["schema_version"] != 1 or record["article"] != 1:
        raise ValueError("Unsupported interpretability plot-input manifest")
    for name in ("beeswarm.npz", "attention.npz"):
        _checked(folder / name, record["archives"][name]["sha256"])
    summary = _read(repository / "results/interpretability/shap_summary.json")
    attention_summary = _read(repository / "results/interpretability/attention_summary.json")
    with np.load(folder / "beeswarm.npz", allow_pickle=False) as data:
        beeswarm = {name: data[name] for name in data.files}
    with np.load(folder / "attention.npz", allow_pickle=False) as data:
        attention = {name: data[name] for name in data.files}
    # Verify additive summary records before writing any plot.
    for case in ("TP", "FP", "FN"):
        _waterfall_sequence(summary["local_cases"][case])
    output.mkdir(parents=True, exist_ok=False)
    paths = [output / name for name in ("shap_global.pdf", "shap_waterfall.pdf", "shap_beeswarm.pdf",
                                       "attention_aggregate.png", "attention_heatmap_fp.png")]
    plt = _pyplot()
    _plot_global(summary, paths[0], plt)
    _plot_waterfalls(summary, paths[1], plt)
    _plot_beeswarm(beeswarm, paths[2], plt)
    _plot_attention(attention, attention_summary, paths[3], paths[4], plt)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--author-root", type=Path)
    parser.add_argument("--export-to", type=Path)
    args = parser.parse_args(argv)
    if args.export_to:
        if args.author_root is None:
            parser.error("Author-only export requires --author-root")
        paths = export_plot_inputs(args.author_root, args.export_to)
    else:
        if args.output is None:
            parser.error("Replay requires a new --output directory")
        paths = regenerate(args.repository, args.output)
    print(json.dumps({"files": [str(path) for path in paths], "fitting": False, "inference": False}, indent=2))


if __name__ == "__main__":
    main()
