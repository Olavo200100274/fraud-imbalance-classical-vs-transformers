"""Export verified, predefined BAF sensitivity tables to a new LaTeX fragment.

The primary factorial matrix is not altered. Every point estimate is checked
against explicitly referenced saved TEST scores; the paired intervals are read
from the completed analysis, never recalculated or selected here. No predictive
model is loaded, fitted or retuned, and no threshold is selected using TEST.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from experiment_protocol import PROJECT_ROOT, array_sha256, file_sha256
from generate_revision_interpretability import validate_output_directory
from revision_paired_analysis import MODELS


MODEL_LABELS = {"logreg": "Logistic Regression", "rf": "Random Forest",
                "lgbm": "LGBM", "catboost": "CatBoost",
                "fttransformer": "FT-Transformer", "ocsvm": "OCSVM"}
METRICS = ("average_precision", "F2")
BOOTSTRAP_REQUESTED = 1000
COLUMN_WIDTHS_MM = (26, 12, 18, 23, 55)
TABCOLSEP_PT = 3
TABLE_WIDTH_MM = sum(COLUMN_WIDTHS_MM) + 8 * TABCOLSEP_PT * 25.4 / 72.27
THESIS_TEXT_WIDTH_MM = 150


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite(value, description: str, *, lower=None, upper=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{description} must be a finite number")
    value = float(value)
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f"{description} lies outside its permissible range")
    return value


def _interval(values, description: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"A two-bound paired interval is required: {description}")
    lower, upper = (_finite(value, description, lower=-1, upper=1) for value in values)
    if lower > upper:
        raise ValueError(f"Paired interval bounds are reversed: {description}")
    return [lower, upper]


def _valid_replicates(value, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= BOOTSTRAP_REQUESTED:
        raise ValueError(f"{description} requires a positive valid count within the declared 1000 replicates")
    return value


def _point_metrics(run: Path, model: str, strategy: str, threshold: float,
                   *, missing_policy="preserve", historical=False) -> dict:
    """Reproduce AP and count-derived F2 without reconstructing a historical tau."""
    run = Path(run)
    run = (run if run.is_absolute() else PROJECT_ROOT / run).resolve()
    config, saved = _read_json(run / "config.json"), _read_json(run / "metrics_test.json")
    allowed_strategies = {"none", "n/a"} if model == "ocsvm" and strategy == "none" else {strategy}
    if (config.get("dataset") != "baf_base" or config.get("model") != model
            or config.get("strategy") not in allowed_strategies
            or config.get("missing_policy", "preserve") != missing_policy):
        raise ValueError("A sensitivity source has the wrong dataset, model, strategy or missingness policy")
    if config.get("sample_fraction") not in (None, 1, 1.0):
        raise ValueError("The predefined BAF sensitivity sources must use the full outer split")
    threshold = _finite(threshold, "Frozen operating threshold")
    if config.get("threshold_exact") is not None:
        recorded_threshold = _finite(config["threshold_exact"], "Saved exact threshold")
        precision = "full precision from config.threshold_exact"
    elif historical:
        recorded_threshold = _finite(saved["threshold"], "Historical saved threshold")
        precision = "historical saved six-decimal threshold; no finer value reconstructed"
    else:
        raise ValueError("A corrected sensitivity member requires an explicit full-precision DEV threshold")
    if threshold != recorded_threshold:
        raise ValueError("The comparison threshold differs from its declared saved precision")
    labels = np.load(run / "y_test.npy", allow_pickle=False)
    scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
    if (labels.ndim != 1 or scores.shape != labels.shape or set(np.unique(labels)) != {0, 1}
            or not np.isfinite(scores).all()):
        raise ValueError("Sensitivity points require finite aligned scores and both binary classes")
    if model != "ocsvm" and np.any((scores < 0) | (scores > 1)):
        raise ValueError("Supervised sensitivity scores must be probabilities in [0, 1]")
    index_path = run / "test_row_indices.npy"
    if index_path.exists():
        indices = np.load(index_path, allow_pickle=False)
        if (indices.shape != labels.shape or not np.issubdtype(indices.dtype, np.integer)
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("Sensitivity row indices are not unique aligned integer positions")
    elif historical:
        indices = None
    else:
        raise ValueError("Corrected sensitivity evidence requires original TEST row indices")
    predictions = scores >= threshold
    counts = {"TP": int(np.sum(predictions & (labels == 1))),
              "FP": int(np.sum(predictions & (labels == 0))),
              "TN": int(np.sum(~predictions & (labels == 0))),
              "FN": int(np.sum(~predictions & (labels == 1)))}
    if any(counts[name] != saved[name] for name in counts):
        raise ValueError("The saved operating threshold does not reproduce its TEST confusion counts")
    denominator = 5 * counts["TP"] + counts["FP"] + 4 * counts["FN"]
    points = {"average_precision": float(average_precision_score(labels, scores)),
              "F2": 5 * counts["TP"] / denominator if denominator else 0.0}
    for name, saved_name in (("average_precision", "PR-AUC"), ("F2", "F2")):
        if abs(points[name] - float(saved[saved_name])) > 1e-6:
            raise ValueError(f"Saved sensitivity point differs from its complete score evidence: {name}")
    return {"points": points, "threshold": threshold, "threshold_precision": precision,
            "source_run": str(run), "rows": len(labels), "fraud_rows": int(labels.sum()),
            "raw_file_sha256": config["dataset_hash_sha256"], "split_seed": config.get("split_seed"),
            "labels": labels, "indices": indices}


def _public_member(member: dict) -> dict:
    return {name: member[name] for name in ("points", "threshold", "threshold_precision", "source_run",
                                           "rows", "fraud_rows", "raw_file_sha256", "split_seed")}


def _normalise_standard_pair(record: dict, model: str, *, first_strategy="none",
                             second_strategy="none", missingness=False) -> dict:
    if record.get("bootstrap_requested") != BOOTSTRAP_REQUESTED or record.get("bootstrap_seed") != 42:
        raise ValueError("A predefined paired comparison requires 1000 requested bootstrap replicates and seed 42")
    if record.get("policy_selected_using_TEST") is not False:
        raise ValueError("A sensitivity policy cannot be selected using TEST")
    valid = _valid_replicates(record.get("bootstrap_valid"), "Paired comparison")
    first = _point_metrics(Path(record["first_run"]), model, first_strategy, record["first_threshold"])
    second = _point_metrics(Path(record["second_run"]), model, second_strategy, record["second_threshold"],
                            missing_policy="nan_indicators" if missingness else "preserve")
    if (not np.array_equal(first["labels"], second["labels"])
            or not np.array_equal(first["indices"], second["indices"])
            or first["raw_file_sha256"] != second["raw_file_sha256"]):
        raise ValueError("Both sensitivity members must describe the same original TEST observations")
    if record["rows"] != first["rows"] or record["fraud_rows"] != first["fraud_rows"]:
        raise ValueError("Sensitivity population counts differ from the saved ordered labels")
    differences, intervals = {}, {}
    for name in METRICS:
        for role, member in (("first", first), ("second", second)):
            declared = _finite(record[f"{role}_metrics"][name], f"{role} {name}", lower=0, upper=1)
            if not np.isclose(declared, member["points"][name], rtol=0, atol=1e-12):
                raise ValueError(f"Paired {role} point does not reproduce its saved TEST scores: {name}")
        difference = second["points"][name] - first["points"][name]
        declared = _finite(record["difference_second_minus_first"][name], f"Difference {name}", lower=-1, upper=1)
        if not np.isclose(declared, difference, rtol=0, atol=1e-12):
            raise ValueError(f"Paired difference has an incorrect value or direction: {name}")
        differences[name] = difference
        intervals[name] = _interval(record["paired_95_percentile_intervals"][name], name)
    return {"model": model, "first": _public_member(first), "second": _public_member(second),
            "difference_second_minus_first": differences, "paired_95_percentile_intervals": intervals,
            "bootstrap_requested": BOOTSTRAP_REQUESTED,
            "bootstrap_valid": {name: valid for name in METRICS}, "bootstrap_seed": 42}


def _normalise_catboost_control(report: dict) -> dict:
    """Respect the independently verified CatBoost comparison's distinct schema."""
    record = report["categorical_controls"]["catboost"]
    if record.get("bootstrap_iterations") != BOOTSTRAP_REQUESTED or record.get("bootstrap_seed") != 42:
        raise ValueError("The CatBoost paired control requires 1000 requested bootstrap replicates and seed 42")
    if (record.get("test_used_for_policy_or_hyperparameter_selection") is not False
            or record.get("test_labels_exactly_equal") is not True):
        raise ValueError("CatBoost control must preserve alignment and exclude TEST-dependent selection")
    first = _point_metrics(Path(record["source_run"]), "catboost", "smote", record["primary_threshold"], historical=True)
    second = _point_metrics(Path(report["catboost_control_source"]["run_dir"]), "catboost", "smotenc_control",
                            record["control_threshold_full_precision"])
    if (not np.array_equal(first["labels"], second["labels"])
            or first["raw_file_sha256"] != second["raw_file_sha256"]
            or first["raw_file_sha256"] != record["raw_file_sha256"]
            or array_sha256(second["indices"]) != record["shared_test_row_indices_sha256"]):
        raise ValueError("CatBoost categorical members do not reproduce the declared shared TEST alignment")
    if bool(first["indices"] is not None) != record["historical_test_indices_explicitly_saved"]:
        raise ValueError("CatBoost historical index precision is incorrectly described")
    if first["indices"] is not None and not np.array_equal(first["indices"], second["indices"]):
        raise ValueError("CatBoost historical and control TEST indices differ")
    if first["indices"] is None and (first["split_seed"] != 42 or second["split_seed"] != 42):
        raise ValueError("Historical CatBoost pairing without saved indices requires the documented seed-42 split")
    if first["threshold_precision"].startswith("historical") and "six-decimal" not in record["primary_threshold_precision"]:
        raise ValueError("A rounded historical CatBoost threshold cannot be described as full precision")
    differences, intervals, valid = {}, {}, {}
    for name, original_name in (("average_precision", "PR-AUC"), ("F2", "F2")):
        for role, member in (("primary", first), ("control", second)):
            if abs(float(record[f"{role}_metrics"][original_name]) - member["points"][name]) > 1e-6:
                raise ValueError(f"CatBoost {role} point differs from its saved scores: {name}")
        difference = second["points"][name] - first["points"][name]
        declared = _finite(record["difference_full_precision"][original_name], f"CatBoost difference {name}", lower=-1, upper=1)
        if not np.isclose(declared, difference, rtol=0, atol=1e-12):
            raise ValueError(f"CatBoost difference has an incorrect value or direction: {name}")
        evidence = record["paired_difference_bootstrap"][original_name]
        differences[name] = difference
        intervals[name] = _interval(evidence["ci_95_percent"], f"CatBoost {name}")
        valid[name] = _valid_replicates(evidence["valid_replicates"], f"CatBoost {name}")
    return {"model": "catboost", "first": _public_member(first), "second": _public_member(second),
            "difference_second_minus_first": differences, "paired_95_percentile_intervals": intervals,
            "bootstrap_requested": BOOTSTRAP_REQUESTED, "bootstrap_valid": valid, "bootstrap_seed": 42,
            "historical_primary_indices_explicit": first["indices"] is not None}


def normalise_report(report: dict) -> dict:
    if (report.get("status") != "complete" or set(report.get("missingness", {})) != set(MODELS)
            or set(report.get("categorical_controls", {})) != {"catboost", "fttransformer"}):
        raise ValueError("Both complete predefined sensitivity comparisons are required")
    missingness = {model: _normalise_standard_pair(report["missingness"][model], model, missingness=True)
                   for model in MODELS}
    categorical = {"catboost": _normalise_catboost_control(report),
                   "fttransformer": _normalise_standard_pair(report["categorical_controls"]["fttransformer"],
                                                             "fttransformer", first_strategy="smote",
                                                             second_strategy="smotenc_control")}
    return {"missingness": missingness, "categorical_controls": categorical}


def _signed(value: float) -> str:
    # Display rounding is explicit; a negative rounded zero is not a distinct effect.
    if abs(value) < .00005:
        value = 0.0
    return f"{value:+.4f}"


def _table(pairs: dict, models: tuple, *, caption: str, label: str,
           first_heading: str, second_heading: str) -> str:
    columns = "@{}" + "".join(f"p{{{width}mm}}" for width in COLUMN_WIDTHS_MM) + "@{}"
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begingroup\small", rf"\setlength{{\tabcolsep}}{{{TABCOLSEP_PT}pt}}",
             r"\renewcommand{\arraystretch}{1.12}", rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
             rf"Model & Metric & {first_heading} & {second_heading} & $\Delta$ [paired 95\% CI] \\", r"\midrule"]
    for position, model in enumerate(models):
        pair = pairs[model]
        for metric in METRICS:
            model_label = MODEL_LABELS[model] if metric == "average_precision" else ""
            metric_label = "AP" if metric == "average_precision" else r"$F_2$"
            lower, upper = pair["paired_95_percentile_intervals"][metric]
            difference = pair["difference_second_minus_first"][metric]
            lines.append(f"{model_label} & {metric_label} & {pair['first']['points'][metric]:.4f} & "
                         f"{pair['second']['points'][metric]:.4f} & "
                         rf"${_signed(difference)}\;[{_signed(lower)},\,{_signed(upper)}]$ \\")
        if position != len(models) - 1:
            lines.append(r"\addlinespace[3pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup", r"\end{table}", ""])
    return "\n".join(lines)


def make_fragment(normalised: dict) -> str:
    if TABLE_WIDTH_MM > THESIS_TEXT_WIDTH_MM:
        raise ValueError("The sensitivity table exceeds the dissertation's 150 mm text width")
    lines = [r"\paragraph{Predefined BAF sensitivity controls.}",
             (r"The following controls are separate from the primary factorial matrix. "
              r"They compare predefined alternatives on the same original BAF Base TEST "
              r"observations, with hyperparameters held fixed within each pair and each "
              r"member's operating threshold frozen from DEV. AP denotes average precision "
              r"(reported elsewhere as PR-AUC). Positive differences favour the second "
              r"member only for the displayed metric; no policy is selected using TEST."), "",
             _table(normalised["missingness"], MODELS,
                    caption=(r"Predefined BAF absence-code sensitivity: preserved codes versus NaN conversion "
                             r"with missingness indicators. Differences are NaN-and-indicators minus preserve; "
                             r"intervals come from 1,000 requested row-paired bootstrap resamples."),
                    label="tab:baf_missingness_sensitivity", first_heading="Preserve", second_heading="NaN + indicators"),
             _table(normalised["categorical_controls"], ("catboost", "fttransformer"),
                    caption=(r"Predefined BAF categorical-sampling controls: primary SMOTE versus SMOTE-NC. "
                             r"Differences are SMOTE-NC minus primary SMOTE; intervals come from 1,000 "
                             r"requested row-paired bootstrap resamples."),
                    label="tab:baf_smotenc_control", first_heading="SMOTE", second_heading="SMOTE-NC"),
             (r"The reported 95\% intervals describe TEST-row uncertainty conditional on "
              r"the two fitted models, their frozen thresholds and the shared split, not "
              r"training-seed variability. Single-class bootstrap resamples are excluded. "
              r"Points, differences and interval bounds are "
              r"rounded to four decimal places for display. An interval containing zero "
              r"does not establish equivalence, and these controls do not establish causality. "
              r"The missingness comparison changes conversion, imputation and indicators "
              r"jointly; the categorical control changes the sampling representation and "
              r"nominal-aware resampling, not an isolated architectural component. CatBoost "
              r"still receives one-hot encoded classifier inputs rather than native "
              r"categorical inputs."), ""]
    if normalised["categorical_controls"]["catboost"]["first"]["threshold_precision"].startswith("historical"):
        lines.extend([(r"For primary CatBoost SMOTE, the historical operating threshold is "
                       r"retained at its archived six-decimal precision, and the saved TEST "
                       r"confusion counts must be reproduced. A more precise historical "
                       r"threshold is not reconstructed or assumed."), ""])
    if not normalised["categorical_controls"]["catboost"]["historical_primary_indices_explicit"]:
        lines.extend([(r"The historical CatBoost primary did not save explicit TEST row "
                       r"indices. Its pairing uses the verified raw-file identity, the "
                       r"documented seed-42 split and exactly matched ordered TEST labels; "
                       r"no historical index artefact is invented."), ""])
    return "\n".join(lines)


def _source_snapshots(report: dict) -> dict:
    snapshots = {}

    def add(run, artefacts):
        run = Path(run)
        run = (run if run.is_absolute() else PROJECT_ROOT / run).resolve()
        for filename, digest in artefacts.items():
            if Path(filename).name != filename:
                raise ValueError("Sensitivity source hashes must identify local artefact filenames")
            path = (run / filename).resolve()
            if file_sha256(path) != digest:
                raise ValueError(f"A sensitivity source hash changed: {path}")
            snapshots[str(path)] = digest

    for section in (report["missingness"], {"fttransformer": report["categorical_controls"]["fttransformer"]}):
        for record in section.values():
            for role in ("first", "second"):
                add(record[f"{role}_run"], record[f"{role}_artefacts_sha256"])
    catboost = report["categorical_controls"]["catboost"]
    add(catboost["source_run"], catboost["source_artefacts_sha256"])
    control = report["catboost_control_source"]
    add(control["run_dir"], control["artefacts_sha256"])
    return snapshots


def generate(report_path: Path, primary_path: Path, sensitivity_path: Path, output_dir: Path) -> dict:
    """Write only a new verified fragment and QA record; never edit the thesis."""
    output_dir = validate_output_directory(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a new isolated sensitivity-text directory; existing files are not overwritten")
    report_path, primary_path, sensitivity_path = (Path(path).resolve() for path in
                                                  (report_path, primary_path, sensitivity_path))
    input_hashes = {str(path): file_sha256(path) for path in (report_path, primary_path, sensitivity_path)}
    report, primary, sensitivity = (_read_json(path) for path in (report_path, primary_path, sensitivity_path))
    from revision_paired_analysis import verify_paired_report
    if verify_paired_report(report, primary, sensitivity) is not True:
        raise ValueError("The complete paired report must pass its independent source verifier")
    sources = _source_snapshots(report)
    normalised = normalise_report(report)
    fragment = make_fragment(normalised)

    def verify_unchanged():
        for path, digest in {**input_hashes, **sources}.items():
            if file_sha256(Path(path)) != digest:
                raise ValueError(f"A sensitivity input changed during text verification: {path}")

    verify_unchanged()
    output_dir.mkdir(parents=True, exist_ok=False)
    fragment_path = output_dir / "sensitivity_tables.tex"
    with fragment_path.open("x", encoding="utf-8") as stream:
        stream.write(fragment)
    verify_unchanged()
    qa = {"status": "complete", "source_report_path": str(report_path),
          "source_report_sha256": input_hashes[str(report_path)],
          "primary_manifest_path": str(primary_path), "primary_manifest_sha256": input_hashes[str(primary_path)],
          "sensitivity_manifest_path": str(sensitivity_path),
          "sensitivity_manifest_sha256": input_hashes[str(sensitivity_path)],
          "source_artefacts_sha256": sources, "source_code_sha256": file_sha256(Path(__file__)),
          "paired_report_original_manifest_sha256": {
              "primary": report.get("source_manifest_sha256"),
              "sensitivity": report.get("sensitivity_manifest_sha256")},
          "manifest_verification_basis": "Current individual source pins; original whole-manifest hashes may predate legitimate manifest expansion",
          "bootstrap_requested_per_pair": BOOTSTRAP_REQUESTED, "difference_direction": "second minus first",
          "model_fitting": False, "threshold_selection_using_TEST": False,
          "policy_selection_using_TEST": False, "bootstrap_recomputed": False,
          "point_metrics_recomputed_from_saved_TEST_scores": True,
          "display_decimal_places": 4, "causal_or_equivalence_claim": False,
          "table_layout": {"column_widths_mm": list(COLUMN_WIDTHS_MM), "tabcolsep_pt": TABCOLSEP_PT,
                           "calculated_table_width_mm": TABLE_WIDTH_MM, "thesis_text_width_mm": THESIS_TEXT_WIDTH_MM},
          "tables": normalised, "fragment_path": str(fragment_path), "fragment_sha256": file_sha256(fragment_path)}
    with (output_dir / "sensitivity_text_qa.json").open("x", encoding="utf-8") as stream:
        json.dump(qa, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sensitivity-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.paired_report, args.manifest, args.sensitivity_manifest, args.output_dir)


if __name__ == "__main__":
    main()
