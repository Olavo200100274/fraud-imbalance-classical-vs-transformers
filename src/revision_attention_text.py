"""Prepare verified attention Appendix text from a pinned corrected baseline.

Only a new revision fragment and its QA record are written. This module never
edits the dissertation, changes predictions, or selects different cases. The
three nearest false positives and false negatives must reproduce the extraction
summary and the complete saved TEST evidence before LaTeX is generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from attention_analysis import classify_predictions, select_hard_cases
from generate_revision_interpretability import (ROOT, file_digest,
                                                validate_output_directory)


CASES = tuple(f"{kind}_{rank}" for kind in ("FP", "FN") for rank in (1, 2, 3))


def scientific_latex(value: float) -> str:
    """Display a non-negative distance without rounding borderline cases to zero."""
    if not np.isfinite(value) or value < 0:
        raise ValueError("A displayed distance must be finite and non-negative")
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"{float(mantissa):.3f}\times10^{{{int(exponent)}}}"


def verify_cases(summary: dict, labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Require the original full-precision error classes and nearest-case rule."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Attention case verification requires complete aligned TEST evidence")
    if float(summary["threshold"]) != threshold:
        raise ValueError("Attention summary and pinned baseline exact thresholds differ")
    if set(summary.get("individual_cases", {})) != set(CASES):
        raise ValueError("Exactly three false positives and three false negatives are required")
    classes, unused_predictions = classify_predictions(labels, scores, threshold)
    nearest = select_hard_cases(classes, scores, threshold, n=3)
    verified = {}
    for case in CASES:
        kind, rank = case.split("_")
        record = summary["individual_cases"][case]
        index = record["sample_idx"]
        if not isinstance(index, int) or not 0 <= index < len(labels):
            raise ValueError(f"Invalid TEST position for {case}")
        if len(nearest.get(kind, [])) != 3 or int(nearest[kind][int(rank) - 1]) != index:
            raise ValueError(f"{case} is not the declared nearest-threshold error case")
        if classes[index] != kind or int(labels[index]) != record["true_label"]:
            raise ValueError(f"Incorrect saved error class or label for {case}")
        if float(scores[index]) != float(record["score"]):
            raise ValueError(f"Displayed score differs from the original TEST score for {case}")
        distance = abs(float(scores[index]) - threshold)
        if not np.isclose(distance, record["distance_to_threshold"], rtol=0, atol=1e-15):
            raise ValueError(f"Incorrect threshold distance for {case}")
        if abs(float(record["attention_forward_score"]) - float(scores[index])) > 1e-5:
            raise ValueError(f"Attention forward does not reproduce the saved score for {case}")
        if bool(record["attention_forward_score"] >= threshold) != bool(scores[index] >= threshold):
            raise ValueError(f"Attention forward changes the threshold error class for {case}")
        verified[case] = {"test_position": index, "true_label": int(labels[index]),
                          "score": float(scores[index]), "threshold": threshold,
                          "distance_to_threshold": distance,
                          "distance_latex": scientific_latex(distance),
                          "bar_file": f"attention_bar_{case}.png"}
    return verified


def make_appendix_fragment(verified: dict, token_count: int) -> str:
    """Use one full-width case per figure, preserving the existing reference labels."""
    if set(verified) != set(CASES) or token_count < 2:
        raise ValueError("Six verified cases and a valid token count are required")
    threshold = verified["FP_1"]["threshold"]
    lines = [
        r"\section{FT-Transformer Attention Diagnostics --- Individual Case Bar Charts}",
        r"\label{appendixB}", "",
        (r"The following figures supplement Section~\ref{sec:attention_diagnostics}. "
         r"Each chart shows the final-layer, head-averaged \texttt{[CLS]} attention "
         r"distribution for one BAF Base TEST observation. Categorical feature tokens "
         r"are orange and numerical tokens are blue; \texttt{[CLS]} self-attention "
         r"is reported separately in the chart heading. The dotted vertical line "
         rf"marks the uniform reference $1/{token_count}$, and the dashed vertical "
         r"lines mark the numerical- and categorical-token group means. These "
         r"references are distinct quantities."), "",
        (rf"The frozen baseline threshold is $\tau={threshold:.6f}$ (display rounding only). "
         r"The three false positives and three false negatives nearest to its "
         r"full-precision value are selected separately. Captions report the "
         r"predicted fraud score, the original label and the absolute distance "
         r"$|p-\tau|$; the displayed six-decimal scores are not used to recompute "
         r"the error classes. These deliberately selected borderline examples "
         r"are not representative estimates of the entire error population."), "",
    ]
    for case in CASES:
        kind, rank = case.split("_")
        record = verified[case]
        label = "Legitimate" if record["true_label"] == 0 else "Fraud"
        long_class = "false-positive" if kind == "FP" else "false-negative"
        figure_label = (f"fig:attention_bar_{kind.lower()}" if rank == "1"
                        else f"fig:attention_bar_{kind.lower()}_{rank}")
        lines.extend([
            r"\clearpage", r"\begin{figure}[H]", r"\centering",
            rf"\includegraphics[width=\textwidth]{{figures/baf/{record['bar_file']}}}",
            (rf"\caption[Per-feature attention for {long_class} case {rank}]"
             rf"{{Final-layer, head-averaged \texttt{{[CLS]}} attention for {long_class} "
             rf"case {rank} (BAF Base): $p={record['score']:.6f}$, true label: {label}, "
             rf"$|p-\tau|={record['distance_latex']}$. Dotted line: uniform attention "
             rf"$1/{token_count}$; dashed lines: token-group means.}}"),
            rf"\label{{{figure_label}}}", r"\end{figure}", "",
        ])
    lines.extend([
        (r"Figures~\ref{fig:attention_bar_fp}--\ref{fig:attention_bar_fn_3} provide "
         r"descriptive case-level allocations only. Their appearance does not "
         r"establish causal feature importance, explanation faithfulness, or the "
         r"reason for an incorrect prediction. The diagnostic omits earlier "
         r"layers and individual attention heads."), "",
        r"\clearpage", r"\flushbottom", "",
    ])
    return "\n".join(lines)


def generate(manifest_path: Path, summary_path: Path, output_dir: Path) -> dict:
    """Verify current hashes and saved case membership before writing a fragment."""
    from experiment_protocol import PROTOCOL_VERSION
    manifest_digest = file_digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest.get("baseline_runs", {}).get("baf_base/fttransformer")
    if not isinstance(selected, dict) or not selected.get("config_sha256"):
        raise ValueError("A hash-pinned corrected BAF FT-Transformer baseline is required")
    run = Path(selected["run_dir"])
    run = run.resolve() if run.is_absolute() else (ROOT / run).resolve()
    if file_digest(run / "config.json") != selected["config_sha256"]:
        raise ValueError("The pinned baseline configuration changed")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if (config.get("protocol_version") != PROTOCOL_VERSION
            or json.loads((run / "completed.json").read_text(encoding="utf-8")).get("status") != "complete"):
        raise ValueError("A completed corrected-protocol baseline is required")
    if (config.get("dataset"), config.get("model"), config.get("strategy"), config.get("missing_policy")) != (
            "baf_base", "fttransformer", "none", "preserve"):
        raise ValueError("Primary attention cannot use a different model, strategy or missingness sensitivity")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if Path(summary["source_run"]).resolve() != run or summary.get("dataset") != "baf_base":
        raise ValueError("Attention summary does not belong to the pinned baseline")
    for name in ("config.json", "model.pt", "preprocessors.joblib", "y_test.npy", "y_test_scores.npy", "test_row_indices.npy"):
        if summary.get("source_sha256", {}).get(name) != file_digest(run / name):
            raise ValueError(f"Attention source hash differs: {name}")
    for name, expected_digest in summary.get("derived_artefacts_sha256", {}).items():
        if Path(name).name != name or file_digest(summary_path.parent / name) != expected_digest:
            raise ValueError(f"Attention derived artefact differs: {name}")
    score_verification = summary["score_verification"]
    if (score_verification["threshold_disagreements"] != 0
            or not 0 <= score_verification["maximum_absolute_error"] <= 1e-5):
        raise ValueError("The complete attention extraction did not reproduce TEST predictions")
    labels = np.load(run / "y_test.npy", allow_pickle=False)
    scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
    verified = verify_cases(summary, labels, scores, float(config["threshold_exact"]))
    for case, record in verified.items():
        if record["bar_file"] not in summary.get("derived_artefacts_sha256", {}):
            raise ValueError(f"The verified case chart is missing: {case}")
    token_count = 1 + summary["n_numerical_features"] + summary["n_categorical_features"]
    fragment = make_appendix_fragment(verified, token_count)
    if file_digest(manifest_path) != manifest_digest:
        raise ValueError("The source manifest changed during attention text verification")
    output_dir = validate_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fragment_path = output_dir / "attention_appendix.tex"
    if fragment_path.exists() or (output_dir / "attention_text_qa.json").exists():
        raise FileExistsError("Use a new text output directory; existing verified fragments are not overwritten")
    fragment_path.write_text(fragment, encoding="utf-8")
    report = {"status": "complete", "manifest_path": str(manifest_path.resolve()), "manifest_sha256": manifest_digest,
              "summary_path": str(summary_path.resolve()), "summary_sha256": file_digest(summary_path),
              "source_run": str(run), "threshold_exact": float(config["threshold_exact"]),
              "score_display_decimals": 6, "cases": verified,
              "all_selected_distances_below_0_001": all(record["distance_to_threshold"] < .001 for record in verified.values()),
              "causal_interpretation": False, "figure_layout": "One full-width case per figure",
              "fragment_path": str(fragment_path.resolve()), "fragment_sha256": file_digest(fragment_path),
              "source_code_sha256": file_digest(Path(__file__))}
    (output_dir / "attention_text_qa.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.manifest, args.summary, args.output_dir)


if __name__ == "__main__":
    main()
