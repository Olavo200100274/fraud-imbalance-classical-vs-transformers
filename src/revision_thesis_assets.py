"""Prepare a mechanical LaTeX table patch from verified revision artefacts.

This is not a narrative editor. It preserves captions, labels and surrounding
prose and emits a patch for deliberate application after the complete evidence
grid has passed generation QA. Original result archives are never modified.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil

from experiment_protocol import file_sha256

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm")
MODEL_NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest",
               "lgbm": "LightGBM", "catboost": "CatBoost", "fttransformer": "FT-Transformer"}
PARAMETER_ORDER = {
    "logreg": ("C", "l1_ratio"),
    "rf": ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"),
    "lgbm": ("num_leaves", "learning_rate", "min_child_samples", "reg_alpha", "reg_lambda",
             "subsample", "colsample_bytree", "n_estimators"),
    "catboost": ("learning_rate", "depth", "l2_leaf_reg", "min_data_in_leaf", "iterations"),
    "fttransformer": ("d_token", "n_blocks", "attention_n_heads", "attention_dropout",
                      "ffn_d_hidden_factor", "ffn_dropout", "residual_dropout", "learning_rate",
                      "weight_decay", "batch_size"),
}


def table_map():
    mapping = {}
    for dataset in ("ulb", "baf"):
        for kind in ("baseline", "ops", "ci", "cost", "transformer_robustness"):
            mapping[f"tab:{kind}_{dataset}"] = f"tables/{dataset}/{kind}.tex"
        for metric in ("f2", "f1", "recall", "alert"):
            mapping[f"tab:threshold_{dataset}_{metric}"] = f"tables/{dataset}/threshold_{metric}.tex"
        for metric in ("prauc", "rocauc", "f2"):
            mapping[f"tab:factorial_{dataset}_{metric}"] = f"tables/{dataset}/factorial_{metric}.tex"
    for metric in ("prauc", "f2", "rocauc"):
        mapping[f"tab:crossdomain_{metric}"] = f"tables/baf/crossdomain_{metric}.tex"
    return mapping


TABULAR = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL)


def extract_tabular(text):
    matches = list(TABULAR.finditer(text))
    if len(matches) != 1:
        raise ValueError("A generated table must contain exactly one tabular environment.")
    return matches[0].group()


def replace_labelled_tabular(text, label, replacement):
    """Replace only the table immediately following a unique explicit label."""
    label_pattern = re.compile(r"\\label\{" + re.escape(label) + r"\}")
    labels = list(label_pattern.finditer(text))
    if len(labels) != 1:
        raise ValueError(f"Expected one explicit table label: {label}")
    start = labels[0].end()
    match = TABULAR.search(text, start)
    if match is None:
        raise ValueError(f"No tabular environment follows {label}")
    intervening = text[start:match.start()]
    if re.search(r"\\(?:label|section|chapter|(?:begin|end)\{(?:minipage|table\*?)\})", intervening):
        raise ValueError(f"A different document unit intervenes before the table for {label}")
    extract_tabular(replacement)
    return text[:match.start()] + replacement + text[match.end():]


def parameter_value(value):
    if value is None:
        return "None"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Unexpected selected hyperparameter type.")
    if isinstance(value, int):
        return str(value)
    return f"{value:.8g}"


def hyperparameter_tabular(manifest, dataset):
    lines = [r"\begin{tabular}{l l r}", r"\toprule",
             r"Model & Hyperparameter & Selected value \\", r"\midrule"]
    for model in MODELS[:-1]:
        reference = manifest["baseline_runs"][f"{dataset}/{model}"]
        run = Path(reference["run_dir"])
        run = run if run.is_absolute() else ROOT / run
        if file_sha256(run / "config.json") != reference["config_sha256"]:
            raise ValueError("A selected hyperparameter source has changed.")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        params = {name.removeprefix("classifier__"): value for name, value in config["best_params"].items()}
        allowed = PARAMETER_ORDER[model]
        if set(params) - set(allowed):
            raise ValueError(f"Review new hyperparameter fields before exporting {model}.")
        selected = [(name, params[name]) for name in allowed if name in params]
        if not selected:
            raise ValueError("A supervised baseline has no selected hyperparameters.")
        for number, (name, value) in enumerate(selected):
            heading = (rf"\multirow{{{len(selected)}}}{{*}}{{{MODEL_NAMES[model]}}}" if number == 0 else "")
            parameter = name.replace("_", r"\_")
            lines.append(heading + rf" & \texttt{{{parameter}}} & {parameter_value(value)} " + r"\\")
        lines.append(r"\midrule")
    lines.extend([r"OCSVM & \multicolumn{2}{l}{RBF; $\nu=0.01$; \texttt{gamma=scale}; no HPO} \\",
                  r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def verify_source_record(record):
    """Revalidate every numeric source file recorded by generation QA."""
    run = Path(record["run_dir"])
    for filename, field in (("config.json", "config_sha256"),
                            ("metrics_test.json", "metrics_sha256"),
                            ("metrics_cv.json", "metrics_cv_sha256"),
                            ("pr_curve_data.json", "pr_curve_data_sha256"),
                            ("y_test.npy", "y_test_sha256"),
                            ("y_test_scores.npy", "y_test_scores_sha256"),
                            ("test_row_indices.npy", "test_row_indices_sha256")):
        if field not in record:
            raise ValueError("A source record omits an explicit artefact hash field.")
        digest = record[field]
        target = run / filename
        if digest is None:
            if filename not in ("pr_curve_data.json", "test_row_indices.npy") or target.exists():
                raise ValueError("An existing or required source artefact is not explicitly hashed.")
        elif not digest or file_sha256(target) != digest:
            raise ValueError(f"A source artefact changed after generation: {target}")


def verify_transfer_record(record):
    """Verify all five cohorts and both retained and original prediction arrays."""
    if file_sha256(Path(record["path"])) != record["sha256"]:
        raise ValueError("A transfer source audit changed after generation.")
    variants = record.get("variants", {})
    if set(variants) != {f"baf_var{number}" for number in range(1, 6)}:
        raise ValueError("Transfer QA must identify all five Variant cohorts.")
    expected = {"y_test.npy", "y_test_scores.npy", "test_row_indices.npy",
                "original_y_test.npy", "original_y_test_scores.npy", "original_test_row_indices.npy"}
    for name, variant in variants.items():
        metrics = Path(variant["path"])
        if metrics.name != "metrics_test.json" or metrics.parent.name != name:
            raise ValueError("A Variant metric source does not identify its declared cohort.")
        if file_sha256(metrics) != variant["sha256"]:
            raise ValueError("Variant metrics changed after generation.")
        artefacts = variant.get("artefacts", {})
        if set(artefacts) != expected:
            raise ValueError("Transfer QA must hash all six retained/original cohort arrays.")
        for filename, evidence in artefacts.items():
            target = Path(evidence["path"])
            if target.resolve() != (metrics.parent / filename).resolve():
                raise ValueError("A transfer artefact path belongs to a different cohort.")
            if file_sha256(target) != evidence["sha256"]:
                raise ValueError(f"A transfer artefact changed after generation: {target}")


def verify_shap_records(qa):
    """Keep attribution matrices and their cohort evidence immutable at handoff."""
    sources = qa.get("compatible_shap_sources", {})
    if set(sources) != {"logreg", "lgbm", "catboost"}:
        raise ValueError("Generation QA must identify the three compatible SHAP sources.")
    expected = {"config.json", "model.joblib", "shap_values.npy", "shap_global.json",
                "shap_local_cases.json", "y_test.npy", "y_test_scores.npy"}
    evidence_records = []
    for source in sources.values():
        if set(source.get("artefacts", {})) != expected:
            raise ValueError("A compatible SHAP source omits required immutable artefacts.")
        for filename, evidence in source["artefacts"].items():
            if Path(evidence["path"]).resolve() != (Path(source["run_dir"]) / filename).resolve():
                raise ValueError("A compatible SHAP artefact belongs to a different run.")
            evidence_records.append(evidence)
    stability = qa.get("variant_shap_stability", {})
    arrays = {f"baf_var{number}/{filename}" for number in range(1, 6)
              for filename in ("shap_values.npy", "y_test.npy", "test_row_indices.npy")}
    if set(stability.get("artefacts", {})) != arrays or not stability.get("partition_manifest"):
        raise ValueError("Variant SHAP QA must identify its fifteen arrays and cohort manifest.")
    for name, evidence in stability["artefacts"].items():
        if Path(evidence["path"]).resolve() != (Path(stability["path"]).parent / name).resolve():
            raise ValueError("A Variant SHAP artefact belongs to a different cohort.")
        evidence_records.append(evidence)
    evidence_records.extend((stability, stability["partition_manifest"]))
    for evidence in evidence_records:
        if file_sha256(Path(evidence["path"])) != evidence["sha256"]:
            raise ValueError(f"A SHAP source artefact changed after generation: {evidence['path']}")


def verify_paired_record(record, manifest_path):
    """Reverify all fixed-policy comparison sources before thesis handoff."""
    from revision_paired_analysis import verify_paired_report
    root = manifest_path.resolve().parent
    paired_path = root / "derived/paired/paired_analysis.json"
    sensitivity_path = root / "sensitivity/revision_manifest.json"
    sensitivity = record.get("sensitivity_manifest", {})
    if (Path(record.get("path", "")).resolve() != paired_path
            or Path(sensitivity.get("path", "")).resolve() != sensitivity_path):
        raise ValueError("Paired QA paths must identify this revision's explicit comparison sources.")
    if (file_sha256(paired_path) != record.get("sha256")
            or file_sha256(sensitivity_path) != sensitivity.get("sha256")):
        raise ValueError("Paired evidence or its sensitivity manifest changed after generation.")
    report = json.loads(paired_path.read_text(encoding="utf-8"))
    if report.get("sensitivity_manifest_sha256") != sensitivity["sha256"]:
        raise ValueError("Paired analysis identifies a different sensitivity manifest.")
    verify_paired_report(report, json.loads(manifest_path.read_text(encoding="utf-8")),
                        json.loads(sensitivity_path.read_text(encoding="utf-8")))


def verify_bootstrap_compatibility_record(qa, manifest_path):
    """Revalidate the exact sidecar instead of trusting a whole-manifest hash."""
    from revision_bootstrap_compatibility import KEY, verify_bootstrap_compatibility
    source = qa.get("source_runs", {}).get(KEY)
    if not source:
        raise ValueError("Generation QA omits its required ULB FT baseline source.")
    run = Path(source["run_dir"])
    run = (run if run.is_absolute() else ROOT / run).resolve()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    recorded = qa.get("bootstrap_compatibility", {})
    if "bootstrap_iterations" in config:
        if config["bootstrap_iterations"] != 1000:
            raise ValueError("The ULB FT baseline has an explicitly incompatible bootstrap budget.")
        if recorded:
            raise ValueError("An explicit bootstrap budget must not use a missing-field compatibility exception.")
        return
    if not isinstance(recorded, dict) or set(recorded) != {KEY}:
        raise ValueError("Generation QA lacks the exact bootstrap compatibility sidecar reference.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded[KEY] != manifest.get("bootstrap_compatibility", {}).get(KEY):
        raise ValueError("Bootstrap compatibility QA differs from the current explicit manifest pin.")
    verified = verify_bootstrap_compatibility(manifest, run, config, expected_iterations=1000)
    if verified != recorded[KEY]:
        raise ValueError("Bootstrap compatibility evidence changed after generation.")


def normalise_output_hashes(outputs):
    """Accept legacy Windows separators without changing saved generation QA."""
    normalised = {}
    for name, digest in outputs.items():
        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("Generated output paths must be non-empty relative file names.")
        relative = PurePosixPath(name.replace("\\", "/"))
        windows = PureWindowsPath(relative.as_posix())
        if windows.drive or windows.root or relative.is_absolute() or ".." in relative.parts or str(relative) == ".":
            raise ValueError("Generated output paths must be relative and cannot escape their output root.")
        canonical = relative.as_posix()
        if canonical in normalised:
            raise ValueError(f"Generated output path aliases collide: {canonical}")
        normalised[canonical] = digest
    return normalised


def verify_generation(derived_root, manifest_path):
    qa = json.loads((derived_root / "generation_qa.json").read_text(encoding="utf-8"))
    if (qa.get("required_runs") != 72 or len(qa.get("source_runs", {})) != 72
            or len(qa.get("threshold_studies", {})) != 12 or len(qa.get("transfer_sources", {})) != 6):
        raise ValueError("The complete 72-run grid, 12 threshold studies and six transfer sources are required.")
    if Path(qa.get("manifest_path", "")).resolve() != manifest_path.resolve():
        raise ValueError("Generation QA refers to another source manifest.")
    recorded_manifest = qa.get("manifest_sha256")
    if not recorded_manifest or recorded_manifest != file_sha256(manifest_path):
        raise ValueError("The source manifest changed or lacks its generation-time hash.")
    verify_bootstrap_compatibility_record(qa, manifest_path)
    for section in ("source_runs", "threshold_studies", "transfer_sources"):
        for record in qa[section].values():
            if section == "source_runs":
                verify_source_record(record)
            elif section == "transfer_sources":
                verify_transfer_record(record)
            elif file_sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError("Derived source evidence changed after generation.")
    verify_shap_records(qa)
    verify_paired_record(qa.get("paired_analysis", {}), manifest_path)
    outputs = qa.get("output_sha256", {})
    if not outputs:
        raise ValueError("Generation QA must identify every generated output.")
    outputs = normalise_output_hashes(outputs)
    verified_paths = set()
    for name, digest in outputs.items():
        path = (derived_root / name).resolve()
        if path in verified_paths:
            raise ValueError(f"Generated output path aliases collide: {name}")
        if not path.is_relative_to(derived_root.resolve()) or file_sha256(path) != digest:
            raise ValueError("Generated output paths or hashes are invalid.")
        verified_paths.add(path)
    for name in table_map().values():
        if name not in outputs:
            raise ValueError(f"A required generated table is missing from QA: {name}")
    qa["output_sha256"] = outputs
    return qa


def whole_file_patch(path, previous, updated):
    if previous == updated:
        return ""
    relative = path.relative_to(ROOT).as_posix()
    return (f"*** Update File: {relative}\n@@\n" +
            "\n".join("-" + line for line in previous.splitlines()) + "\n" +
            "\n".join("+" + line for line in updated.splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-figures", action="store_true",
                        help="Copy only verified generated figures to active Overleaf assets; does not apply the table patch.")
    args = parser.parse_args()
    derived = args.derived_root.resolve()
    output = args.output_dir.resolve()
    revision = args.manifest.parent.resolve()
    if not derived.is_relative_to(revision) or not output.is_relative_to(revision):
        raise ValueError("Generation and handoff files must remain in the isolated revision root.")
    if output.exists():
        raise FileExistsError("Choose a new handoff directory; do not replace a previous audit.")
    qa = verify_generation(derived, args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results_path = ROOT / "Overleaf/Chapters/5-Results.tex"
    appendix_path = ROOT / "Overleaf/Chapters/7-appendices.tex"
    previous = results_path.read_text(encoding="utf-8")
    updated = previous
    for label, name in table_map().items():
        updated = replace_labelled_tabular(updated, label, extract_tabular((derived / name).read_text(encoding="utf-8")))
    old_appendix = appendix_path.read_text(encoding="utf-8")
    new_appendix = old_appendix
    for dataset, suffix in (("ulb_2013", "ulb"), ("baf_base", "baf")):
        new_appendix = replace_labelled_tabular(new_appendix, f"tab:hyperparams_{suffix}",
                                             hyperparameter_tabular(manifest, dataset))
    patch = ("*** Begin Patch\n" + whole_file_patch(results_path, previous, updated)
             + whole_file_patch(appendix_path, old_appendix, new_appendix) + "*** End Patch\n")
    output.mkdir(parents=True, exist_ok=False)
    patch_path = output / "thesis_tables.patch"
    patch_path.write_text(patch, encoding="utf-8")
    copies = {}
    if args.copy_figures:
        for name, digest in qa["output_sha256"].items():
            if not name.startswith("figures/") or Path(name).suffix.lower() not in (".pdf", ".png"):
                continue
            destination = ROOT / "Overleaf" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(derived / name, destination)
            if file_sha256(destination) != digest:
                raise ValueError("A copied figure does not match its verified source.")
            copies[name] = digest
    report = {"status": "mechanical_patch_prepared_not_applied",
              "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_manifest_sha256": file_sha256(args.manifest),
              "generation_qa_sha256": file_sha256(derived / "generation_qa.json"),
              "source_document_sha256": {str(path.relative_to(ROOT)): file_sha256(path)
                                         for path in (results_path, appendix_path)},
              "tables": table_map(), "patch_sha256": file_sha256(patch_path),
              "copied_figures_sha256": copies,
              "remaining_work": "Apply and review the patch; independently revise numerical prose/captions and scientific conclusions; compile and visually inspect the complete PDF."}
    (output / "handoff_preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"patch": str(patch_path), "tables": len(table_map()), "figure_copies": len(copies)}, indent=2))


if __name__ == "__main__":
    main()
