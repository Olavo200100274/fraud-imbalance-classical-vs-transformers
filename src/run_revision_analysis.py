"""Dependency-gated saved-evidence analyses and isolated revision exports.

This queue never fits a predictive model, selects a policy using TEST, modifies
thesis sources, replaces active figures/tables, or chooses the newest result
directory. One explicitly labelled historical sampler-only reconstruction
resamples the verified frozen-preprocessor representation without retraining.
Every analysis is logged and revalidated on resume. Scientific failures stop
the queue; the final status requests human numeric and visual review.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment_protocol import PROTOCOL_VERSION, file_sha256, replace_with_retry
from revision_resources import (exclusive_resource, keep_system_awake, process_identity,
                                queue_resource_name, record_resource_child)
from run_revision_training import CLASSICAL, ROOT, STRATEGIES, is_complete, read_manifest, _resolve_run
from run_revision_followups import wait_for_pins, step_complete, write_state


MODELS = (*CLASSICAL, "fttransformer")
VARIANTS = ("baf_var1", "baf_var2", "baf_var3", "baf_var4", "baf_var5")


def wait_until(predicate, label, *, poll_seconds=20):
    """Wait for an existing producer without starting replacement work."""
    print(f"WAIT {label}", flush=True)
    while not predicate():
        time.sleep(poll_seconds)
    print(f"READY {label}", flush=True)


def partition_directory(manifest_path):
    reference = read_manifest(manifest_path).get("transfer_partitions")
    if not isinstance(reference, dict) or not reference.get("path") or not reference.get("sha256"):
        raise ValueError("The prepared transfer partition manifest must be explicitly pinned.")
    path = _resolve_run(reference["path"])
    if path.name != "partition_manifest.json" or file_sha256(path) != reference["sha256"]:
        raise ValueError("The prepared partition manifest differs from its explicit pin.")
    # The producer's verifier checks every saved cohort index/label/hash.
    from revision_transfer import load_partition_manifest
    load_partition_manifest(path.parent)
    return path.parent


def _verify_file(path, digest):
    if not Path(path).is_file() or file_sha256(path) != digest:
        raise ValueError(f"An analysis source or derived artefact changed: {path}")


def validate_diagnostics(manifest_path, root):
    path = root / "audit/lgbm_score_diagnostics.json"
    if not path.is_file():
        return False
    report = read_manifest(path)
    if report.get("status") != "complete" or report.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("The saved LGBM diagnostic is not complete under the current protocol.")
    manifest = read_manifest(manifest_path)
    if set(report.get("strategies", {})) != set(STRATEGIES) or report.get("fit_or_resampling_performed") is not False:
        raise ValueError("All seven saved-score diagnostic strategies are required, without new fitting.")
    for strategy, record in report["strategies"].items():
        reference = manifest["runs"][f"ulb_2013/lgbm/{strategy}"]
        run = _resolve_run(reference["run_dir"])
        if _resolve_run(record["source_run"]) != run:
            raise ValueError("The LGBM diagnostic uses a different strategy source.")
        _verify_file(run / "config.json", record["source_config_sha256"])
        _verify_file(run / "model.joblib", record["source_model_sha256"])
        for filename, field in (("y_test_scores.npy", "source_scores_sha256"),
                                ("y_test.npy", "source_labels_sha256"),
                                ("test_row_indices.npy", "source_indices_sha256")):
            _verify_file(run / filename, record[field])
        for artefact in record["artefacts"].values():
            _verify_file(artefact["path"], artefact["file_sha256"])
        for group in record["top_exact_score_groups"]:
            if "leaf_vectors_artefact" in group:
                artefact = group["leaf_vectors_artefact"]
                _verify_file(artefact["path"], artefact["file_sha256"])
    return True


def validate_thresholds(manifest_path, root, dataset):
    directory = root / "derived/thresholds" / dataset
    qa_path = directory / "threshold_qa.json"
    if not qa_path.is_file():
        return False
    qa = read_manifest(qa_path)
    if (qa.get("dataset") != dataset or set(qa.get("models", [])) != set(MODELS)
            or qa.get("new_fits") != 0 or qa.get("all_primary_thresholds_reproduced") is not True):
        raise ValueError("The threshold QA does not describe all six saved-validation studies.")
    manifest = read_manifest(manifest_path)
    for model in MODELS:
        path = directory / model / "threshold_study.json"
        if not path.is_file():
            return False
        study = read_manifest(path)
        run = _resolve_run(manifest["baseline_runs"][f"{dataset}/{model}"]["run_dir"])
        if _resolve_run(study["source_run"]) != run or study.get("dataset") != dataset:
            raise ValueError("The threshold study uses a different baseline source.")
        if set(study.get("test_results", {})) != {"fixed_05", "max_f1", "max_f2", "prec_ge_05"}:
            raise ValueError("The threshold study must contain exactly the four predefined rules.")
        for filename, digest in study["source_sha256"].items():
            _verify_file(run / filename, digest)
        if abs(study["thresholds_median"]["max_f2"] - read_manifest(run / "config.json")["threshold_exact"]) > 1e-12:
            raise ValueError("The threshold study changed its primary validation-selected threshold.")
    return True


def validate_transfer(manifest_path, root):
    directory = root / "derived/transfer"
    path = directory / "transfer_manifest.json"
    if not path.is_file():
        return False
    report = read_manifest(path)
    partitions = partition_directory(manifest_path)
    if (report.get("status") != "complete" or report.get("protocol_version") != PROTOCOL_VERSION
            or report.get("bootstrap_iterations") != 1000 or report.get("threshold_selection_on_variants") is not False
            or set(report.get("models", {})) != set(MODELS)
            or _resolve_run(report["partition_directory"]) != partitions
            or report.get("partition_manifest_sha256") != file_sha256(partitions / "partition_manifest.json")):
        raise ValueError("The completed transfer does not satisfy the explicit six-model/cohort protocol.")
    manifest = read_manifest(manifest_path)
    for model, source in report["models"].items():
        run = _resolve_run(manifest["baseline_runs"][f"baf_base/{model}"]["run_dir"])
        if _resolve_run(source["source_run"]) != run or set(source.get("variants", {})) != set(VARIANTS):
            raise ValueError("The completed transfer uses a different source or omits a Variant.")
        for filename, digest in source["source_artefacts_sha256"].items():
            _verify_file(run / filename, digest)
        if source["threshold_used"] != read_manifest(run / "config.json")["threshold_exact"]:
            raise ValueError("The transfer did not freeze the exact Base threshold.")
        for variant, reference in source["variants"].items():
            target = directory / model / variant
            _verify_file(target / "metrics_test.json", reference["metrics_test_sha256"])
            record = read_manifest(target / "metrics_test.json")
            for filename, evidence in record["artefacts"].items():
                _verify_file(target / filename, evidence["file_sha256"])
    return True


def validate_paired(manifest_path, root):
    path = root / "derived/paired/paired_analysis.json"
    if not path.is_file():
        return False
    report = read_manifest(path)
    if (report.get("status") != "complete" or set(report.get("missingness", {})) != set(MODELS)
            or set(report.get("categorical_controls", {})) != {"catboost", "fttransformer"}):
        raise ValueError("The paired report omits a predefined sensitivity/control comparison.")
    primary, sensitivity = read_manifest(manifest_path), read_manifest(root / "sensitivity/revision_manifest.json")
    from revision_paired_analysis import verify_paired_report
    verify_paired_report(report, primary, sensitivity)
    for model, record in report["missingness"].items():
        for role, manifest in (("first", primary), ("second", sensitivity)):
            run = _resolve_run(manifest["runs"][f"baf_base/{model}/none"]["run_dir"])
            if _resolve_run(record[f"{role}_run"]) != run:
                raise ValueError("A paired absence comparison uses a different fitted source.")
            _verify_file(run / "config.json", record[f"{role}_config_sha256"])
        if record.get("bootstrap_requested") != 1000 or not record.get("paired_95_percentile_intervals"):
            raise ValueError("The paired comparison lacks its prescribed 1,000 resamples/intervals.")
    ft = report["categorical_controls"]["fttransformer"]
    for role, strategy in (("first", "smote"), ("second", "smotenc_control")):
        run = _resolve_run(primary["runs"][f"baf_base/fttransformer/{strategy}"]["run_dir"])
        if _resolve_run(ft[f"{role}_run"]) != run:
            raise ValueError("The paired FT categorical comparison uses a different fitted source.")
        _verify_file(run / "config.json", ft[f"{role}_config_sha256"])
    if ft.get("bootstrap_requested") != 1000 or not ft.get("paired_95_percentile_intervals"):
        raise ValueError("The FT categorical comparison lacks its paired bootstrap evidence.")
    cb = report["categorical_controls"]["catboost"]
    control = _resolve_run(primary["runs"]["baf_base/catboost/smotenc_control"]["run_dir"])
    if cb != read_manifest(control / "primary_smote_comparison.json"):
        raise ValueError("The paired report differs from the independently verified CatBoost comparison.")
    source = _resolve_run(primary["historical_runs"]["baf_base/catboost/smote"]["run_dir"])
    if _resolve_run(cb["source_run"]) != source:
        raise ValueError("The CatBoost paired comparison uses a different primary SMOTE source.")
    for filename, digest in cb["source_artefacts_sha256"].items():
        _verify_file(source / filename, digest)
    if cb.get("bootstrap_iterations") != 1000 or not cb.get("paired_difference_bootstrap"):
        raise ValueError("The CatBoost categorical comparison lacks its paired bootstrap evidence.")
    return True


def validate_sampler_audit(manifest_path, root):
    path = root / "audit/sampler_identity.json"
    if not path.is_file():
        return False
    report = read_manifest(path)
    expected = {f"ulb_2013/{model}" for model in MODELS if model != "ocsvm"} | {"baf_base/fttransformer"}
    historical_models = set(CLASSICAL) - {"ocsvm"}
    if (report.get("status") != "complete" or report.get("protocol_version") != PROTOCOL_VERSION
            or report.get("model_fitting_performed") is not False
            or set(report.get("new_training_observations", {})) != expected
            or set(report.get("historical_saved_scores", {})) != historical_models):
        raise ValueError("The sampler audit omits a predefined observed or historical comparison.")
    manifest = read_manifest(manifest_path)
    for key, record in report["new_training_observations"].items():
        for field, strategy in (("smote_run", "smote"), ("tomek_run", "smote_tomek")):
            run = _resolve_run(manifest["runs"][f"{key}/{strategy}"]["run_dir"])
            if _resolve_run(record[field]) != run:
                raise ValueError("The sampler observations use a different fitted source.")
        for filename, digest in {**record["source_sha256"], **record["sampler_diagnostics_sha256"]}.items():
            _verify_file(filename, digest)
        model = key.split("/")[1]
        stages = ({"inner_training", "full_dev"} if model == "fttransformer"
                  else {"final_dev", *(f"fold_{fold}" for fold in range(1, 6))})
        if set(record["training_observations"]) != stages:
            raise ValueError("The observed sampler audit omits an actual training stage.")
    for model, record in report["historical_saved_scores"].items():
        for field, strategy in (("smote_run", "smote"), ("tomek_run", "smote_tomek")):
            run = _resolve_run(manifest["historical_runs"][f"baf_base/{model}/{strategy}"]["run_dir"])
            if _resolve_run(record[field]) != run:
                raise ValueError("The historical saved-score comparison uses a different source.")
        for filename, digest in record["source_sha256"].items():
            _verify_file(filename, digest)
    reconstruction = report["historical_BAF_reconstruction"]
    if (reconstruction.get("historical_fold_sampler_records_available") is not False
            or reconstruction.get("raw_file_sha256") != manifest["datasets"]["baf_base"]["raw_file_sha256"]
            or set(reconstruction.get("preprocessor_sources", {})) != historical_models
            or set(reconstruction.get("observations", {})) != {"reconstructed_final_dev"}):
        raise ValueError("The historical sampler reconstruction misidentifies its scope or sources.")
    representations = set()
    for model, reference in reconstruction["preprocessor_sources"].items():
        run = _resolve_run(manifest["historical_runs"][f"baf_base/{model}/none"]["run_dir"])
        if _resolve_run(reference["source_run"]) != run:
            raise ValueError("A reconstructed historical preprocessor belongs to another source.")
        _verify_file(run / "config.json", reference["config_sha256"])
        _verify_file(run / "model.joblib", reference["model_sha256"])
        representations.add(reference["transformed_DEV_sha256"])
    if len(representations) != 1:
        raise ValueError("The historical final-DEV representations were not exactly shared.")
    return True


def variant_shap_ready(manifest_path, root):
    path = root / "derived/interpretability/variant_stability/shap_variant_stability.json"
    if not path.is_file():
        return False
    report = read_manifest(path)
    if report.get("status") != "complete":
        return False  # An existing scientific producer is still running.
    source = _resolve_run(read_manifest(manifest_path)["interpretability"]["baseline_runs"]["lgbm"])
    if (_resolve_run(report["source_run"]) != source or report.get("model") != "lgbm"
            or report.get("model_fitting") is not False or report.get("variant_threshold_selection") is not False
            or set(report.get("variant_keys", [])) != {"baf_base", *VARIANTS}
            or set(report.get("variants", {})) != set(VARIANTS)):
        raise ValueError("The Variant SHAP marker does not identify all predefined frozen-model populations.")
    for filename, digest in report["source_artefacts_sha256"].items():
        _verify_file(source / filename, digest)
    partitions = partition_directory(manifest_path)
    if (_resolve_run(report["partition_directory"]) != partitions
            or report["partition_manifest_sha256"] != file_sha256(partitions / "partition_manifest.json")):
        raise ValueError("The Variant SHAP and transfer partition evidence do not align.")
    for variant in VARIANTS:
        for filename, digest in report["variants"][variant]["artefacts_sha256"].items():
            _verify_file(path.parent / variant / filename, digest)
    return True


def pin_variant_shap(manifest_path, root):
    """Pin the completed existing producer before freezing reporting sources."""
    path = root / "derived/interpretability/variant_stability/shap_variant_stability.json"
    if not variant_shap_ready(manifest_path, root):
        raise ValueError("Cannot pin incomplete Variant SHAP evidence.")
    lock = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    deadline = time.monotonic() + 30
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError("The source manifest is locked by another producer.")
            time.sleep(0.1)
    temporary = manifest_path.with_name(manifest_path.name + f".analysis_{os.getpid()}.tmp")
    try:
        manifest = read_manifest(manifest_path)
        interpretation = manifest.setdefault("interpretability", {})
        existing = interpretation.get("variant_stability")
        if existing and _resolve_run(existing) != path.resolve():
            raise ValueError("An existing Variant SHAP pin identifies another evidence source.")
        interpretation["variant_stability"] = str(path.resolve())
        interpretation["variant_stability_sha256"] = file_sha256(path)
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        replace_with_retry(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
        lock.unlink()


def primary_grid_ready(manifest_path):
    manifest = read_manifest(manifest_path)
    for dataset in ("ulb_2013", "baf_base"):
        for model in MODELS:
            for strategy in (("none",) if model == "ocsvm" else STRATEGIES):
                key = f"{dataset}/{model}/{strategy}"
                if key in manifest.get("runs", {}):
                    trials = (50 if dataset == "ulb_2013" and strategy == "none" and model != "ocsvm"
                              else 0 if dataset == "baf_base" and strategy == "none" else None)
                    fixed_source = None
                    if strategy != "none":
                        baseline_reference = manifest.get("baseline_runs", {}).get(f"{dataset}/{model}")
                        if not baseline_reference:
                            return False
                        fixed_source = (baseline_reference["run_dir"] if isinstance(baseline_reference, dict)
                                        else baseline_reference)
                    elif dataset == "baf_base":
                        fixed_source = manifest["historical_baseline_runs"]["baf_base"][model]
                    if not is_complete(manifest, dataset, model, strategy, expected_n_trials=trials,
                                       expected_fixed_params_run=fixed_source,
                                       expected_bootstrap_iterations=1000 if strategy == "none" else 0):
                        return False
                elif dataset == "baf_base" and model != "fttransformer" and strategy != "none":
                    reference = manifest.get("historical_runs", {}).get(key)
                    if not isinstance(reference, dict) or not reference.get("config_sha256"):
                        return False
                    run = _resolve_run(reference["run_dir"])
                    _verify_file(run / "config.json", reference["config_sha256"])
                    config = read_manifest(run / "config.json")
                    if (config.get("dataset"), config.get("model"), config.get("strategy")) != (dataset, model, strategy):
                        raise ValueError("A preserved factorial source has the wrong identity.")
                    if not all((run / filename).is_file() for filename in ("model.joblib", "metrics_test.json",
                                                                          "metrics_cv.json", "y_test.npy", "y_test_scores.npy")):
                        return False
                else:
                    return False
    return True


def validate_generation(manifest_path, root):
    path = root / "derived/reporting/generation_qa.json"
    if not path.is_file():
        return False
    report = read_manifest(path)
    if report.get("required_runs") != 72 or report.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("The generated artefacts do not identify the frozen complete 72-source evidence base.")
    for filename, digest in report.get("output_sha256", {}).items():
        target = (path.parent / filename).resolve()
        if path.parent.resolve() not in target.parents:
            raise ValueError("A generated artefact path escapes its isolated output root.")
        _verify_file(target, digest)
    if not report.get("output_sha256"):
        raise ValueError("The generation QA does not hash any produced table or figure.")
    return True


def validate_attention_text(manifest_path, root):
    directory = root / "derived/interpretability/attention_text"
    path = directory / "attention_text_qa.json"
    if not path.is_file():
        return False
    report = read_manifest(path)
    summary = root / "derived/interpretability/attention/attention_summary.json"
    run = _resolve_run(read_manifest(manifest_path)["baseline_runs"]["baf_base/fttransformer"]["run_dir"])
    if (report.get("status") != "complete" or _resolve_run(report["source_run"]) != run
            or Path(report["manifest_path"]).resolve() != manifest_path.resolve()
            or report.get("manifest_sha256") != file_sha256(manifest_path)
            or Path(report["summary_path"]).resolve() != summary.resolve()
            or report.get("causal_interpretation") is not False
            or set(report.get("cases", {})) != {f"{kind}_{rank}" for kind in ("FP", "FN") for rank in (1, 2, 3)}):
        raise ValueError("The attention Appendix fragment does not identify the frozen six-case evidence.")
    fragment = directory / "attention_appendix.tex"
    if Path(report["fragment_path"]).resolve() != fragment.resolve():
        raise ValueError("The attention text fragment is outside its isolated destination.")
    _verify_file(summary, report["summary_sha256"])
    _verify_file(fragment, report["fragment_sha256"])
    if report["threshold_exact"] != read_manifest(run / "config.json")["threshold_exact"]:
        raise ValueError("The attention text changed the exact frozen baseline threshold.")
    return True


def run_analysis_step(script, arguments, name, validator, root, state_path, state):
    complete = validator()
    if name in state.get("completed_steps", []) and not complete:
        raise ValueError(f"Recorded analysis {name} lost its evidence; do not silently skip or overwrite it.")
    if complete:
        if name not in state.setdefault("completed_steps", []):
            state["completed_steps"].append(name)
        write_state(state_path, state)
        print(f"SKIP revalidated {name}", flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    log_path = root / "logs" / f"analysis_{name}_{stamp}.log"
    command = [sys.executable, "-B", "-u", str(ROOT / "src/revision_runtime.py"),
               str(ROOT / "src" / script), *map(str, arguments)]
    state.update(status="running", current_step=name, current_log=str(log_path), current_command=command)
    write_state(state_path, state)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT,
                                   env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
                                   stdout=log, stderr=subprocess.STDOUT)
        state.update(child_pid=process.pid, child_creation_token=process_identity(process.pid)["creation_token"])
        record_resource_child(state["queue_resource_owner"], process.pid)
        write_state(state_path, state)
        print(f"RUN {name}; PID={process.pid}; log={log_path}", flush=True)
        exit_code = process.wait()
        record_resource_child(state["queue_resource_owner"])
    state.pop("child_pid", None)
    state.pop("child_creation_token", None)
    state["last_child_exit_code"] = exit_code
    if exit_code or not validator():
        state.update(status="failed", failure=f"{name}: exit={exit_code} or missing/invalid evidence")
        write_state(state_path, state)
        raise RuntimeError(f"Analysis {name} failed; inspect {log_path}")
    state.setdefault("completed_steps", []).append(name)
    write_state(state_path, state)


def transfer_step_arguments(manifest_path, root, partitions, state):
    """Retain the ordinary path unless a hash-pinned operator recovery exists."""
    arguments = ["--manifest", manifest_path, "--models", "all", "--partitions-dir", partitions,
                 "--output-dir", root / "derived/transfer", "--bootstrap-iterations", "1000"]
    recovery = state.get("explicit_transfer_recovery")
    if recovery is None:
        return [*arguments, "--device", "cpu"]
    reference = recovery.get("verification_report", {})
    report_path = root / "audit/transfer_cpu_cuda_20261007/ft_base_verification.json"
    if (recovery.get("automatic_retry") is not False or recovery.get("device") != "cuda"
            or recovery.get("batch_size") != 2048
            or Path(recovery.get("partial_directory", "")).resolve() != (root / "derived/transfer").resolve()
            or Path(reference.get("path", "")).resolve() != report_path.resolve()
            or recovery.get("transfer_code_sha256") != file_sha256(ROOT / "src/revision_transfer.py")):
        raise ValueError("The explicit transfer recovery does not match its verified inference configuration.")
    _verify_file(report_path, reference.get("sha256"))
    if (root / "derived/transfer/transfer_manifest.json").is_file():
        if not validate_transfer(manifest_path, root):
            raise ValueError("A recorded completed transfer failed revalidation.")
        return [*arguments, "--device", "cuda", "--batch-size", "2048"]
    from revision_transfer import verify_partial_transfer
    verified = verify_partial_transfer(root / "derived/transfer", manifest_path, partitions)
    if verified["reuse_provenance"] != recovery.get("verified_reuse"):
        raise ValueError("The approved partial transfer provenance changed before dispatch.")
    return [*arguments, "--device", "cuda", "--batch-size", "2048",
            "--resume-verified-partial", root / "derived/transfer"]


def execute_queue(args, owner):
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    state_path = root / "analysis_state.json"
    state = read_manifest(state_path)
    if state.get("source_manifest") and Path(state["source_manifest"]).resolve() != manifest_path:
        raise ValueError("An analysis queue cannot resume using another source manifest.")
    if state.get("child_pid"):
        child = process_identity(state["child_pid"])
        if child["status"] != "dead" and (state.get("child_creation_token") is None
                                          or child["creation_token"] == state["child_creation_token"]):
            raise RuntimeError("A previous analysis child may still be running; do not duplicate it.")
    state.pop("child_pid", None)
    state.pop("child_creation_token", None)
    state.update(status="waiting", source_manifest=str(manifest_path), queue_pid=owner["pid"],
                 queue_creation_token=owner["creation_token"], queue_resource_owner=owner,
                 started_at_utc=datetime.now(timezone.utc).isoformat())
    (root / "logs").mkdir(parents=True, exist_ok=True)
    write_state(state_path, state)
    wait_for_pins(manifest_path, [("ulb_2013", "lgbm", strategy) for strategy in STRATEGIES], "seven corrected LGBM configurations")
    run_analysis_step("revision_lgbm_diagnostics.py", ["--manifest", manifest_path, "--results-root", root, "--threads", "2"],
                      "lgbm_diagnostics", lambda: validate_diagnostics(manifest_path, root), root, state_path, state)
    baseline_keys = [(dataset, model, "none") for dataset in ("ulb_2013", "baf_base") for model in MODELS]
    wait_for_pins(manifest_path, baseline_keys, "twelve corrected primary baselines")
    for dataset in ("ulb_2013", "baf_base"):
        run_analysis_step("revision_thresholds.py", ["--manifest", manifest_path, "--dataset", dataset,
                          "--output-dir", root / "derived/thresholds" / dataset], f"thresholds_{dataset}",
                          lambda dataset=dataset: validate_thresholds(manifest_path, root, dataset), root, state_path, state)
    partitions = partition_directory(manifest_path)
    run_analysis_step("revision_transfer.py", transfer_step_arguments(manifest_path, root, partitions, state), "transfer_six_models",
                      lambda: validate_transfer(manifest_path, root), root, state_path, state)
    wait_until(lambda: (step_complete("baf_absence_sensitivity_classical", manifest_path, root)
                        and step_complete("baf_absence_sensitivity_transformer", manifest_path, root)
                        and step_complete("baf_catboost_smotenc", manifest_path, root)
                        and step_complete("baf_ft_smotenc", manifest_path, root)), "six absence sensitivities and two categorical controls")
    run_analysis_step("revision_paired_analysis.py", ["--manifest", manifest_path,
                      "--sensitivity-manifest", root / "sensitivity/revision_manifest.json",
                      "--output-dir", root / "derived/paired", "--bootstrap-iterations", "1000"], "paired_sensitivities",
                      lambda: validate_paired(manifest_path, root), root, state_path, state)
    wait_until(lambda: primary_grid_ready(manifest_path), "complete 72-source primary reporting grid")
    run_analysis_step("revision_sampler_audit.py", ["--manifest", manifest_path,
                      "--output", root / "audit/sampler_identity.json"], "sampler_identity",
                      lambda: validate_sampler_audit(manifest_path, root), root, state_path, state)
    wait_until(lambda: variant_shap_ready(manifest_path, root), "existing Variant SHAP producer; never launch a replacement")
    wait_until(lambda: step_complete("baf_ft_attention", manifest_path, root), "completed corrected attention analysis")
    wait_until(lambda: all(read_manifest(root / f"followup_{lane}_state.json").get("status") == "complete"
                          for lane in ("cpu", "gpu")), "all scientific follow-up writers finished")
    pin_variant_shap(manifest_path, root)
    state["frozen_manifest_sha256"] = file_sha256(manifest_path)
    write_state(state_path, state)
    run_analysis_step("revision_attention_text.py", ["--manifest", manifest_path,
                      "--summary", root / "derived/interpretability/attention/attention_summary.json",
                      "--output-dir", root / "derived/interpretability/attention_text"], "verified_attention_text",
                      lambda: validate_attention_text(manifest_path, root), root, state_path, state)
    run_analysis_step("generate_results.py", ["--manifest", manifest_path, "--results-root", root,
                      "--output-root", root / "derived/reporting",
                      "--threshold-study-root", root / "derived/thresholds",
                      "--cross-domain-root", root / "derived/transfer"], "isolated_reporting_export",
                      lambda: validate_generation(manifest_path, root), root, state_path, state)
    if file_sha256(manifest_path) != state["frozen_manifest_sha256"]:
        raise ValueError("The source manifest changed during final export; do not promote these outputs.")
    # Only queue state is updated here. The frozen source manifest is NOT edited.
    state.update(status="analysis_ready_for_human_review", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 next_action="Human numeric and visual review before applying any generated thesis assets.")
    write_state(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results_revision/20261005/revision_manifest.json")
    args = parser.parse_args()
    root = args.manifest.resolve().parent
    with keep_system_awake(), exclusive_resource(root, queue_resource_name(root, "analysis"),
                                                 "Saved-evidence analysis queue", wait=False) as owner:
        try:
            execute_queue(args, owner)
        except Exception as error:
            path = root / "analysis_state.json"
            state = read_manifest(path)
            state.update(status="failed", failure=str(error), failed_at_utc=datetime.now(timezone.utc).isoformat())
            write_state(path, state)
            raise


if __name__ == "__main__":
    main()
