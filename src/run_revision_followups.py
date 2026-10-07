"""Logged CPU/GPU follow-up queues for the authorised thesis correction.

Each queue waits for completed, full-data manifest pins before starting dependent
work. CPU and GPU sensitivity fits are separated so that two transformer fits
cannot inadvertently compete for the same device. Failures stop the queue and
retain every log; no failed experiment is promoted into the evidence base.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment_protocol import replace_with_retry, file_sha256
from revision_resources import exclusive_resource, process_identity, queue_resource_name, record_resource_child
from run_revision_training import CLASSICAL, ROOT, STRATEGIES, is_complete, read_manifest, _resolve_run


def write_state(path, state):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def wait_for_pins(manifest_path, keys, label):
    previous = None
    while True:
        manifest = read_manifest(manifest_path)
        completed = 0
        for dataset, model, strategy in keys:
            expected_trials = (50 if dataset == "ulb_2013" and strategy == "none" and model != "ocsvm"
                               else 0 if dataset == "baf_base" and strategy == "none" else None)
            fixed_source = (manifest.get("historical_baseline_runs", {}).get("baf_base", {}).get(model)
                            if dataset == "baf_base" and strategy == "none" else None)
            if strategy != "none":
                reference = manifest.get("baseline_runs", {}).get(f"{dataset}/{model}")
                if reference:
                    fixed_source = reference["run_dir"] if isinstance(reference, dict) else reference
            completed += is_complete(manifest, dataset, model, strategy, expected_n_trials=expected_trials,
                                     expected_fixed_params_run=fixed_source,
                                     expected_bootstrap_iterations=1000 if strategy == "none" else 0)
        if completed != previous:
            print(f"WAIT {label}: {completed}/{len(keys)} complete", flush=True)
            previous = completed
        if completed == len(keys):
            return
        time.sleep(20)


def step_complete(name, manifest_path, output_root):
    """Revalidate each dependency and derived output before a queue skips it."""
    manifest = read_manifest(manifest_path)
    historical = manifest.get("historical_baseline_runs", {}).get("baf_base", {})
    if name == "baf_validation_recovery":
        return all(is_complete(manifest, "baf_base", model, "none",
                               expected_n_trials=0, require_validation_recovery=True,
                               expected_fixed_params_run=historical[model], expected_bootstrap_iterations=1000)
                   for model in CLASSICAL)
    if name.startswith("baf_absence_sensitivity_"):
        models = CLASSICAL if name.endswith("classical") else ("fttransformer",)
        sensitivity = read_manifest(output_root / "sensitivity/revision_manifest.json")
        return all(is_complete(sensitivity, "baf_base", model, "none",
                               expected_missing_policy="nan_indicators", expected_n_trials=0,
                               expected_fixed_params_run=historical[model], expected_bootstrap_iterations=1000)
                   for model in models)
    if name in ("baf_catboost_smotenc", "baf_ft_smotenc"):
        model = "catboost" if name == "baf_catboost_smotenc" else "fttransformer"
        fixed_source = historical.get(model)
        if model == "fttransformer":
            reference = manifest.get("baseline_runs", {}).get("baf_base/fttransformer")
            if reference:
                fixed_source = reference["run_dir"] if isinstance(reference, dict) else reference
        if not is_complete(manifest, "baf_base", model, "smotenc_control",
                           expected_fixed_params_run=fixed_source,
                           expected_bootstrap_iterations=1000 if model == "catboost" else 0):
            return False
        if model == "catboost":
            reference = manifest["runs"]["baf_base/catboost/smotenc_control"]
            run = _resolve_run(reference["run_dir"])
            config = read_manifest(run / "config.json")
            completion_file = Path(config["audit_directory"]) / "completed_control.json"
            comparison_file = run / "primary_smote_comparison.json"
            if not completion_file.is_file() or not comparison_file.is_file():
                return False
            completion = read_manifest(completion_file)
            if (completion.get("status") != "complete" or _resolve_run(completion["run_dir"]) != run
                    or completion.get("config_sha256") != file_sha256(run / "config.json")
                    or completion.get("comparison_file_sha256") != file_sha256(comparison_file)):
                raise ValueError("The CatBoost control completion does not identify its immutable run/comparison.")
        return True
    if name == "baf_ft_attention":
        if not is_complete(manifest, "baf_base", "fttransformer", "none", expected_n_trials=0,
                           expected_fixed_params_run=historical["fttransformer"], expected_bootstrap_iterations=1000):
            return False
        directory = output_root / "derived/interpretability/attention"
        summary_file = directory / "attention_summary.json"
        if not summary_file.is_file():
            return False
        summary = read_manifest(summary_file)
        run = _resolve_run(manifest["baseline_runs"]["baf_base/fttransformer"]["run_dir"])
        if _resolve_run(summary.get("source_run", "")) != run or summary.get("dataset") != "baf_base":
            raise ValueError("The attention summary belongs to a different baseline or dataset.")
        expected_sources = ("config.json", "model.pt", "preprocessors.joblib", "y_test.npy",
                            "y_test_scores.npy", "test_row_indices.npy")
        for filename in expected_sources:
            if summary.get("source_sha256", {}).get(filename) != file_sha256(run / filename):
                raise ValueError("An attention source artefact differs from the summary's recorded hash.")
        derived = summary.get("derived_artefacts_sha256", {})
        if "attention_cls_weights.npy" not in derived or "attention_aggregate_cls.png" not in derived:
            return False
        for filename, digest in derived.items():
            if Path(filename).name != filename or not (directory / filename).is_file():
                return False
            if file_sha256(directory / filename) != digest:
                raise ValueError("A derived attention artefact differs from the recorded hash.")
        verification = summary.get("score_verification", {})
        if (verification.get("threshold_disagreements") != 0
                or not 0 <= verification.get("maximum_absolute_error", float("inf")) <= 1e-5):
            raise ValueError("The attention extraction did not reproduce the saved model scores/decisions.")
        return True
    raise ValueError(f"Unknown follow-up step: {name}")


def run_step(script, arguments, name, output_root, state_path, state, manifest_path):
    complete = step_complete(name, manifest_path, output_root)
    if name in state.get("completed_steps", []) and not complete:
        raise ValueError(f"Recorded step {name} has missing or invalid evidence; cannot silently skip or overwrite it.")
    if complete:
        if name not in state.setdefault("completed_steps", []):
            state["completed_steps"].append(name)
        state.update(status="validated", current_step=name)
        write_state(state_path, state)
        print(f"SKIP revalidated scientific evidence for {name}", flush=True)
        return
    log_dir = output_root / "logs"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"followup_{name}_{stamp}.log"
    command = [sys.executable, "-B", "-u", str(ROOT / "src/revision_runtime.py"),
               str(ROOT / "src" / script), *map(str, arguments)]
    state.update(status="running", current_step=name, current_log=str(log_path),
                 current_command=command)
    write_state(state_path, state)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT,
                                   env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
                                   stdout=log, stderr=subprocess.STDOUT)
        state["child_pid"] = process.pid
        state["child_creation_token"] = process_identity(process.pid)["creation_token"]
        if state.get("queue_resource_owner"):
            record_resource_child(state["queue_resource_owner"], process.pid)
        write_state(state_path, state)
        print(f"RUN {name}; PID={process.pid}; log={log_path}", flush=True)
        exit_code = process.wait()
        if state.get("queue_resource_owner"):
            record_resource_child(state["queue_resource_owner"])
    state.pop("child_pid", None)
    state.pop("child_creation_token", None)
    state["last_child_exit_code"] = exit_code
    if exit_code:
        state.update(status="failed", exit_code=exit_code)
        write_state(state_path, state)
        raise RuntimeError(f"Follow-up {name} failed ({exit_code}); inspect {log_path}")
    if not step_complete(name, manifest_path, output_root):
        state.update(status="failed", failure="Process exited without the required complete scientific evidence.")
        write_state(state_path, state)
        raise RuntimeError(f"Follow-up {name} exited without complete, validated artefacts; inspect {log_path}")
    state.setdefault("completed_steps", []).append(name)
    write_state(state_path, state)
    print(f"DONE {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "results_revision/20261005/revision_manifest.json")
    args = parser.parse_args()
    output_root = args.manifest.parent.resolve()
    resource = queue_resource_name(output_root, args.lane)
    with exclusive_resource(output_root, resource, f"{args.lane.upper()} follow-up queue", wait=False) as owner:
        try:
            _run_queue(args, output_root, owner)
        except Exception as error:
            state_path = output_root / f"followup_{args.lane}_state.json"
            state = read_manifest(state_path)
            state.update(status="failed", failure=str(error),
                         failed_at_utc=datetime.now(timezone.utc).isoformat())
            write_state(state_path, state)
            raise


def _run_queue(args, output_root, owner):
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    state_path = output_root / f"followup_{args.lane}_state.json"
    state = read_manifest(state_path)
    if state.get("child_pid"):
        child = process_identity(state["child_pid"])
        if child["status"] != "dead" and (state.get("child_creation_token") is None
                or child["creation_token"] == state["child_creation_token"]):
            raise RuntimeError("A previous queue child may still be alive; refuse to launch duplicate scientific work.")
    if state.get("source_manifest") not in (None, str(args.manifest.resolve())):
        raise ValueError("The saved queue state belongs to another source manifest.")
    state.pop("child_pid", None)
    state.pop("child_creation_token", None)
    state.update(lane=args.lane, status="waiting", source_manifest=str(args.manifest.resolve()),
                 queue_pid=owner["pid"], queue_creation_token=owner["creation_token"],
                 queue_resource_owner=owner,
                 started_at_utc=datetime.now(timezone.utc).isoformat())
    write_state(state_path, state)
    common = ["--manifest", args.manifest, "--results-root", output_root]
    if args.lane == "cpu":
        keys = [("ulb_2013", model, strategy) for model in CLASSICAL
                for strategy in (("none",) if model == "ocsvm" else STRATEGIES)]
        wait_for_pins(args.manifest, keys, "ULB classical primary matrix")
        run_step("run_revision_training.py", ["--lane", "baf-validation", *common],
                 "baf_validation_recovery", output_root, state_path, state, args.manifest)
        run_step("run_revision_training.py", ["--lane", "sensitivity", "--models", *CLASSICAL, *common],
                 "baf_absence_sensitivity_classical", output_root, state_path, state, args.manifest)
        run_step("revision_catboost_control.py", [*common, "--threads", "4", "--bootstrap-iterations", "1000"],
                 "baf_catboost_smotenc", output_root, state_path, state, args.manifest)
    else:
        keys = [(dataset, "fttransformer", strategy) for dataset in ("ulb_2013", "baf_base")
                for strategy in STRATEGIES]
        wait_for_pins(args.manifest, keys, "ULB and BAF transformer primary matrices")
        run_step("run_revision_training.py", ["--lane", "sensitivity", "--models", "fttransformer", *common],
                 "baf_absence_sensitivity_transformer", output_root, state_path, state, args.manifest)
        run_step("main_transformer.py", ["--dataset", "baf_base", "--strategy", "smotenc_control",
                 "--results-root", output_root, "--run-manifest", args.manifest,
                 "--bootstrap-iterations", "0"], "baf_ft_smotenc", output_root, state_path, state, args.manifest)
        run = read_manifest(args.manifest)["baseline_runs"]["baf_base/fttransformer"]["run_dir"]
        run_step("attention_analysis.py", ["--dataset", "baf_base", "--run-dir", run,
                 "--output-dir", output_root / "derived/interpretability/attention",
                 "--device", "cuda", "--threads", "2"], "baf_ft_attention", output_root, state_path, state, args.manifest)
    state.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    state.pop("child_pid", None)
    write_state(state_path, state)
    print(f"Follow-up {args.lane} queue complete", flush=True)


if __name__ == "__main__":
    main()
