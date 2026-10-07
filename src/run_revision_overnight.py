"""Adopt or resume the revision queues without an active AI conversation.

This supervisor does not fit models itself, stop live processes, retry failed
jobs automatically, edit thesis sources, or promote generated assets. The
existing queues retain their scientific validation and resource locks.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment_protocol import PROTOCOL_VERSION, file_sha256
from revision_resources import exclusive_resource, keep_system_awake, process_identity, queue_resource_name
from run_revision_training import ROOT, STRATEGIES, CLASSICAL, is_complete, read_manifest
from run_revision_followups import step_complete, write_state


QUEUE_NAMES = ("classical", "transformer", "followup_cpu", "followup_gpu", "analysis")
BOOTSTRAP_METADATA_FAILURE = "The completed run uses a different bootstrap budget."
ATTENTION_DRIFT_FAILURE = ("Attention forward and saved TEST predictions differ: "
                           "max error=3.016e-05, threshold decisions changed=0")
TRANSFER_CPU_FAILURE = "Frozen Base TEST predictions do not reproduce the source's saved score arrays."
REPORTING_PATH_FAILURE = "A required generated table is missing from QA: tables/ulb/baseline.tex"
REPORTING_QA_SHA256 = "7bafe378ab0c2842b03ad408c285e9e8f2e9d09779038176ccd3947227b66f42"
REPORTING_MANIFEST_SHA256 = "dbdf3c9d97ce51c219b26573e62ff8739fd460ed69c4faec5bacd45ca532e216"
REPORTING_COMPLETED_STEPS = ("lgbm_diagnostics", "thresholds_ulb_2013", "thresholds_baf_base",
                             "transfer_six_models", "paired_sensitivities", "sampler_identity",
                             "verified_attention_text", "isolated_reporting_export")


def identity_matches(reference):
    """A reused PID must never be mistaken for the saved queue."""
    if not reference or not reference.get("pid") or not reference.get("creation_token"):
        return False
    current = process_identity(int(reference["pid"]))
    if current["status"] == "unknown":
        raise RuntimeError(f"Cannot verify queue PID {reference['pid']}; refuse duplicate work")
    return current["status"] == "alive" and str(current["creation_token"]) == str(reference["creation_token"])


def primary_complete(manifest_path, lane):
    manifest = read_manifest(manifest_path)
    datasets = ("ulb_2013",) if lane == "classical" else ("ulb_2013", "baf_base")
    models = CLASSICAL if lane == "classical" else ("fttransformer",)
    for dataset in datasets:
        for model in models:
            for strategy in (("none",) if model == "ocsvm" else STRATEGIES):
                fixed = None
                if strategy != "none":
                    reference = manifest.get("baseline_runs", {}).get(f"{dataset}/{model}")
                    if not reference:
                        return False
                    fixed = reference["run_dir"] if isinstance(reference, dict) else reference
                elif dataset == "baf_base":
                    fixed = manifest["historical_baseline_runs"][dataset][model]
                trials = (50 if dataset == "ulb_2013" and strategy == "none" and model != "ocsvm"
                          else 0 if dataset == "baf_base" and strategy == "none" else None)
                if not is_complete(manifest, dataset, model, strategy, expected_n_trials=trials,
                                   expected_fixed_params_run=fixed,
                                   expected_bootstrap_iterations=1000 if strategy == "none" else 0):
                    return False
    return True


def queue_state_path(root, name):
    return root / ("analysis_state.json" if name == "analysis" else f"{name}_state.json")


def queue_complete(manifest_path, name):
    root = manifest_path.parent
    if name in ("classical", "transformer"):
        return primary_complete(manifest_path, name)
    state = read_manifest(queue_state_path(root, name))
    if name == "analysis":
        if state.get("status") != "analysis_ready_for_human_review":
            return False
        from revision_thesis_assets import verify_generation
        verify_generation(root / "derived/reporting", manifest_path)
        return True
    if state.get("status") != "complete":
        return False
    steps = (("baf_validation_recovery", "baf_absence_sensitivity_classical", "baf_catboost_smotenc")
             if name == "followup_cpu" else
             ("baf_absence_sensitivity_transformer", "baf_ft_smotenc", "baf_ft_attention"))
    return all(step_complete(step, manifest_path, root) for step in steps)


def command_for(name, manifest_path):
    common = ["--manifest", str(manifest_path)]
    if name in ("classical", "transformer"):
        script, arguments = "run_revision_training.py", ["--lane", name, *common,
                                                         "--results-root", str(manifest_path.parent)]
        checkpoint = manifest_path.parent / "tuning_cache/ulb_2013_lgbm_67407c24567dd09e.sqlite3"
        if name == "classical" and checkpoint.is_file():
            arguments += ["--resume-lgbm-study", str(checkpoint)]
    elif name.startswith("followup_"):
        script, arguments = "run_revision_followups.py", ["--lane", name.split("_")[1], *common]
    else:
        script, arguments = "run_revision_analysis.py", common
    return [sys.executable, "-B", "-X", "utf8", "-u", str(ROOT / "src/revision_runtime.py"),
            str(ROOT / "src" / script), *arguments]


def reject_orphan_children(reference, queue_state):
    child = queue_state.get("child_pid")
    if child:
        candidate = {"pid": child, "creation_token": queue_state.get("child_creation_token")}
        if not candidate["creation_token"] or identity_matches(candidate):
            raise RuntimeError(f"Previous scientific child PID {child} may still be running; no replacement launched")
    # Older primary runners did not record their child in a queue-state file.
    # Native read-only process inventory prevents resuming over such an orphan.
    if reference and os.name == "nt":
        parent = int(reference["pid"])
        query = (f"@(Get-CimInstance Win32_Process -Filter 'ParentProcessId = {parent}' | "
                 "Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress")
        result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", query],
                                capture_output=True, text=True, check=True, timeout=30)
        children = json.loads(result.stdout.strip() or "[]")
        if children:
            raise RuntimeError(f"Previous queue PID {parent} still has children {children}; no replacement launched")


def attach_or_start(name, manifest_path, references, previous, logs):
    root = manifest_path.parent
    saved = read_manifest(queue_state_path(root, name)) if name not in ("classical", "transformer") else {}
    candidates = [previous.get(name), references.get(name)]
    if saved.get("queue_pid"):
        candidates.insert(0, {"pid": saved["queue_pid"], "creation_token": str(saved.get("queue_creation_token"))})
    for reference in candidates:
        if identity_matches(reference):
            print(f"ADOPT {name}: PID={reference['pid']}; existing work left running", flush=True)
            return {**reference, "adopted": True}, None
    if queue_complete(manifest_path, name):
        print(f"SKIP {name}: complete evidence revalidated", flush=True)
        return {"status": "complete"}, None
    if saved.get("status") == "failed":
        raise RuntimeError(f"{name} previously failed; inspect {queue_state_path(root, name)} before an explicit recovery")
    reject_orphan_children(next((reference for reference in candidates if reference), None), saved)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = logs / f"overnight_{name}_{stamp}.log"
    with path.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command_for(name, manifest_path), cwd=ROOT,
                                   env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
                                   stdout=stream, stderr=subprocess.STDOUT)
    identity = process_identity(process.pid)
    reference = {"pid": process.pid, "creation_token": str(identity["creation_token"]),
                 "adopted": False, "log": str(path)}
    if not identity_matches(reference):
        raise RuntimeError(f"{name} exited during launch; inspect {path}")
    print(f"START {name}: PID={process.pid}; log={path}", flush=True)
    return reference, process


def prepare_explicit_metadata_recovery(manifest_path, names, references, prior):
    """Archive the diagnosed failure, then authorise only its stopped queues.

    This is a one-shot operator request, not a failed-fit retry policy. The
    independent compatibility report must already pass the full run guard.
    All requested states and process identities are checked before any writes.
    """
    names = tuple(names)
    if not names or len(set(names)) != len(names) or not set(names) <= {"followup_gpu", "analysis"}:
        raise ValueError("Explicit metadata recovery is restricted to the two diagnosed queues.")
    root = manifest_path.parent
    manifest = read_manifest(manifest_path)
    if not is_complete(manifest, "ulb_2013", "fttransformer", "none",
                       expected_n_trials=50, expected_bootstrap_iterations=1000):
        raise RuntimeError("The diagnosed baseline lacks independently validated completion.")
    states = {}
    for name in names:
        path = queue_state_path(root, name)
        saved = read_manifest(path)
        if (saved.get("status") != "failed" or saved.get("failure") != BOOTSTRAP_METADATA_FAILURE
                or Path(saved.get("source_manifest", "")).resolve() != manifest_path.resolve()):
            raise RuntimeError(f"{name} is not the diagnosed metadata failure; refuse a generic retry")
        candidates = [prior.get("queues", {}).get(name), references.get("queues", {}).get(name)]
        if saved.get("queue_pid"):
            candidates.insert(0, {"pid": saved["queue_pid"], "creation_token": saved.get("queue_creation_token")})
        if any(identity_matches(candidate) for candidate in candidates):
            raise RuntimeError(f"{name} is still alive; no recovery state is written")
        reject_orphan_children(next((candidate for candidate in candidates if candidate), None), saved)
        states[name] = (path, saved)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = root / "operational_recovery" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    archived = {}
    for name, (path, _) in states.items():
        destination = archive / path.name
        destination.write_bytes(path.read_bytes())
        archived[name] = {"path": str(destination), "sha256": file_sha256(destination)}
    old_supervisor = root / "overnight_status.json"
    if old_supervisor.is_file():
        destination = archive / old_supervisor.name
        destination.write_bytes(old_supervisor.read_bytes())
        archived["supervisor"] = {"path": str(destination), "sha256": file_sha256(destination)}
    for name, (path, saved) in states.items():
        recovered = {key: value for key, value in saved.items()
                     if key not in ("failure", "failed_at_utc", "exit_code")}
        recovered.update(status="explicit_metadata_recovery_authorised",
                         explicit_recovery={"reason": "Independently verified missing legacy bootstrap metadata",
                                            "authorised_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "previous_failure": archived[name],
                                            "automatic_retry": False})
        write_state(path, recovered)
    print(f"EXPLICIT RECOVERY {', '.join(names)}; failed states retained in {archive}", flush=True)
    return {"queues": list(names), "archive": str(archive), "archived_states": archived}


def prepare_verified_attention_recovery(manifest_path, references, prior):
    """Recover the known attention-only failure after full output verification.

    Extraction is performed and reviewed separately before this operator action.
    All three GPU steps must already be complete; no fit or failed extraction is
    retried by this function. Other live queues and their states are untouched.
    """
    root = manifest_path.parent
    path = queue_state_path(root, "followup_gpu")
    saved = read_manifest(path)
    log = root / "logs/followup_baf_ft_attention_20261006_100429.log"
    expected_failure = f"Follow-up baf_ft_attention failed (1); inspect {log}"
    if (saved.get("status") != "failed" or saved.get("current_step") != "baf_ft_attention"
            or saved.get("failure") != expected_failure or saved.get("last_child_exit_code") != 1
            or Path(saved.get("source_manifest", "")).resolve() != manifest_path.resolve()
            or not log.is_file() or ATTENTION_DRIFT_FAILURE not in log.read_text(encoding="utf-8")):
        raise RuntimeError("This is not the diagnosed attention drift failure; refuse a generic retry")
    candidates = [prior.get("queues", {}).get("followup_gpu"), references.get("queues", {}).get("followup_gpu")]
    if saved.get("queue_pid"):
        candidates.insert(0, {"pid": saved["queue_pid"], "creation_token": saved.get("queue_creation_token")})
    if any(identity_matches(candidate) for candidate in candidates):
        raise RuntimeError("GPU follow-up is still alive; no recovery state is written")
    reject_orphan_children(next((candidate for candidate in candidates if candidate), None), saved)
    for step in ("baf_absence_sensitivity_transformer", "baf_ft_smotenc", "baf_ft_attention"):
        if not step_complete(step, manifest_path, root):
            raise RuntimeError(f"The GPU dependency is not complete and verified: {step}")
    manifest = read_manifest(manifest_path)
    baseline = Path(manifest["baseline_runs"]["baf_base/fttransformer"]["run_dir"])
    baseline = (baseline if baseline.is_absolute() else ROOT / baseline).resolve()
    config = read_manifest(baseline / "config.json")
    directory = root / "derived/interpretability/attention"
    summary_path = directory / "attention_summary.json"
    summary = read_manifest(summary_path)
    verification = summary.get("score_verification", {})
    if (summary.get("n_test_samples") != config["test_samples"]
            or verification.get("compared_test_rows") != config["test_samples"]
            or verification.get("absolute_tolerance") != 1e-5
            or summary.get("extractor_code_sha256") != file_sha256(ROOT / "src/attention_analysis.py")
            or summary.get("model_implementation_sha256") != file_sha256(ROOT / "src/models/fttransformer.py")):
        raise RuntimeError("Full-population attention verification and current extraction provenance are required")
    import numpy as np
    weights = np.load(directory / "attention_cls_weights.npy", allow_pickle=False, mmap_mode="r")
    token_count = 1 + summary["n_numerical_features"] + summary["n_categorical_features"]
    if (weights.shape != (config["test_samples"], token_count) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise RuntimeError("Complete finite, non-negative, normalised attention weights are required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = root / "operational_recovery" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    archived = {}
    for name, source in (("followup_gpu", path), ("supervisor", root / "overnight_status.json")):
        if source.is_file():
            destination = archive / source.name
            destination.write_bytes(source.read_bytes())
            archived[name] = {"path": str(destination), "sha256": file_sha256(destination)}
    recovered = {key: value for key, value in saved.items()
                 if key not in ("failure", "failed_at_utc", "exit_code", "last_child_exit_code",
                                "current_log", "current_command")}
    recovered.update(status="explicit_verified_attention_recovery_authorised",
                     current_output_directory=str(directory),
                     explicit_recovery={"reason": "Native-layer attention extraction independently verified on all TEST rows",
                                        "authorised_at_utc": datetime.now(timezone.utc).isoformat(),
                                        "previous_failure": archived["followup_gpu"],
                                        "verified_summary": {"path": str(summary_path), "sha256": file_sha256(summary_path)},
                                        "automatic_retry": False})
    write_state(path, recovered)
    print(f"EXPLICIT ATTENTION RECOVERY; full TEST verified; failed states retained in {archive}", flush=True)
    return {"queues": ["followup_gpu"], "archive": str(archive), "archived_states": archived,
            "verified_test_rows": config["test_samples"], "summary_sha256": file_sha256(summary_path)}


def prepare_verified_transfer_recovery(manifest_path, references, prior):
    """Authorise the diagnosed CPU/CUDA inference recovery, never a fit retry.

    A separately reviewed, full-CUDA-population report and all four completed
    partial transfer families must pass before a failed queue state is changed.
    Partial scientific outputs remain in place and are revalidated by the child.
    """
    from revision_transfer import verify_partial_transfer, resolve_transfer_source
    from run_revision_analysis import partition_directory

    root = manifest_path.parent
    path = queue_state_path(root, "analysis")
    saved = read_manifest(path)
    log = root / "logs/analysis_transfer_six_models_20261006_235034_037097.log"
    expected_failure = f"Analysis transfer_six_models failed; inspect {log}"
    command = saved.get("current_command", [])
    device_position = command.index("--device") if "--device" in command else -1
    failed_device = command[device_position + 1] if 0 <= device_position < len(command) - 1 else None
    if (saved.get("status") != "failed" or saved.get("current_step") != "transfer_six_models"
            or saved.get("failure") != expected_failure or saved.get("last_child_exit_code") != 1
            or Path(saved.get("source_manifest", "")).resolve() != manifest_path.resolve()
            or not log.is_file() or TRANSFER_CPU_FAILURE not in log.read_text(encoding="utf-8")
            or failed_device != "cpu"):
        raise RuntimeError("This is not the diagnosed transfer CPU/CUDA failure; refuse a generic retry")
    candidates = [prior.get("queues", {}).get("analysis"), references.get("queues", {}).get("analysis")]
    if saved.get("queue_pid"):
        candidates.insert(0, {"pid": saved["queue_pid"], "creation_token": saved.get("queue_creation_token")})
    if any(identity_matches(candidate) for candidate in candidates):
        raise RuntimeError("Analysis is still alive; no recovery state is written")
    reject_orphan_children(next((candidate for candidate in candidates if candidate), None), saved)
    for name in ("classical", "transformer", "followup_cpu", "followup_gpu"):
        if not queue_complete(manifest_path, name):
            raise RuntimeError(f"The completed scientific dependency is not verified: {name}")

    report_path = root / "audit/transfer_cpu_cuda_20261007/ft_base_verification.json"
    report = read_manifest(report_path)
    run, config = resolve_transfer_source(manifest_path, "fttransformer")
    verified = report.get("cuda_verification", {})
    expected_files = {"config.json", "model.pt", "preprocessors.joblib", "y_test.npy",
                      "y_test_scores.npy", "test_row_indices.npy"}
    if (report.get("status") != "complete" or report.get("new_fits") != 0
            or report.get("source_files_modified") is not False
            or Path(report.get("source_run", "")).resolve() != run
            or report.get("protocol_version") != PROTOCOL_VERSION or config.get("protocol_version") != PROTOCOL_VERSION
            or report.get("batch_size") != 2048 or report.get("labels_and_indices_exactly_equal") is not True
            or report.get("model_implementation_sha256") != file_sha256(ROOT / "src/models/fttransformer.py")
            or report.get("score_absolute_tolerance") != 1e-6 or report.get("score_relative_tolerance") != 1e-5
            or verified.get("device") != "cuda" or verified.get("rows") != config["test_samples"]
            or verified.get("scores_bitwise_equal") is not True
            or verified.get("maximum_absolute_score_difference") != 0
            or verified.get("rows_outside_existing_tolerance") != 0
            or verified.get("decisions_changed_at_frozen_threshold") != 0
            or set(report.get("source_artefacts_sha256", {})) != expected_files):
        raise RuntimeError("Current full-population CUDA reproduction evidence is required")
    for filename, digest in report["source_artefacts_sha256"].items():
        if file_sha256(run / filename) != digest:
            raise RuntimeError(f"A protected FT source differs from its CUDA verification: {filename}")
    directory = root / "derived/transfer"
    partial = verify_partial_transfer(directory, manifest_path, partition_directory(manifest_path))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not replace exact inference with a looser tolerance")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = root / "operational_recovery" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    archived = {}
    for name, source in (("analysis", path), ("supervisor", root / "overnight_status.json")):
        if source.is_file():
            destination = archive / source.name
            destination.write_bytes(source.read_bytes())
            archived[name] = {"path": str(destination), "sha256": file_sha256(destination)}
    recovery = {"reason": "Verified original CUDA backend exactly reproduces all saved FT Base TEST scores",
                "authorised_at_utc": datetime.now(timezone.utc).isoformat(),
                "previous_failure": archived["analysis"], "automatic_retry": False,
                "partial_directory": str(directory.resolve()), "device": "cuda", "batch_size": 2048,
                "verification_report": {"path": str(report_path.resolve()), "sha256": file_sha256(report_path)},
                "verified_reuse": partial["reuse_provenance"],
                "transfer_code_sha256": file_sha256(ROOT / "src/revision_transfer.py")}
    recovered = {key: value for key, value in saved.items()
                 if key not in ("failure", "failed_at_utc", "exit_code", "last_child_exit_code")}
    recovered.update(status="explicit_verified_transfer_recovery_authorised", explicit_transfer_recovery=recovery)
    write_state(path, recovered)
    print(f"EXPLICIT TRANSFER RECOVERY; 20 partial evaluations preserved; failed states retained in {archive}", flush=True)
    return {"queues": ["analysis"], "archive": str(archive), "archived_states": archived,
            "verification_report_sha256": file_sha256(report_path), "reused_models": list(partial["models"])}


def prepare_verified_reporting_recovery(manifest_path, references, prior):
    """Recover only the diagnosed Windows output-key failure after all work ended.

    The original QA and scientific outputs are not rewritten. All five queues
    must pass their complete-evidence guards, and no process may be adopted or
    launched in this final-export-only recovery.
    """
    root = manifest_path.parent
    path = root / "analysis_state.json"
    saved = read_manifest(path)
    qa_path = root / "derived/reporting/generation_qa.json"
    error_log = root / "logs/overnight_supervisor_20261007T0842385510422Z.error.log"
    if (prior.get("status") != "failed" or prior.get("failure") != REPORTING_PATH_FAILURE
            or Path(prior.get("manifest_path", "")).resolve() != manifest_path.resolve()
            or prior.get("active_thesis_sources_modified") is not False
            or prior.get("primary_pins_present") != 43 or prior.get("primary_pins_required") != 43
            or prior.get("queues_finished") != 4
            or saved.get("status") != "analysis_ready_for_human_review"
            or saved.get("current_step") != "isolated_reporting_export"
            or saved.get("last_child_exit_code") != 0
            or tuple(saved.get("completed_steps", ())) != REPORTING_COMPLETED_STEPS
            or Path(saved.get("source_manifest", "")).resolve() != manifest_path.resolve()
            or saved.get("frozen_manifest_sha256") != REPORTING_MANIFEST_SHA256
            or not error_log.is_file()
            or f"ValueError: {REPORTING_PATH_FAILURE}" not in error_log.read_text(encoding="utf-8")):
        raise RuntimeError("This is not the diagnosed reporting path failure; refuse a generic retry")
    if (file_sha256(manifest_path) != REPORTING_MANIFEST_SHA256
            or file_sha256(qa_path) != REPORTING_QA_SHA256):
        raise RuntimeError("The diagnosed frozen manifest or original reporting QA changed")
    qa = read_manifest(qa_path)
    if ("tables\\ulb\\baseline.tex" not in qa.get("output_sha256", {})
            or "tables/ulb/baseline.tex" in qa.get("output_sha256", {})):
        raise RuntimeError("The reporting QA no longer has the diagnosed Windows output-key mismatch")
    supervisor = prior.get("supervisor", {})
    if not supervisor.get("pid") or not supervisor.get("creation_token"):
        raise RuntimeError("The previous supervisor lacks a complete process identity; refuse recovery")
    if identity_matches(supervisor):
        raise RuntimeError("The previous supervisor is still alive; no recovery state is written")
    reject_orphan_children(prior.get("supervisor"), {})
    for name in QUEUE_NAMES:
        queue_path = queue_state_path(root, name)
        queue_state = read_manifest(queue_path) if queue_path.is_file() else {}
        candidates = [prior.get("queues", {}).get(name), references.get("queues", {}).get(name)]
        if queue_state.get("queue_pid"):
            candidates.insert(0, {"pid": queue_state["queue_pid"],
                                  "creation_token": queue_state.get("queue_creation_token")})
        if any(candidate and candidate.get("pid") and not candidate.get("creation_token")
               for candidate in candidates):
            raise RuntimeError(f"{name} lacks a complete process identity; refuse recovery")
        if any(identity_matches(candidate) for candidate in candidates):
            raise RuntimeError(f"{name} is still alive; no final-export recovery state is written")
        for candidate in candidates:
            if candidate and candidate.get("pid"):
                reject_orphan_children(candidate, queue_state)
        if not queue_complete(manifest_path, name):
            raise RuntimeError(f"The completed scientific dependency is not verified: {name}")
    if (file_sha256(manifest_path) != REPORTING_MANIFEST_SHA256
            or file_sha256(qa_path) != REPORTING_QA_SHA256):
        raise RuntimeError("Reporting inputs changed during recovery verification")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = root / "operational_recovery" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    archived = {}
    for name, source in (("analysis", path), ("supervisor", root / "overnight_status.json"),
                         ("generation_qa", qa_path)):
        destination = archive / source.name
        destination.write_bytes(source.read_bytes())
        archived[name] = {"path": str(destination), "sha256": file_sha256(destination)}
    return {"reason": "Verified legacy Windows output keys normalised only in validator memory",
            "authorised_at_utc": datetime.now(timezone.utc).isoformat(), "archive": str(archive),
            "archived_states": archived, "queues": [], "revalidated_queues": list(QUEUE_NAMES),
            "original_generation_qa_sha256": REPORTING_QA_SHA256,
            "source_manifest_sha256": REPORTING_MANIFEST_SHA256,
            "validator_code_sha256": file_sha256(ROOT / "src/revision_thesis_assets.py"),
            "analysis_queue_restarted": False, "reporting_generator_reexecuted": False,
            "isolated_sensitivity_fragment_export_allowed": True,
            "scientific_outputs_modified": False, "automatic_retry": False}


def complete_verified_reporting_recovery(manifest_path, references, prior, owner):
    """Finalise verified exports without entering any queue-launch path."""
    recovery = prepare_verified_reporting_recovery(manifest_path, references, prior)
    path = manifest_path.parent / "overnight_status.json"
    state = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "supervisor": owner, "manifest_path": str(manifest_path),
             "queues": {name: {"status": "complete"} for name in QUEUE_NAMES},
             "primary_pins_present": 43, "primary_pins_required": 43, "queues_finished": 5,
             "active_thesis_sources_modified": False, "automatic_failed_job_retries": False,
             "explicit_reporting_recovery": recovery}
    write_state(path, state)
    print("EXPLICIT REPORTING RECOVERY; all five queues revalidated; no scientific process launched", flush=True)
    try:
        state.update(status="ready_for_thesis_revision", evidence=finish_exports(manifest_path),
                     completed_at_utc=datetime.now(timezone.utc).isoformat(),
                     next_action="Notify Codex to review new evidence, revise tables/prose/abstracts, then compile in Overleaf. Not a submission-ready thesis.")
        write_state(path, state)
        print("COMPLETE: computations and isolated exports ready; thesis and PDF review still required", flush=True)
    except Exception as error:
        state.update(status="failed", failure=str(error), failed_at_utc=datetime.now(timezone.utc).isoformat(),
                     next_action="Notify Codex with overnight_status.json and logs. Saved results are preserved; no scientific process was launched.")
        write_state(path, state)
        raise


def finish_exports(manifest_path):
    """Fresh verification and isolated fragments only; no active-source edits."""
    root = manifest_path.parent
    from run_revision_analysis import (validate_paired, validate_thresholds, validate_transfer,
                                      validate_sampler_audit, validate_attention_text, validate_diagnostics,
                                      variant_shap_ready)
    checks = [validate_paired(manifest_path, root), validate_transfer(manifest_path, root),
              validate_sampler_audit(manifest_path, root), validate_attention_text(manifest_path, root),
              validate_diagnostics(manifest_path, root), variant_shap_ready(manifest_path, root),
              *(validate_thresholds(manifest_path, root, dataset) for dataset in ("ulb_2013", "baf_base"))]
    if not all(checks):
        raise RuntimeError("A final scientific dependency is missing; no completion declared")
    from revision_thesis_assets import verify_generation
    verify_generation(root / "derived/reporting", manifest_path)
    destination = root / "derived/sensitivity_text"
    if destination.exists():
        qa = read_manifest(destination / "sensitivity_text_qa.json")
        if (qa.get("status") != "complete" or qa.get("primary_manifest_sha256") != file_sha256(manifest_path)
                or qa.get("source_report_sha256") != file_sha256(root / "derived/paired/paired_analysis.json")
                or qa.get("sensitivity_manifest_sha256") != file_sha256(root / "sensitivity/revision_manifest.json")
                or qa.get("fragment_sha256") != file_sha256(destination / "sensitivity_tables.tex")):
            raise RuntimeError("Existing sensitivity export is partial or changed; it is not overwritten")
        for path, digest in qa["source_artefacts_sha256"].items():
            if file_sha256(Path(path)) != digest:
                raise RuntimeError(f"A sensitivity source changed: {path}")
    else:
        from revision_sensitivity_text import generate
        generate(root / "derived/paired/paired_analysis.json", manifest_path,
                 root / "sensitivity/revision_manifest.json", destination)
    return {"manifest_sha256": file_sha256(manifest_path),
            "generation_qa_sha256": file_sha256(root / "derived/reporting/generation_qa.json"),
            "sensitivity_text_qa_sha256": file_sha256(destination / "sensitivity_text_qa.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results_revision/20261005/revision_manifest.json")
    parser.add_argument("--attachments", type=Path, default=ROOT / "results_revision/20261005/overnight_attachments.json")
    parser.add_argument("--poll-seconds", type=int, default=60)
    recovery_options = parser.add_mutually_exclusive_group()
    recovery_options.add_argument("--recover-failed-queues", nargs="+", choices=("followup_gpu", "analysis"),
                                  help="One-shot recovery of the diagnosed missing-bootstrap metadata failure only.")
    recovery_options.add_argument("--recover-verified-attention", action="store_true",
                                  help="Recover the known attention-only failure after all TEST rows have passed extraction QA.")
    recovery_options.add_argument("--recover-verified-transfer", action="store_true",
                                  help="Recover the diagnosed CPU/CUDA transfer failure after source and partial-output QA.")
    recovery_options.add_argument("--recover-verified-reporting", action="store_true",
                                  help="Finalise the diagnosed Windows reporting-key failure only; never launch a scientific queue.")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    if not manifest_path.is_file() or not root.is_relative_to(ROOT / "results_revision"):
        raise ValueError("Use an existing isolated revision manifest, never the historical archive")
    if args.poll_seconds < 10:
        raise ValueError("Polling must be at least 10 seconds")
    references = read_manifest(args.attachments)
    if Path(references.get("manifest_path", "")).resolve() != manifest_path:
        raise ValueError("Queue attachments identify a different manifest")
    path = root / "overnight_status.json"
    prior = read_manifest(path)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    with keep_system_awake(), exclusive_resource(root, queue_resource_name(root, "overnight"),
                                                 "Overnight revision supervisor", wait=False) as owner:
        if args.recover_verified_reporting:
            complete_verified_reporting_recovery(manifest_path, references, prior, owner)
            return
        recovery = (prepare_explicit_metadata_recovery(manifest_path, args.recover_failed_queues, references, prior)
                    if args.recover_failed_queues else None)
        attention_recovery = (prepare_verified_attention_recovery(manifest_path, references, prior)
                              if args.recover_verified_attention else None)
        transfer_recovery = (prepare_verified_transfer_recovery(manifest_path, references, prior)
                             if args.recover_verified_transfer else None)
        state = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                 "supervisor": owner, "manifest_path": str(manifest_path), "queues": {},
                 "active_thesis_sources_modified": False, "automatic_failed_job_retries": False}
        if recovery:
            state["explicit_metadata_recovery"] = recovery
        if attention_recovery:
            state["explicit_attention_recovery"] = attention_recovery
        if transfer_recovery:
            state["explicit_transfer_recovery"] = transfer_recovery
        processes = {}
        try:
            for name in QUEUE_NAMES:
                reference, process = attach_or_start(name, manifest_path, references["queues"],
                                                    prior.get("queues", {}), logs)
                state["queues"][name] = reference
                if process:
                    processes[name] = process
                write_state(path, state)
            while True:
                finished = 0
                for name, reference in state["queues"].items():
                    if reference.get("status") == "complete":
                        finished += 1
                        continue
                    if name not in ("classical", "transformer"):
                        current = read_manifest(queue_state_path(root, name))
                        if current.get("status") == "failed":
                            raise RuntimeError(f"{name} failed: {current.get('failure', current)}; inspect its logs")
                    if identity_matches(reference):
                        continue
                    if name in processes and processes[name].poll() not in (None, 0):
                        raise RuntimeError(f"{name} exited with code {processes[name].returncode}; inspect {reference.get('log')}")
                    if not queue_complete(manifest_path, name):
                        raise RuntimeError(f"{name} stopped before validated completion; no automatic retry or dependent promotion")
                    reference["status"] = "complete"
                    finished += 1
                    print(f"DONE {name}: scientific artefacts revalidated", flush=True)
                manifest = read_manifest(manifest_path)
                expected = [f"ulb_2013/{model}/{strategy}" for model in CLASSICAL
                            for strategy in (("none",) if model == "ocsvm" else STRATEGIES)]
                expected += [f"{dataset}/fttransformer/{strategy}" for dataset in ("ulb_2013", "baf_base")
                             for strategy in STRATEGIES]
                state.update(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                             primary_pins_present=sum(key in manifest.get("runs", {}) for key in expected),
                             primary_pins_required=43, queues_finished=finished)
                write_state(path, state)
                if finished == len(QUEUE_NAMES):
                    break
                time.sleep(args.poll_seconds)
            state.update(status="ready_for_thesis_revision", evidence=finish_exports(manifest_path),
                         completed_at_utc=datetime.now(timezone.utc).isoformat(),
                         next_action="Notify Codex to review new evidence, revise tables/prose/abstracts, then compile in Overleaf. Not a submission-ready thesis.")
            write_state(path, state)
            print("COMPLETE: computations and isolated exports ready; thesis and PDF review still required", flush=True)
        except Exception as error:
            state.update(status="failed", failure=str(error), failed_at_utc=datetime.now(timezone.utc).isoformat(),
                         next_action="Notify Codex with overnight_status.json and logs. Existing jobs are not terminated and saved results are preserved.")
            write_state(path, state)
            raise


if __name__ == "__main__":
    main()
