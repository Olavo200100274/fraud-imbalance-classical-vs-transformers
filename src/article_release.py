"""Publish and verify portable article evidence without fitting any model.

Author-side export requires the pinned local archive. Public verification uses
the standard library; numerical replay uses the recorded NumPy/sklearn stack.
All replay outputs go to a separate directory, never to frozen results/.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "results/provenance/article_manifest.json"
PARENT = "https://github.com/Olavo200100274/fraud-imbalance-classical-vs-transformers"
SLUGS = {1: "transformers-vs-ml-fraud-detection", 2: "tabular-ml-dl-threshold-benchmark"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Expected a safe repository-relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes repository")
    return path


def json_bytes(document):
    return (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def verify(root=ROOT):
    root = Path(root).resolve()
    document = read_json(root / MANIFEST)
    if document.get("schema") != "dedicated_article_evidence_v1":
        raise ValueError("Unsupported article manifest")
    for relative, record in document["files"].items():
        path = safe_path(root, relative)
        if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Release file differs from its pin: {relative}")
    expected_keys = {f"{dataset}/{model}/{strategy}" for dataset in ("ulb_2013", "baf_base")
                     for model in ("logreg", "rf", "lgbm", "catboost", "fttransformer")
                     for strategy in ("none", "rus", "ros", "smote", "smote_tomek", "smoteenn", "weights")}
    expected_keys.update({"ulb_2013/ocsvm/none", "baf_base/ocsvm/none"})
    if set(document["primary_runs"]) != expected_keys:
        raise ValueError("Incomplete primary evidence")
    if document["article"] == 2 and len(document["threshold_studies"]) != 12:
        raise ValueError("Incomplete threshold evidence")
    for record in document["primary_runs"].values():
        for field in ("archive", "metrics", "config"):
            if record[field] not in document["files"]:
                raise ValueError("Unpinned primary input")
    if len(document["control_runs"]) != 8 or (document["article"] == 1 and len(document["transfer_runs"]) != 30):
        raise ValueError("Incomplete article control or transfer evidence")
    for mapping in (document["control_runs"], document["transfer_runs"]):
        for record in mapping.values():
            for field in ("archive", "metrics"):
                if record[field] not in document["files"]:
                    raise ValueError("Unpinned derived numeric input")
    for relative in document["threshold_studies"].values():
        if relative not in document["files"]:
            raise ValueError("Unpinned threshold study")
    return document


def export(author_root, destination, article):
    """Copy explicitly pinned evidence and losslessly compress numeric arrays."""
    import numpy as np
    from export_public_evidence import portable, source_path

    author_root, destination = Path(author_root).resolve(), Path(destination).resolve()
    if destination == author_root or not (destination / ".git").is_dir():
        raise ValueError("Destination must be a separate existing repository checkout")
    for protected in ("results", "results_revision", "Overleaf", "Article 1", "Article 2"):
        if destination.is_relative_to(author_root / protected):
            raise ValueError("Cannot export into a protected source archive")
    previous_path = destination / MANIFEST
    previous = read_json(previous_path) if previous_path.exists() else {}
    inventory = {}

    def emit(relative, data, sources=()):
        path = safe_path(destination, relative)
        if path.exists() and path.read_bytes() != data:
            old = previous.get("files", {}).get(relative)
            if old is None or digest(path) != old["sha256"]:
                raise FileExistsError(f"Unowned or modified destination: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        inventory[relative] = {"sha256": digest(path), "bytes": len(data)}
        if sources:
            inventory[relative]["sources"] = sources

    def copy(relative, source):
        emit(relative, source.read_bytes(), [{"archive_path": portable(str(source), author_root), "sha256": digest(source)}])

    def arrays(relative, run, names, pins=None):
        contents, identities = {}, {}
        for name in names:
            path = run / name
            if not path.is_file():
                if name in ("y_test.npy", "y_test_scores.npy", *validation):
                    raise FileNotFoundError(f"Required frozen array missing: {path}")
                continue
            actual = digest(path)
            expected = (pins or {}).get(name)
            if expected is not None and actual != expected:
                raise ValueError(f"Frozen numeric source changed: {path}")
            contents[path.stem] = np.load(path, allow_pickle=False)
            identities[name] = {"sha256": actual, "shape": list(contents[path.stem].shape), "dtype": str(contents[path.stem].dtype)}
        if not {"y_test", "y_test_scores"}.issubset(contents):
            raise ValueError(f"Missing frozen TEST inputs: {run}")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **contents)
        emit(relative, buffer.getvalue())
        return identities

    revision = author_root / "results_revision/20261005"
    qa_path = revision / "derived/reporting/generation_qa.json"
    qa = read_json(qa_path)
    original = revision / "revision_manifest.json"
    if digest(original) != qa["manifest_sha256"]:
        raise ValueError("Frozen reporting and selection manifests differ")
    # Shared numerical comparisons are relevant to both manuscripts. Do not
    # copy thesis-dependent verifier manifests or unrelated article outputs.
    for subtree in ("metrics", "controls") + (("thresholds",) if article == 2 else ("transfer", "interpretability")):
        for source in sorted((author_root / "results" / subtree).rglob("*")):
            if source.is_file():
                copy("results/" + source.relative_to(author_root / "results").as_posix(), source)
    for name in ("selected_runs.json", "historical_baf_selection.json", "dataset_protocol.json", "baf_absence_codes.json", "sampler_identity.json", "lgbm_score_diagnostics.json", "execution_provenance_notes.json"):
        source = author_root / "results/provenance" / name
        if source.is_file():
            document = read_json(source)
            if name == "dataset_protocol.json" and article == 2:
                document = {key: value for key, value in document.items() if not key.startswith("baf_var")}
            emit("results/provenance/" + name, json_bytes(document))
    article_root = author_root / f"Article {article}"
    text = (article_root / "main.tex").read_text(encoding="utf-8-sig")
    active_figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)))
    active_tables = sorted(set(re.findall(r"\\input\{(tables/[^}]+)\}", text)))
    for relative in active_figures:
        source = article_root / relative
        if not source.is_file():
            source = article_root / "figs" / relative
        copy("results/figures/" + Path(relative).name, source)
    for relative in active_tables:
        source = article_root / relative
        if not source.suffix:
            source = source.with_suffix(".tex")
        copy("results/tables/" + source.name, source)
    if article == 2:  # This manuscript's fourteen tables are inline.
        tabulars = [match.group(0) for match in re.finditer(r"\\begin\{(tabularx?)\}.*?\\end\{\1\}", text, re.S)]
        if len(tabulars) != 14:
            raise ValueError("Expected fourteen inline manuscript tables")
        for number, tabular in enumerate(tabulars, 1):
            emit(f"results/tables/table_{number:02d}.tex", (tabular + "\n").encode("utf-8"))
    primary, studies = {}, {}
    names = ("y_test.npy", "y_test_scores.npy", "test_row_indices.npy")
    validation = ("y_val.npy", "y_val_scores.npy", "validation_fold_ids.npy", "validation_row_indices.npy")
    for key, record in sorted(qa["source_runs"].items()):
        run = source_path(record["run_dir"], author_root)
        fields = {"y_test.npy": record["y_test_sha256"], "y_test_scores.npy": record["y_test_scores_sha256"]}
        if digest(run / "config.json") != record["config_sha256"] or digest(run / "metrics_test.json") != record["metrics_sha256"]:
            raise ValueError(f"Changed primary selection: {key}")
        selected_names = names
        if article == 2 and key.endswith("/none"):
            threshold_key = key.removesuffix("/none")
            study = read_json(author_root / "results/thresholds" / (threshold_key + ".json"))
            fields.update(study["source_sha256"])
            selected_names += validation
            studies[threshold_key] = "results/thresholds/" + threshold_key + ".json"
        archive = "results/frozen/primary/" + key + ".npz"
        identities = arrays(archive, run, selected_names, fields)
        metric_path = "results/metrics/" + key + "/metrics_full_precision.json"
        primary[key] = {"archive": archive, "config": "results/metrics/" + key + "/config.json", "metrics": metric_path,
                        "threshold": read_json(destination / metric_path)["threshold"], "source_artifacts": identities}
    paired = read_json(author_root / "results/controls/paired_analysis.json")
    controls, control_pins, control_expected = {}, {}, {}
    for model, record in paired["missingness"].items():
        controls["missingness/" + model] = source_path(record["second_run"], author_root)
        control_pins["missingness/" + model] = record["second_artefacts_sha256"]
        control_expected["missingness/" + model] = record["second_metrics"]
    controls["categorical/fttransformer"] = source_path(paired["categorical_controls"]["fttransformer"]["second_run"], author_root)
    control_pins["categorical/fttransformer"] = paired["categorical_controls"]["fttransformer"]["second_artefacts_sha256"]
    control_expected["categorical/fttransformer"] = paired["categorical_controls"]["fttransformer"]["second_metrics"]
    controls["categorical/catboost"] = source_path(paired["catboost_control_source"]["run_dir"], author_root)
    control_pins["categorical/catboost"] = paired["catboost_control_source"]["artefacts_sha256"]
    control_expected["categorical/catboost"] = paired["categorical_controls"]["catboost"]["control_metrics"]
    control_records = {}
    for key, run in controls.items():
        archive = "results/frozen/controls/" + key + ".npz"
        identities = arrays(archive, run, names, control_pins[key])
        for name in ("config.json", "metrics_test.json", "primary_smote_comparison.json"):
            if (run / name).is_file():
                if name in control_pins[key] and digest(run / name) != control_pins[key][name]:
                    raise ValueError(f"Frozen control record changed: {key}/{name}")
                emit("results/controls/runs/" + key + "/" + name, json_bytes(portable(read_json(run / name), author_root)))
        control_records[key] = {"archive": archive, "config": "results/controls/runs/" + key + "/config.json",
                                "metrics": "results/controls/runs/" + key + "/metrics_test.json", "source_artifacts": identities,
                                "expected_metrics": control_expected[key], "expected_tolerance": 1e-6 if key == "categorical/catboost" else 1e-12}
    transfers = {}
    if article == 1:
        for model, record in qa["transfer_sources"].items():
            for variant, selection in record["variants"].items():
                run = source_path(selection["path"], author_root).parent
                archive = f"results/frozen/transfer/{model}/{variant}.npz"
                pins = {name: value["sha256"] for name, value in selection["artefacts"].items()}
                identities = arrays(archive, run, names, pins)
                transfers[f"{model}/{variant}"] = {"archive": archive, "metrics": f"results/transfer/{model}/{variant}/metrics_test.json", "source_artifacts": identities}
    copy(f"publications/article_{article}.pdf", author_root / f"publications/article_{article}.pdf")
    reference = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=author_root, text=True).strip()
    manifest = {"schema": "dedicated_article_evidence_v1", "article": article,
                "repository": "https://github.com/Applied-Intelligence-Hub/" + SLUGS[article],
                "parent_repository": PARENT, "parent_source_commit": reference,
                "original_reporting_qa_sha256": digest(qa_path), "original_selection_sha256": digest(original),
                "primary_runs": primary, "control_runs": control_records, "threshold_studies": studies, "transfer_runs": transfers,
                "manuscript_status": "Unpublished manuscript in preparation for submission; reviewed PDF includes dedicated repository links",
                "omitted": ["raw datasets", "trained model weights", "full training archives", "private writing sources", "editorial correspondence"],
                "files": dict(sorted(inventory.items()))}
    # Adding plot-ready inputs later updates this dedicated manifest, not the
    # immutable author's original manifest. Every extra input must be inventoried.
    path = destination / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(manifest))
    return manifest


def refresh_inventory(root):
    """Pin newly exported plot inputs and scientific source copies before release."""
    root = Path(root).resolve()
    document = read_json(root / MANIFEST)
    for directory in ("src", "tests", "datasets", "notebooks", "results", "publications"):
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST or relative.endswith("README.md"):
                continue
            previous = document["files"].get(relative, {})
            document["files"][relative] = {**previous, "sha256": digest(path), "bytes": path.stat().st_size}
    for relative in ("requirements.txt", ".gitattributes", "LICENSE"):
        path = root / relative
        document["files"][relative] = {"sha256": digest(path), "bytes": path.stat().st_size}
    safe_path(root, MANIFEST).write_bytes(json_bytes(document))
    return document


def frozen_metrics(labels, scores, threshold, probability=True):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Frozen metric inputs must be aligned, finite and binary-labelled")
    prediction = scores >= threshold
    tp, fp = int(np.sum(prediction & (labels == 1))), int(np.sum(prediction & (labels == 0)))
    fn, tn = int(np.sum(~prediction & (labels == 1))), int(np.sum(~prediction & (labels == 0)))
    result = {"PR-AUC": float(average_precision_score(labels, scores)), "ROC-AUC": float(roc_auc_score(labels, scores)),
              "TP": tp, "FP": fp, "FN": fn, "TN": tn, "threshold": float(threshold) if math.isfinite(threshold) else None,
              "precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / (tp + fn),
              "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
              "F2": 5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else 0., "alert_rate": (tp + fp) / len(labels)}
    if probability:
        result["brier_score"] = float(np.mean((scores - labels) ** 2))
    return result


def recompute(root, output):
    import numpy as np
    document = verify(root)
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or any(output.is_relative_to(root / p) for p in ("results", "publications", "datasets", "src", "tests", "notebooks", ".git")):
        raise ValueError("Recomputation output must not overwrite frozen inputs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new, empty recomputation output directory")
    output.mkdir(parents=True, exist_ok=True)
    primary, threshold_results, controls, transfer = {}, {}, {}, {}
    for key, record in document["primary_runs"].items():
        with np.load(root / record["archive"], allow_pickle=False) as data:
            values = frozen_metrics(data["y_test"], data["y_test_scores"], record["threshold"], "/ocsvm/" not in key)
            expected = read_json(root / record["metrics"])
            for metric in ("PR-AUC", "ROC-AUC", "F1", "F2", "TP", "FP", "TN", "FN", "alert_rate"):
                if abs(values[metric] - expected[metric]) > 1e-12:
                    raise ValueError(f"Frozen replay differs: {key}/{metric}")
            primary[key] = values
            if key.endswith("/none") and document["article"] == 2:
                from revision_thresholds import derive_thresholds
                model = key.split("/")[1]
                derived = derive_thresholds(data["y_val"], data["y_val_scores"], data["validation_fold_ids"], model)
                if np.intersect1d(data["validation_row_indices"], data["test_row_indices"]).size:
                    raise ValueError("Saved DEV and TEST indices overlap")
                study_key = key.removesuffix("/none")
                expected_study = read_json(root / document["threshold_studies"][study_key])
                rules = {}
                for rule, threshold in derived["thresholds_median"].items():
                    saved = expected_study["test_results"][rule]
                    saved_threshold = saved["threshold_exact"]
                    if saved_threshold is None:
                        saved_threshold = math.inf
                    if threshold != saved_threshold and not np.isclose(threshold, saved_threshold, atol=1e-12, rtol=0):
                        raise ValueError(f"Validation threshold replay differs: {study_key}/{rule}")
                    rules[rule] = frozen_metrics(data["y_test"], data["y_test_scores"], threshold, model != "ocsvm")
                    for count in ("TP", "FP", "TN", "FN"):
                        if rules[rule][count] != saved[count]:
                            raise ValueError("Threshold confusion counts differ")
                threshold_results[study_key] = rules
    for key, record in document["control_runs"].items():
        saved, config = read_json(root / record["metrics"]), read_json(root / record["config"])
        threshold = config.get("threshold_exact", saved["threshold"])
        with np.load(root / record["archive"], allow_pickle=False) as data:
            values = frozen_metrics(data["y_test"], data["y_test_scores"], threshold, not key.endswith("/ocsvm"))
        for count in ("TP", "FP", "TN", "FN"):
            if values[count] != saved[count]:
                raise ValueError(f"Control replay differs: {key}/{count}")
        for replay_key, aliases in (("PR-AUC", ("average_precision", "PR-AUC")), ("ROC-AUC", ("roc_auc", "ROC-AUC")), ("F2", ("F2",))):
            expected_values = record.get("expected_metrics", {})
            for alias in aliases:
                if alias in expected_values:
                    if abs(values[replay_key] - expected_values[alias]) > record.get("expected_tolerance", 1e-12):
                        raise ValueError(f"Control ranking replay differs: {key}/{replay_key}")
                    break
        controls[key] = values
    for key, record in document["transfer_runs"].items():
        saved = read_json(root / record["metrics"])
        with np.load(root / record["archive"], allow_pickle=False) as data:
            values = frozen_metrics(data["y_test"], data["y_test_scores"], saved["threshold_used"], not key.startswith("ocsvm/"))
        for count in ("TP", "FP", "TN", "FN"):
            if values[count] != saved["metrics"][count]:
                raise ValueError("Transfer counts differ")
        for metric, alias in (("PR-AUC", "average_precision"), ("ROC-AUC", "roc_auc")):
            if abs(values[metric] - saved["metrics_full_precision"][alias]) > 1e-12:
                raise ValueError("Transfer ranking replay differs")
        transfer[key] = values
    result = {"article": document["article"], "primary_metrics": primary, "threshold_studies": threshold_results,
              "control_metrics": controls, "transfer_metrics": transfer, "new_fits": 0,
              "bootstrap": "Archived interval evidence supplied separately; this command checks points/thresholds, not 1000 bootstrap resamples."}
    (output / "recomputed_metrics.json").write_bytes(json_bytes(result))
    return {"primary_cells": len(primary), "threshold_studies": len(threshold_results), "controls": len(controls), "transfer_cells": len(transfer), "new_fits": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--recompute", action="store_true")
    mode.add_argument("--export", action="store_true", help="Author-side operation requiring the full pinned archive")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/recomputed")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--article", type=int, choices=(1, 2))
    args = parser.parse_args()
    if args.export:
        if args.destination is None or args.article is None:
            parser.error("--export requires --destination and --article")
        document = export(args.root, args.destination, args.article)
        print(json.dumps({"article": args.article, "files": len(document["files"]), "bytes": sum(v["bytes"] for v in document["files"].values())}))
    elif args.recompute:
        print(json.dumps(recompute(args.root, args.output_dir)))
    else:
        document = verify(args.root)
        print(json.dumps({"verified": True, "article": document["article"], "files": len(document["files"])}))


if __name__ == "__main__":
    main()
