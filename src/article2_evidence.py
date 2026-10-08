"""Read the dedicated Article 2 release; never fit or select models.

All source paths are relative to this checkout. Full PR coordinates are derived
from the distributed frozen predictions, not compact plotting coordinates.
Temporary NPY inputs used by the inherited plotting interface stay in runs/.
"""

from functools import lru_cache
from pathlib import Path
import hashlib
import json

import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/provenance/article_manifest.json"
CACHE = ROOT / "runs/figure_inputs"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve(path):
    candidate = (ROOT / path).resolve()
    candidate.relative_to(ROOT.resolve())
    return candidate


def load_sources():
    from article_release import verify

    verify(ROOT)
    sources = read_json(MANIFEST)
    if len(sources["primary_runs"]) != 72 or len(sources["threshold_studies"]) != 12:
        raise ValueError("Article 2 requires its complete primary and threshold evidence.")
    return sources


def pin_current_sources():
    """Compatibility name: load verified existing pins without rewriting them."""
    return load_sources()


def _record(sources, dataset, model, strategy="none"):
    return sources["primary_runs"][f"{dataset}/{model}/{strategy}"]


def _checked_path(sources, path):
    source = resolve(path)
    record = sources["files"][Path(path).as_posix()]
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != record["sha256"] or source.stat().st_size != record["bytes"]:
        raise ValueError(f"Article source changed: {path}")
    return source


@lru_cache(maxsize=16)
def _arrays(path, expected_hash):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Frozen article predictions changed.")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    labels, scores = arrays["y_test"], arrays["y_test_scores"]
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid frozen article TEST arrays.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Frozen article TEST labels must contain both classes.")
    return arrays


def run_arrays(sources, dataset, model, strategy="none"):
    record = _record(sources, dataset, model, strategy)
    path = _checked_path(sources, record["archive"])
    return _arrays(str(path), sources["files"][record["archive"]]["sha256"])


def metrics(sources, dataset, model, strategy="none"):
    record = _record(sources, dataset, model, strategy)
    values = read_json(_checked_path(sources, record["metrics"]))
    tp, fp, tn, fn = (int(values[key]) for key in ("TP", "FP", "TN", "FN"))
    values.update(
        Precision=tp / (tp + fp) if tp + fp else 0.0,
        Recall=tp / (tp + fn) if tp + fn else 0.0,
        F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        F2=5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else 0.0,
        alert_rate=(tp + fp) / (tp + fp + tn + fn),
    )
    return values


def threshold_study(sources, dataset, model):
    path = sources["threshold_studies"][f"{dataset}/{model}"]
    return read_json(_checked_path(sources, path))


def read_run(sources, dataset, model, strategy, filename):
    """Provide the inherited figure-builder interface using public evidence."""
    record = _record(sources, dataset, model, strategy)
    if filename == "metrics_test.json":
        return metrics(sources, dataset, model, strategy)
    if filename == "config.json":
        return read_json(_checked_path(sources, record["config"]))
    arrays = run_arrays(sources, dataset, model, strategy)
    if filename == "pr_curve_data.json":
        precision, recall, thresholds = precision_recall_curve(
            arrays["y_test"], arrays["y_test_scores"])
        return {"precisions": precision.tolist(), "recalls": recall.tolist(),
                "thresholds": thresholds.tolist()}
    if filename.endswith(".npy"):
        key = Path(filename).stem
        if key not in arrays:
            raise FileNotFoundError(f"The release does not distribute {filename} for this run.")
        destination = CACHE / dataset / model / strategy / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = np.load(destination, allow_pickle=False)
            if existing.dtype != arrays[key].dtype or not np.array_equal(existing, arrays[key]):
                raise FileExistsError(f"Choose a fresh figure-input cache; altered file: {destination}")
        else:
            np.save(destination, arrays[key], allow_pickle=False)
        return destination
    raise ValueError(f"Unsupported article plotting input: {filename}")


def cost(sources, dataset, model):
    selected = "results/provenance/selected_runs.json"
    return read_json(_checked_path(sources, selected))[
        f"{dataset}/{model}/none"]["computational_cost_basis"]


def paired_analysis(sources):
    return read_json(_checked_path(sources, "results/controls/paired_analysis.json"))


def figure_output(path=None):
    output = Path(path or ROOT / "runs/article2_figures").resolve()
    for name in ("results", "src", "datasets", "publications"):
        protected = (ROOT / name).resolve()
        if output == protected or protected in output.parents:
            raise ValueError(f"Figure output must not modify the frozen {name} tree.")
    if output == ROOT.resolve():
        raise ValueError("Choose a dedicated output directory, not the checkout root.")
    output.mkdir(parents=True, exist_ok=True)
    return output
