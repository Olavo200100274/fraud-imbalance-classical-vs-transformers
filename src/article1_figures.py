"""Regenerate the seven Article 1 figures from the verified saved evidence."""
import argparse
from pathlib import Path

import numpy as np

from article_release import read_json, verify
from article_interpretability import regenerate

ROOT = Path(__file__).resolve().parents[1]


def main(output=None):
    manifest = verify(ROOT)
    if manifest["article"] != 1:
        raise ValueError("This figure workflow requires the Article 1 release")
    output = Path(output or ROOT / "runs/article1_figures").resolve()
    if output == ROOT or any(output.is_relative_to(ROOT / name) for name in
                             ("results", "src", "tests", "datasets", "notebooks", "publications", ".git")):
        raise ValueError("Choose a new isolated figure output directory")
    paths = regenerate(ROOT, output)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    strategies = ("none", "rus", "ros", "smote", "smote_tomek", "smoteenn", "weights")
    labels = ("None", "RUS", "ROS", "SMOTE", "SM+T", "SMENN", "Weights")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), constrained_layout=True)
    for axis, dataset, title in zip(axes, ("ulb_2013", "baf_base"), ("ULB: distinct profiles", "BAF Base")):
        for offset, model, label, colour in ((-.18, "fttransformer", "FT-Transformer", "#ff7f0e"), (.18, "catboost", "CatBoost", "#d62728")):
            values = [read_json(ROOT / manifest["primary_runs"][f"{dataset}/{model}/{strategy}"]["metrics"])["PR-AUC"] for strategy in strategies]
            axis.bar(np.arange(7) + offset, values, width=.35, label=label, color=colour)
        axis.set_xticks(range(7), labels, rotation=35, ha="right")
        axis.set(title=title, ylabel="Average precision", ylim=(0, .9 if dataset == "ulb_2013" else .21))
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    path = output / "smote_asymmetry.pdf"
    fig.savefig(path, bbox_inches="tight")
    paths.append(path)
    plt.close(fig)
    models = (("logreg", "LR", "#9467bd"), ("rf", "RF", "#2ca02c"), ("lgbm", "LGBM", "#1f77b4"),
              ("catboost", "CatBoost", "#d62728"), ("fttransformer", "FT-Transformer", "#ff7f0e"), ("ocsvm", "OCSVM", "#777777"))
    fig, axis = plt.subplots(figsize=(7.8, 4.2), constrained_layout=True)
    for model, label, colour in models:
        values = [read_json(ROOT / manifest["primary_runs"][f"baf_base/{model}/none"]["metrics"])["PR-AUC"]]
        values += [read_json(ROOT / manifest["transfer_runs"][f"{model}/baf_var{number}"]["metrics"])["metrics_full_precision"]["average_precision"] for number in range(1, 6)]
        axis.plot(range(6), values, color=colour, marker="o", label=label)
    axis.set_xticks(range(6), ("Base", "Variant I", "Variant II", "Variant III", "Variant IV", "Variant V"))
    axis.set(ylabel="Average precision", ylim=(0, .21))
    axis.grid(alpha=.2)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    path = output / "crossdomain_prauc.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    print(f"Regenerated {len(paths)} figures from saved evidence; no fitting or inference.")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    main(parser.parse_args().output_dir)
