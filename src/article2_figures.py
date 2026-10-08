"""Regenerate the three Article 2 figures from its dedicated public evidence.

No private archive, raw dataset or model fitting is required. Numerical inputs
are checked against the release inventory before plotting.
"""

import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from article2_study_overview import main as build_overview
from article2_evidence import figure_output, pin_current_sources, read_run, metrics, threshold_study

MODELS = ["logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm"]
LABELS = ["LR", "RF", "LGBM", "CatBoost", "FT-Trans.", "OCSVM"]
COLOURS = ["#9467bd", "#2ca02c", "#1f77b4", "#d62728", "#ff7f0e", "#7f7f7f"]
DATASETS = [("ulb_2013", "ULB", 95/56746), ("baf_base", "BAF Base", 2206/200000)]
STRATEGIES = ["rus", "ros", "smote", "smote_tomek", "smoteenn", "weights"]
RULES = [("fixed_05", "Fixed 0.5", "o"), ("max_f1", r"max-$F_1$", "^"),
         ("max_f2", r"max-$F_2$", "s"), ("prec_ge_05", "Precision target", "D")]


def save(fig, output, filename):
    fig.savefig(output/filename, bbox_inches="tight", pad_inches=.06)
    fig.savefig((output/filename).with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=.06)


def main(output_dir=None):
    sources = pin_current_sources()
    output = figure_output(output_dir)
    build_overview(sources, output)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42,
                         "font.family": "DejaVu Sans"})

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.85), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, top=.90, bottom=.30, wspace=.15)
    for ax, (dataset, title, prevalence) in zip(axes, DATASETS):
        for model, label, colour in zip(MODELS, LABELS, COLOURS):
            pr = read_run(sources, dataset, model, "none", "pr_curve_data.json")
            ax.plot(pr["recalls"], pr["precisions"], color=colour, lw=1.2,
                    ls="--" if model == "ocsvm" else "-")
            if model != "ocsvm":
                study = threshold_study(sources, dataset, model)["test_results"]
                for key, _, marker in RULES:
                    point = study[key]
                    ax.scatter(point["Recall"], point["Precision"], s=27,
                               marker=marker, color=colour, edgecolors="black",
                               linewidths=.45, zorder=5)
        ax.axhline(prevalence, color="#666666", ls=":", lw=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", title=title)
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Precision")
    model_handles = [Line2D([], [], color=c, lw=1.6, label=l,
                           ls="--" if m == "ocsvm" else "-")
                     for m, l, c in zip(MODELS, LABELS, COLOURS)]
    rule_handles = [Line2D([], [], marker=m, ls="", color="black", markerfacecolor="white",
                          markersize=5, label=l) for _, l, m in RULES]
    fig.legend(handles=model_handles, loc="lower center", bbox_to_anchor=(.5,.112),
               ncol=6, fontsize=8, frameon=False, columnspacing=1.2, handlelength=1.5)
    fig.legend(handles=rule_handles, loc="lower center", bbox_to_anchor=(.5,.048),
               ncol=4, fontsize=8, frameon=False, columnspacing=1.5)
    fig.text(.5, .014, "Dotted line: TEST prevalence. Points: saved baseline threshold-study results.",
             ha="center", fontsize=8)
    save(fig, output, "pr_curves_threshold.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    fig.subplots_adjust(left=.09, right=.89, top=.88, bottom=.24, wspace=.35)
    for ax, (dataset, title, _) in zip(axes, DATASETS):
        values = []
        for model in MODELS[:-1]:
            baseline = metrics(sources, dataset, model)["PR-AUC"]
            values.append([metrics(sources, dataset, model, st)["PR-AUC"]-baseline
                           for st in STRATEGIES])
        values = np.asarray(values)
        im = ax.imshow(values, cmap="RdYlGn", vmin=-.27, vmax=.27, aspect="auto")
        ax.set_xticks(range(6), ["RUS", "ROS", "SMOTE", "SM+T", "SM+ENN", "Weights"],
                      rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(5), LABELS[:-1])
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=11)
        for row in range(5):
            for col in range(6):
                value = values[row,col]
                ax.text(col,row,f"{value:+.3f}",ha="center",va="center",fontsize=7.5,
                        color="white" if abs(value)>.16 else "#18252c")
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.92,.24,.015,.64])
    cb = fig.colorbar(im,cax=cax)
    cb.set_label("PR-AUC change",fontsize=9)
    fig.text(.5,.025,"Common colour scale; changes from each model's None baseline.",
             ha="center",fontsize=8)
    save(fig, output, "imbalance_heatmaps.pdf")
    plt.close(fig)
    print("Generated the three manuscript vector PDFs and PNG copies; no models fitted.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="New figure destination; defaults to runs/article2_figures")
    main(parser.parse_args().output_dir)
