"""Regenerate Article 2's study overview from its verified public predictions.

No private archive, raw dataset, model fitting or threshold selection is used.
Only the selected output directory and runs/figure_inputs cache are written.
"""

import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from article2_evidence import figure_output, pin_current_sources, read_run, threshold_study

DATASETS = {"ULB": "ulb_2013", "BAF Base": "baf_base"}


def main(sources=None, output_dir=None):
    sources = sources or pin_current_sources()
    output = figure_output(output_dir)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 17,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#24323c", "text.color": "#24323c",
        "xtick.color": "#586875", "ytick.color": "#586875",
    })
    colours = {"ULB": "#12566c", "BAF Base": "#b72131"}
    green, orange = "#207644", "#bd741a"
    fig = plt.figure(figsize=(15, 7.6), facecolor="white")
    fig.text(.025, .954, "From fraud scores to validated alert decisions",
             fontsize=25, weight="bold")
    fig.text(.025, .894, "1  Inspect saved fraud scores", fontsize=19,
             weight="bold", color=colours["ULB"])
    fig.text(.365, .894, "2  Select within DEV only", fontsize=19,
             weight="bold", color=orange)
    fig.text(.695, .894, "3  Quantify the trade-off", fontsize=19,
             weight="bold", color=green)

    bins = np.linspace(0, 1, 21)
    for label, bottom in [("ULB", .61), ("BAF Base", .30)]:
        dataset = DATASETS[label]
        y = np.load(read_run(sources, dataset, "fttransformer", "none", "y_test.npy"), allow_pickle=False)
        scores = np.load(read_run(sources, dataset, "fttransformer", "none", "y_test_scores.npy"), allow_pickle=False)
        positive_scores = scores[y == 1]
        assert len(y) == len(scores) and np.isfinite(scores).all()
        assert np.all((scores >= 0) & (scores <= 1))
        ax = fig.add_axes([.075, bottom, .235, .19])
        ax.hist(positive_scores, bins=bins,
                weights=np.ones(len(positive_scores))/len(positive_scores),
                color=colours[label], edgecolor="white", linewidth=.7)
        ax.axvline(.5, color=orange, ls="--", lw=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(labelsize=14)
        ax.set_title(f"{label}: {len(positive_scores):,} TEST frauds",
                     fontsize=17, loc="left", color=colours[label], pad=10)
        if label == "BAF Base":
            ax.set_xlabel("Predicted fraud score", fontsize=16)
        else:
            ax.text(.52, .86, r"$\tau=0.5$", color=orange, fontsize=16)
    fig.text(.025, .145, "FT-Transformer, None baseline\nShare of fraud cases in each score bin",
             fontsize=15, linespacing=1.5)

    middle = fig.add_axes([.355, .165, .3, .655])
    middle.set_axis_off()
    boxes = [
        (.86, "80% DEV / 20% TEST", "ULB: deduplicate before split\nStratified holdout, seed 42"),
        (.55, "Model-specific validation", "Classical / OCSVM: five folds\nFT-Transformer: internal holdout"),
        (.23, "Freeze the operating rules", "Reuse saved validation scores\nFour rules, unchanged TEST ranking"),
    ]
    for yy, heading, detail in boxes:
        middle.text(.03, yy, heading, weight="bold", fontsize=18,
                    va="top", transform=middle.transAxes)
        middle.text(.03, yy-.085, detail, fontsize=16, va="top",
                    linespacing=1.5, transform=middle.transAxes)
    for yy in [.63, .29]:
        middle.annotate("", xy=(.5, yy), xytext=(.5, yy+.045),
                        arrowprops={"arrowstyle": "->", "color": orange, "lw": 2},
                        xycoords="axes fraction")
    fig.text(.367, .145, "TEST excluded from selection",
             fontsize=15, color=orange, weight="bold")

    study = threshold_study(sources, "baf_base", "fttransformer")
    fixed = study["test_results"]["fixed_05"]
    chosen = study["test_results"]["max_f2"]
    fig.text(.695, .835, "BAF Base / FT-Transformer", fontsize=17)
    for key, bottom, limit, title in [
        ("F2", .55, .4, r"$F_2$ score"),
        ("alert_rate", .25, 7, "Alert rate (%)"),
    ]:
        ax = fig.add_axes([.755, bottom, .213, .18])
        vals = [fixed[key], chosen[key]]
        if key == "alert_rate":
            vals = [100*v for v in vals]
        ax.barh([1, 0], vals, height=.48, color=[colours["BAF Base"], green])
        ax.set_xlim(0, limit)
        ax.set_ylim(-.6, 1.6)
        ax.set_yticks([1, 0], ["Fixed", r"max-$F_2$"])
        ax.tick_params(labelsize=14)
        ax.set_xticks([0, limit/2, limit])
        ax.set_title(title, loc="left", fontsize=18, pad=10)
        for yy, value in zip([1, 0], vals):
            ax.text(value+limit*.025, yy, f"{value:.3f}", va="center", fontsize=16)
    fig.text(.695, .145, f"Recall: {fixed['Recall']:.1%} to {chosen['Recall']:.1%}",
             fontsize=17, weight="bold", color=green)
    fig.text(.025, .054, "Validate the operating point and its workload; do not infer decisions from ranking alone.",
             fontsize=20, weight="bold")
    out = output / "study_overview.pdf"
    fig.savefig(out, metadata={"Title": "Study overview from saved fraud-detection results",
                              "Author": "Olavo Caixeiro et al."})
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print(out)
    print(f"Fixed alerts: {fixed['alert_rate']*200000:.0f}; max-F2 alerts: {chosen['alert_rate']*200000:.0f}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="New figure destination; defaults to runs/article2_figures")
    main(output_dir=parser.parse_args().output_dir)
