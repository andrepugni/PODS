import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib.ticker import FormatStrFormatter

wdir = os.path.abspath(os.path.dirname(__name__))
img_fold = "C:/Users/andre/Dropbox/Applicazioni/Overleaf/Ideas for RDD in Abstaining classifiers/figs"
style = "seaborn-v0_8-whitegrid"
white_back = False
# general settings
plt.style.use(style)
plt.rc("font", size=20)
plt.rc("legend", fontsize=20)
plt.rc("lines", linewidth=2)
plt.rc("axes", linewidth=2)
plt.rc("axes", edgecolor="k")
plt.rc("xtick.major", width=2)
plt.rc("xtick.major", size=15)
plt.rc("ytick.major", width=2)
plt.rc("ytick.major", size=15)


if style == "fivethirtyeight" and white_back == True:
    style += "WHITE"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = False

markers_dict = {
    "CompareConfidence": "s",
    "DifferentiableTriage": "d",
    "LceSurrogate": "^",
    "MixtureOfExperts": "*",
    "OVASurrogate": "H",
    "RealizableSurrogate": "o",
    "SelectivePrediction": "X",
}


cl_palette = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
]

palette = {
    q: sns.color_palette("colorblind")[i]
    for i, q in enumerate(sorted(markers_dict.keys(), reverse=True))
}


def plot_single_classifier(df, classifier, figsize=(20, 12), title=""):
    db = df[df["method"] == classifier].copy()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    classifiers = db["method"].unique()
    target_covs = db["target_coverage"].unique()
    for c in classifiers:
        db_c = db[db["method"] == c].copy()
        (_, caps, _) = ax.errorbar(
            db_c["target_coverage"],
            db_c["coef_rob"],
            label=c,
            color=palette[c],
            yerr=(db_c["ci_rob_u"] - db_c["ci_rob_l"]) / 2,
            fmt="o",
            alpha=0.9,
            capsize=10,
            elinewidth=4,
            marker=markers_dict[c],
            markersize=20,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        for cap in caps:
            cap.set_markeredgewidth(8)
    ax.hlines(y=0, linestyles="dashed", colors="black", xmin=0, xmax=1, linewidth=4)
    ax.legend(loc="upper left", fontsize=22, markerscale=1, fancybox=True, shadow=False)
    ax.set_xlabel("Target coverage", fontdict={"fontsize": 20})
    ax.set_ylabel("Estimated $\\tau$", fontdict={"fontsize": 20})
    ax.set_yticklabels(["{:.2f}".format(x) for x in ax.get_yticks()], fontsize=18)
    ax.set_xticks(target_covs)
    ax.set_xticklabels(["{:.2f}".format(x) for x in target_covs], fontsize=18)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.75, 0.75)
    ax.set_title(title, fontsize=20)
    plt.savefig(
        "{}/{}_{}_allcoeffs.png".format(img_fold, title, classifier),
        bbox_inches="tight",
        dpi=300,
    )
    return ax


def plot_single_classifier_two_ys(df, classifier, figsize=(20, 12), title=""):
    db = df[df["method"] == classifier].copy()
    classifiers = db["method"].unique()
    target_covs = db["target_coverage"].unique()
    ax = plot_single_classifier(df, classifier, figsize=figsize, title=title)
    ax2 = ax.twinx()
    for c in classifiers:
        db_c = db[db["method"] == c].copy()
        ax2.plot(
            db_c["target_coverage"],
            db_c["system_acc"],
            label=c,
            color="grey",
            alpha=0.4,
            marker=".",
            markersize=20,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
    ax2.set_ylabel(
        "System accuracy", fontdict={"fontsize": 20}, rotation=-90, labelpad=30
    )
    ax2.set_yticks(
        sorted([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, db["classifier_all_acc"].max()])
    )
    ax2.set_yticklabels(["{:.2f}".format(x) for x in ax2.get_yticks()], fontsize=18)
    ax2.set_ylim(0.7, 1)
    ax2.hlines(
        y=db["classifier_all_acc"].max(),
        linestyles="dotted",
        colors="grey",
        xmin=0,
        xmax=1,
        linewidth=4,
    )
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    plt.savefig(
        "{}/{}_{}_allcoeffs.png".format(img_fold, title, classifier),
        bbox_inches="tight",
        dpi=300,
    )
    return ax


def plot_single_classifier_two_ys_V2(df, classifier, figsize=(20, 12), title=""):
    db = df[df["method"] == classifier].copy()
    classifiers = db["method"].unique()
    target_covs = db["target_coverage"].unique()
    ax = plot_single_classifier(df, classifier, figsize=figsize, title=title)
    ax2 = ax.twinx()
    for c in classifiers:
        db_c = db[db["method"] == c].copy()
        ax2.plot(
            db_c["target_coverage"],
            db_c["system_acc"] / db_c["classifier_all_acc"] - 1,
            label=c,
            color="grey",
            alpha=0.4,
            marker=".",
            markersize=20,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
    ax2.set_ylabel(
        "Accuracy Increase", fontdict={"fontsize": 20}, rotation=-90, labelpad=30
    )
    # ax2.set_yticks(sorted([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, db["classifier_all_acc"].max()]))
    ax2.set_ylim(
        -(db_c["system_acc"] / db_c["classifier_all_acc"]).max() + 0.95,
        (db_c["system_acc"] / db_c["classifier_all_acc"]).max() - 0.95,
    )
    ax2.set_yticks(
        np.linspace(
            -(db_c["system_acc"] / db_c["classifier_all_acc"]).max() + 0.95,
            (db_c["system_acc"] / db_c["classifier_all_acc"]).max() - 0.95,
            5,
        )
    )
    ax2.set_yticklabels(
        [
            "{:.1%}".format(x)
            if x != 0
            else "{:.2f}".format(db_c["classifier_all_acc"].max())
            for x in ax2.get_yticks()
        ],
        fontsize=18,
    )

    # ax2.hlines(y=db["classifier_all_acc"].max(), linestyles='dotted', colors='grey', xmin=0, xmax=1, linewidth=4)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    plt.savefig(
        "{}/{}_{}_allcoeffs.png".format(img_fold, title, classifier),
        bbox_inches="tight",
        dpi=300,
    )
    return ax
