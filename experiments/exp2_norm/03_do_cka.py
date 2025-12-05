"""Perform CKA on experiment 2."""

from typing import List
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pybrid.postprocessing import do_cka

DO_CKA = False
FIG_SIZE = (4.25, 3)

labels = {
    "hybrid": "Amort+Inf",
    "pc": "Inf",
    "amort": "Amort",
    "Progenitor": "Progenitor",
    "Normal": "Control",
    "Swapped": "Delusional",
    "Progenitor:Normal": "Control",
    "Progenitor:Swapped": "Delusional",
}
# define colours for twins
twin_colours = {
    "Progenitor:Normal": "#000000",
    "Progenitor:Swapped": "#be0119",
}


def cka_exp(seeds: List[int]):
    """Perform CKA"""
    prog_folder = os.path.join(
        "results/exp_1_norm/progenitor"
    )  # progenitor comes from experiment 1
    normal_folder = os.path.join("results/exp_2_norm/normal_twin")
    swapped_folder = os.path.join("results/exp_2_norm/swapped_twin")
    out_dir = os.path.join("results/exp_2_norm/cka")
    model_names = ["Progenitor", "Normal", "Swapped"]

    all_ckas = pd.DataFrame()
    for e in [0, 4] + list(range(9, 50, 10)):
        for seed in seeds:
            pkl_names = [
                os.path.join(prog_folder, str(seed), "features/99.pkl"),
                os.path.join(normal_folder, str(e), str(seed), "features/49.pkl"),
                os.path.join(swapped_folder, str(e), str(seed), "features/49.pkl"),
            ]
            cka_res = do_cka(
                feature_pkls=pkl_names,
                model_names=model_names,
            )
            # add epoch and batch
            cka_res["epoch"] = e
            cka_res["batch"] = (
                (e + 1) * 56 if e != 0 else 0
            )  # The 0 is no delusion phase
            # add seed
            cka_res["seed"] = seed
            # add to all_ckas
            all_ckas = pd.concat([all_ckas, cka_res], ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    all_ckas.to_csv(os.path.join(out_dir, "cka_all.csv"), index=False)


def plot_cka(cka_file: str, out_file: str = "cka_summary_across.svg"):
    all_ckas = pd.read_csv(cka_file)
    # aggregate
    all_ckas = (
        all_ckas.groupby(["comparison", "network", "layer", "epoch", "batch"])
        .agg(
            mean=("cka", "mean"),
            sem=("cka", "sem"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(
        3, 3, figsize=FIG_SIZE, sharex=True, sharey="row", layout="constrained"
    )
    range_mins = [0.40, 0.40, 0.95]
    range_maxs = [1.01, 1.01, 1.01]
    xlabs = all_ckas["batch"].unique()
    xticks = [0, 3, 6]
    xlabs = [xlabs[x] for x in xticks]
    xticks = all_ckas["batch"].unique()[xticks]
    yticks = [[0.6, 0.8, 1.0], [0.6, 0.8, 1.0], [1.0, 0.9]]

    for layer in [0, 1, 2]:
        for j, net in enumerate(["hybrid", "pc", "amort"]):
            # get data
            dat = all_ckas[(all_ckas["network"] == net) & (all_ckas["layer"] == layer)]
            ax = axes[layer, j]
            for comp in ["Progenitor:Normal", "Progenitor:Swapped"]:
                # get data for this comp
                dat_comp = dat[dat["comparison"] == comp]
                # plot
                ax.plot(
                    dat_comp["batch"],
                    dat_comp["mean"],
                    label=labels[comp],
                    color=twin_colours[comp],
                )
                # add ribbon for sem
                ax.fill_between(
                    dat_comp["batch"],
                    dat_comp["mean"] - dat_comp["sem"],
                    dat_comp["mean"] + dat_comp["sem"],
                    alpha=0.2,
                    color=twin_colours[comp],
                )
                ax.set_ylim(range_mins[layer], range_maxs[layer])
                ax.set_yticks(yticks[layer])
                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabs)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

            if j == 0 & layer == 0:
                ax.set_ylabel(f"Layer {3-layer}\nCKA")

            # put network names on the title
            if layer == 0:
                ax.set_title(labels[net])
            if layer == 2:
                ax.set_xlabel("Batch")

    # add legend to bottom left panel
    axes[2, 0].legend(
        loc="best",
        fontsize=8,
    )

    fig.savefig(out_file)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    if DO_CKA:
        cka_exp(seeds=list(range(8)))
    plot_cka(
        "results/exp_2_norm/cka/cka_all.csv",
        out_file="plots/exp_2_norm/cka_summary_across.svg",
    )
