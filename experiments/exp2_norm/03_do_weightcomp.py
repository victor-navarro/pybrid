"""Perform weightcomp on experiment 2."""

from typing import List
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pybrid.postprocessing import compare_models

DPI = 600
DO_COMP = False
FIG_SIZE = (6, 3)
METRIC = "l2-norm"

labels = {
    "hybrid": "Amort+Inf",
    "pc": "Inf",
    "amort": "Amort",
    "Progenitor": "Progenitor",
    "Normal": "Control",
    "Swapped": "Delusional",
    "Progenitor:Normal": "Control",
    "Progenitor:Swapped": "Delusional",
    'Normal:Swapped': "Del & Ctrl",
    "inference": "Inf Network",
    "amort_0": "Digits Amort",
    "amort_1": "Letters Amort",
    "l2-norm": "L2 to Prog",
    "cos": "Cos. Sim.",
}
# define colours for twins
twin_colours = {
    "Progenitor:Normal": "#000000",
    "Progenitor:Swapped": "#be0119",
}


def comp_exp(seeds: List[int]):
    """Perform weightcomp"""
    prog_folder = os.path.join("results/exp_1_norm/progenitor") # progenitor comes from experiment 1
    normal_folder = os.path.join("results/exp_2_norm/normal_twin")
    swapped_folder = os.path.join("results/exp_2_norm/swapped_twin")
    out_dir = os.path.join("results/exp_2_norm/weightcomp")
    model_names = ["Progenitor", "Normal", "Swapped"]

    all_comps = pd.DataFrame()
    for e in [0, 4] + list(range(9, 50, 10)):
        for seed in seeds:
            pkl_names = [os.path.join(prog_folder, str(seed), "model_99.pkl"),
                        os.path.join(normal_folder, str(e), str(seed), f"model_49.pkl"), 
                        os.path.join(swapped_folder, str(e), str(seed), f"model_49.pkl")]
            
            comps = compare_models(
                model_pkls = pkl_names,
                model_names = model_names,
            )
            # add epoch and batch
            comps["epoch"] = e
            comps["batch"] = (e + 1) * 56 if e != 0 else 0 # The 0 is no delusion phase
            # add seed
            comps["seed"] = seed
            # add to all_comps
            all_comps = pd.concat([all_comps, comps], ignore_index=True)
    # save all_comps
    os.makedirs(out_dir, exist_ok=True)
    all_comps.to_csv(os.path.join(out_dir, "weightcomp_all.csv"), index=False)

def plot_comp(comp_file:str, out_file: str = "weightcomp.png"):
    # read csv with weightcomp results
    all_comps = pd.read_csv(comp_file)
    # summarise
    all_comps = all_comps.groupby(["metric", "layer", "comparison", "network", "epoch", "batch"]).agg(
        mean=("value", "mean"),
        sem = ("value", "sem"),
    ).reset_index()
    all_comps = all_comps[all_comps["metric"] == METRIC]

    # now plot the weightcomp results
    # the figure will have 2 by 3 panels
    # rows are normal-progenitor and swapped-progenitor
    # columns are the networks
    fig, axes = plt.subplots(3, 3, figsize=FIG_SIZE, sharex=True, sharey=True, layout="constrained")
    range_min = -.25
    range_max = all_comps["mean"].max()+.25
    xlabs = (all_comps["batch"].unique()) 
    xticks = range(0, len(all_comps["batch"].unique()), 2)
    xlabs = [xlabs[x] for x in xticks]
    xticks = all_comps["batch"].unique()[xticks]

    
    for layer in [0, 1, 2]:
        for j, net in enumerate(["inference", "amort_0", "amort_1"]):
            # get data
            dat = all_comps[(all_comps["network"] == net) & 
                            (all_comps["layer"] == layer)]
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
                    alpha=0.5,
                    color=twin_colours[comp],
                )
                ax.set_ylim(range_min, range_max)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabs)

            # put network names on the y axis
            if j == 0 & layer == 0:
                ax.set_ylabel(f"Layer{3-layer}\n{labels[METRIC]}")

            # put network names on the title
            if layer == 0:
                ax.set_title(labels[net])
            if layer == 2:
                ax.set_xlabel("Batch")


    # add legend
    axes[1, 2].legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
        fontsize=8,
    )

    fig.savefig(out_file, dpi=DPI)
    plt.close(fig)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    if DO_COMP:
        comp_exp(seeds=list(range(8)))
    plot_comp("results/exp_2_norm/weightcomp/weightcomp_all.csv", 
              "plots/exp_2_norm/weightcomp_summary_across.png")
