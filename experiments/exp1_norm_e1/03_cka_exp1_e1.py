"""Perform CKA on experiment 1."""

from typing import List
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pybrid.postprocessing import do_cka

DPI = 600
DO_CKA = False
FIG_SIZE = (6, 3)

labels = {
    "hybrid": "Amort+Inf",
    "pc": "Inf",
    "amort": "Amort",
    "Progenitor": "Progenitor",
    "Normal": "Control",
    "Swapped": "Delusional",
    "Progenitor:Normal": "Ctrl & Prog",
    "Progenitor:Swapped": "Del & Prog",
}

def cka_exp(output_dir: str, seeds: List[int]):
    """Perform CKA"""
    prog_folder = os.path.join(output_dir, "progenitor")
    normal_folder = os.path.join(output_dir, "normal_twin")
    swapped_folder = os.path.join(output_dir, "swapped_twin")
    model_names = ["Progenitor", "Normal", "Swapped"]

    all_ckas = pd.DataFrame()
    for seed in seeds:
        for e in [0, 1, 5, 10, 20, 30, 40, 50, 55]:
            pkl_names = [os.path.join(prog_folder, str(seed), "features/99.pkl"), 
                        os.path.join(normal_folder, str(seed), f"features/{e}.pkl"), 
                        os.path.join(swapped_folder, str(seed), f"features/{e}.pkl")]
            cka_res = do_cka(
                feature_pkls=pkl_names,
                model_names=model_names,
            )
            # add epoch and batch
            cka_res["epoch"] = 0
            cka_res["batch"] = e
            # add seed
            cka_res["seed"] = seed
            # add to all_ckas
            all_ckas = pd.concat([all_ckas, cka_res], ignore_index=True)
    all_ckas.to_csv(os.path.join(output_dir, "cka", "cka_all.csv"), index=False)



def plot_cka(cka_file: str, out_file: str = "cka_summary_across.png"):
    # now plot the CKA results
    # the figure will have 2 by 3 panels
    # rows are normal-progenitor and swapped-progenitor
    # columns are hybrid, pc, amort
    # each panel contains a matrix of layer by batch CKA
    all_ckas = pd.read_csv(cka_file)
    # aggregate
    all_ckas = all_ckas.groupby(["comparison", "network", "layer", "epoch", "batch"]).agg(
        mean=("cka", "mean"),
        sem=("cka", "sem"),
    ).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE, layout="constrained")
    range_min = 0
    range_min = all_ckas["mean"].min()
    range_max = 1
    xticks = range(0, len(all_ckas["batch"].unique()), 2)
    xlabs = (all_ckas["batch"].unique()) 
    xlabs = [xlabs[x] for x in xticks]
    
    
    for i, comp in enumerate(["Progenitor:Swapped", "Progenitor:Normal"]):
        for j, net in enumerate(["hybrid", "pc", "amort"]):
            # get data
            dat = all_ckas[(all_ckas["network"] == net) & (all_ckas["comparison"] == comp)]
            ax = axes[i, j]

            for layer in dat["layer"].unique():
                # get data for this layer
                dat_layer = dat[dat["layer"] == layer]
                # plot
                ax.plot(
                    dat_layer["batch"],
                    dat_layer["mean"],
                    label=3-layer,
                )
                ax.set_ylim(range_min-.01, range_max+.01)

            if j == 0 & i == 0:
                ax.set_ylabel(f"{labels[comp]}\nCKA")

            # put network names on the title
            if i == 0:
                ax.set_title(labels[net])
            if i == 1:
                ax.set_xlabel("Batch")

    # add legend
    axes[0, 2].legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
        title="Layer",
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
    if DO_CKA:
        cka_exp(output_dir="results/exp_1_norm_e1/", seeds=list(range(1)))
    plot_cka("results/exp_1_norm_e1/cka/cka_all.csv",
              "plots/exp_1_norm_e1/cka_summary_across.png")
