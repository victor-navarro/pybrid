"""Perform RSA on experiment 1."""

from typing import List
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from pybrid.postprocessing import do_rsa
from pybrid.utils import load_pkl

DO_RSA = False


def rsa_exp(output_dir: str, seeds: List[int]):
    """Process the results of experiment 1."""
    prog_folder = os.path.join(output_dir, "progenitor")
    normal_folder = os.path.join(output_dir, "normal_twin")
    swapped_folder = os.path.join(output_dir, "swapped_twin")
    pretty_names = ["0", "1", "2", "3", "4", "5", "A", "B", "C", "D", "E", "S"]

    feature_folders = [prog_folder, normal_folder, swapped_folder]
    model_names = ["Progenitor", "Normal", "Swapped"]
    pkl_names = ["99.pkl", "49.pkl", "49.pkl"]

    if DO_RSA:
        do_rsa(
            feature_folders=feature_folders,
            seeds=seeds,
            model_names=model_names,
            pkl_names=pkl_names,
            class_names=pretty_names,
            output_dir=output_dir,
        )

    # unpickle correlation results
    all_corrs = load_pkl(os.path.join(output_dir, "rsa/rsa_summary.pkl"))

    # now plot the RSA results
    # the figure will have one row per network, and one column per layer
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), layout="constrained")
    # calculate highest and lowest correlations in all_corrs
    max_corr = np.max(all_corrs["mean"])
    min_corr = np.min(all_corrs["mean"])
    network_labels = {"hybrid": "Hybrid", "pc": "PC", "amort": "Amort"}

    for i, ax in enumerate(axes.flat):
        im = ax.imshow(all_corrs["mean"][i], vmin=min_corr, vmax=max_corr)
        # put means in the middle of the squares
        for j in range(3):
            for k in range(3):
                ax.text(
                    j,
                    k,
                    f"{all_corrs['mean'][i][j, k]:.2f}",
                    color="black",
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.5,
                        edgecolor="none",
                        boxstyle="round",
                    ),
                )
        # put network names on the y axis
        if i % 4 == 0:
            ax.set_ylabel(network_labels[all_corrs["nets"][i]])
        # put layer names on the title
        if i < 4:
            ax.set_title(f"Layer {all_corrs['layers'][i]}")

        # put model_names on both x and y axes
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(model_names)
        ax.set_yticklabels(model_names)

    # save the figure
    fig.savefig(os.path.join(output_dir, "rsa/rsa_summary.png"))
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    rsa_exp(output_dir="results/exp_1/", seeds=list(range(8)))
