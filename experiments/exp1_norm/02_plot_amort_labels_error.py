"""Creates more decent plots for experiment 1"""

import os
import glob
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

# define DPI
DPI = 600

# get a viridis colour map
cmap = plt.get_cmap("viridis")
# define colour palette for the three networks
colours = {"hybrid": cmap(0.0), "pc": cmap(0.4), "amort": cmap(0.8)}
# define labels for the three networks
labels = {
    "hybrid": "Amort+Inf",
    "pc": "Inf",
    "amort": "Amort",
    "normal": "Control",
    "swapped": "Delusional",
}

# define linetypes for twins
linetypes = {
    "normal": "-",
    "swapped": "--",
}
# define colours for twins
twin_colours = {
    "normal": "#000000",
    "swapped": "#be0119",
}


def plot_exp_1(seeds: List[int]):
    """Plot some results of experiment 1"""
    # Plot the twins
    twin_data = pd.DataFrame()
    for twin in ["normal", "swapped"]:
        for s in seeds:
            csv_files = glob.glob(
                f"results/exp_1_norm/{twin}_twin/{s}/amort_label_error/*.csv"
            )
            for f in csv_files:
                epoch = int(os.path.basename(f).split(".")[0])
                twin_data = pd.concat(
                    [
                        twin_data,
                        pd.read_csv(f).assign(
                            epoch=epoch, seed=s, batch=(epoch + 1) * 56, twin=twin
                        ),
                    ]
                )
    twin_data.to_csv("results/exp_1_norm/twin_amort_label_error.csv", index=False)
    # drop "mean" layer
    twin_data = twin_data[twin_data["layer"] != "mean"]
    # now get mean and se
    twin_aggdata = (
        twin_data.groupby(["twin", "batch", "layer"])["error"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    # plot the twin data
    batches = twin_aggdata.batch.unique()
    layers = twin_aggdata.layer.unique()
    xticks = [batches[b] for b in range(0, len(batches), 4)]

    fig, ax = plt.subplots(
        1, len(layers), figsize=(6, 2), sharey=True, layout="constrained"
    )
    for i, layer in enumerate(layers):
        for twin in ["normal", "swapped"]:
            # select layer and twin
            twin_data = twin_aggdata[
                (twin_aggdata["layer"] == layer) & (twin_aggdata["twin"] == twin)
            ]
            ax[i].plot(
                twin_data["batch"],
                twin_data["mean"],
                label=labels[twin],
                color=twin_colours[twin],
            )
            ax[i].fill_between(
                twin_data["batch"],
                y1=twin_data["mean"] - twin_data["sem"],
                y2=twin_data["mean"] + twin_data["sem"],
                alpha=0.5,
                color=twin_colours[twin],
            )
        # put x tick labels
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xticks)
        if i == 0:
            ax[i].set_ylabel("Prediction Error")
        # put title for each subplot
        ax[i].set_title(f"Layer {layer}")
        ax[i].set_xlabel("Batch")
    ax[i].legend(loc="best")
    fig.savefig(
        "plots/exp_1_norm/twin_amort_label_error.png", bbox_inches="tight", dpi=DPI
    )


if __name__ == "__main__":
    plot_exp_1(list(range(8)))
