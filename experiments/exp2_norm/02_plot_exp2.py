""" Creates decent plots for experiment 2"""

import os
import glob
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def plot_exp_2(seeds: List[int]):
    """Plot the results of experiment 2"""

    os.makedirs("plots/exp_2_norm", exist_ok=True)

    # Now plot the twins
    # Define the epochs
    epochs = list(range(9, 50, 10)) + [0, 4]

    for ee in epochs:
        twin_acc_data = pd.DataFrame()
        for twin in ["normal", "swapped"]:
            for s in seeds:
                csv_files = glob.glob(
                    f"results/exp_2_norm/{twin}_twin/{ee}/{s}/label_accuracy/*.csv"
                )
                for f in csv_files:
                    epoch = int(os.path.basename(f).split(".")[0])
                    twin_acc_data = pd.concat(
                        [
                            twin_acc_data,
                            pd.read_csv(f).assign(
                                epoch=epoch, seed=s, batch=(epoch + 1) * 56, twin=twin
                            ),
                        ]
                    )
        twin_acc_data.to_csv(f"results/exp_2_norm/twin_accuracy_{ee}.csv", index=False)
        # now make the plot of overall accuracy on the last iteration step
        # select only the last iteration per class/network
        ftwin_acc_data = (
            twin_acc_data.groupby(["seed", "epoch", "class", "network", "twin"])
            .last()
            .reset_index()
        )
        # get average accuracy per batch
        ftwin_acc_data = (
            ftwin_acc_data.groupby(["seed", "batch", "network", "twin"])["accuracy"]
            .mean()
            .reset_index()
        )
        # now get mean and se
        twin_acc = (
            ftwin_acc_data.groupby(["network", "twin", "batch"])["accuracy"]
            .agg(["mean", "sem"])
            .reset_index()
            .set_index(["network", "twin"])
        )

        # plot the twin data
        # The plan here is to plot the different networks in different subplots
        # Within each subplot, each twin gets a different colour

        fig, ax = plt.subplots(1, 3, figsize=(6, 2), sharey=True, layout="constrained")
        batches = twin_acc.batch.unique()
        xticks = [batches[b] for b in range(0, len(batches), 4)]
        for i, network in enumerate(["hybrid", "pc", "amort"]):
            for twin in ["normal", "swapped"]:
                twin_data = twin_acc.loc[(network, twin)]
                ax[i].axhline(1 / 12, color="gray", linestyle="--")
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
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(xticks)
            if i == 0:
                ax[i].set_ylabel("Classification Accuracy")
            # put title for each subplot
            ax[i].set_title(labels[network])
            ax[i].set_xlabel("Batch")
            ax[i].set_ylim(-0.01, 1)
            ax[i].set_yticks(np.arange(0, 1.25, 0.25))
        ax[0].legend(loc="best")
        fig.savefig(
            f"plots/exp_2_norm/twin_accuracy_{ee}.png", bbox_inches="tight", dpi=300
        )

        # plot Local Rec. Error of the twins
        twin_rec_data = pd.DataFrame()

        for twin in ["normal", "swapped"]:
            for s in seeds:
                csv_files = glob.glob(
                    f"results/exp_2_norm/{twin}_twin/{ee}/{s}/reconstruction/*.csv"
                )
                for f in csv_files:
                    epoch = int(os.path.basename(f).split(".")[0])
                    twin_rec_data = pd.concat(
                        [
                            twin_rec_data,
                            pd.read_csv(f).assign(
                                epoch=epoch, seed=s, batch=(epoch + 1) * 56, twin=twin
                            ),
                        ]
                    )

        twin_rec_data.to_csv(
            f"results/exp_2_norm/twin_reconstruction_{ee}.csv", index=False
        )
        # now make the plot of overall rec_error
        # get average rec_error across classes, per batch
        ftwin_rec_data = (
            twin_rec_data.groupby(["seed", "batch", "network", "twin"])["rec_error"]
            .mean()
            .reset_index()
        )
        # now get mean and se
        twin_rec = (
            ftwin_rec_data.groupby(["network", "twin", "batch"])["rec_error"]
            .agg(["mean", "sem"])
            .reset_index()
            .set_index(["network", "twin"])
        )

        # plot the twin data
        # The plan here is to plot the different networks in different subplots
        # Within each subplot, each twin gets a different colour

        fig, ax = plt.subplots(1, 3, figsize=(6, 2), sharey=True, layout="constrained")
        for i, network in enumerate(["hybrid", "pc", "amort"]):
            for twin in ["normal", "swapped"]:
                twin_data = twin_rec.loc[(network, twin)]
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
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(xticks)
            if i == 0:
                ax[i].set_ylabel("Local Rec. Error")
            # put title for each subplot
            ax[i].set_title(labels[network])
            ax[i].set_xlabel("Batch")
            ax[i].set_ylim(0.038, 0.12)
        ax[0].legend(loc="best")
        fig.savefig(
            f"plots/exp_2_norm/twin_reconstruction_{ee}.png",
            bbox_inches="tight",
            dpi=300,
        )


if __name__ == "__main__":
    plot_exp_2(list(range(8)))
