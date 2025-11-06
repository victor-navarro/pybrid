""" Creates decent plots for experiment 1"""

import os
import glob
from typing import List
import pandas as pd
import numpy as np
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
    "Progenitor:Normal": "#000000",
    "Progenitor:Swapped": "#be0119",
}

def plot_exp_1(seeds: List[int]):
    """Plot the results of experiment 1"""

    # plot classification accuracy of the progenitor model
    acc_data = pd.DataFrame()
    for s in seeds:
        # get all csv files in the folder
        csv_files = glob.glob(f"results/exp_1_norm/progenitor/{s}/label_accuracy/*.csv")
        # load the data and append epoch (leading digits in filename) as a column
        for f in csv_files:
            epoch = int(os.path.basename(f).split(".")[0])
            acc_data = pd.concat(
                [
                    acc_data,
                    pd.read_csv(f).assign(epoch=epoch, seed=s, batch=(epoch + 1) * 56),
                ]
            )

    acc_data.to_csv("results/exp_1_norm/progenitor_accuracy.csv", index=False)
    # now make the plot of overall accuracy on the last iteration step
    # select only the last iteration per class/network
    facc_data = (
        acc_data.groupby(["seed", "epoch", "class", "network"]).last().reset_index()
    )
    # get average accuracy per batch
    facc_data = (
        facc_data.groupby(["seed", "batch", "network"])["accuracy"].mean().reset_index()
    )
    # now get mean and se
    prog_acc = (
        facc_data.groupby(["network", "batch"])["accuracy"]
        .agg(["mean", "sem"])
        .reset_index()
        .set_index("network")
    )

    # plot the data
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.axhline(1 / 12, color="gray", linestyle="--")
    batches = prog_acc.batch.unique()
    xticks = [batches[b] for b in range(0, len(batches), 4)]
    # plot accuracy as a function of batch number with standard error
    # networks are represented by different colors
    for network in ["hybrid", "pc", "amort"]:
        ax.plot(
            prog_acc.loc[network]["batch"],
            prog_acc.loc[network]["mean"],
            label=labels[network],
            color=colours[network],
        )
        ax.fill_between(
            prog_acc.loc[network]["batch"],
            y1=prog_acc.loc[network]["mean"] - prog_acc.loc[network]["sem"],
            y2=prog_acc.loc[network]["mean"] + prog_acc.loc[network]["sem"],
            alpha=0.5,
            color=colours[network],
        )
        # put x tick labels
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
    # add legend and labels
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Batch")
    ax.set_ylabel("Classification Accuracy")
    # set limits
    ax.set_ylim(-0.01, 1)
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    os.makedirs("plots/exp_1_norm/", exist_ok=True)
    fig.savefig(
        "plots/exp_1_norm/progenitor_accuracy.png", bbox_inches="tight", dpi=300
    )

    # plot reconstruction error of the progenitor model
    rec_data = pd.DataFrame()
    for s in seeds:
        # get all csv files in the folder
        csv_files = glob.glob(f"results/exp_1_norm/progenitor/{s}/reconstruction/*.csv")
        # load the data and append epoch (leading digits in filename) as a column
        for f in csv_files:
            epoch = int(os.path.basename(f).split(".")[0])
            rec_data = pd.concat(
                [
                    rec_data,
                    pd.read_csv(f).assign(epoch=epoch, seed=s, batch=(epoch + 1) * 56),
                ]
            )
    rec_data.to_csv("results/exp_1_norm/progenitor_reconstruction.csv", index=False)
    # now make the plot of overall rec_error
    # get average rec_error across classes, per batch
    frec_data = (
        rec_data.groupby(["seed", "batch", "network"])["rec_error"].mean().reset_index()
    )
    # now get mean and se
    prog_rec = (
        frec_data.groupby(["network", "batch"])["rec_error"]
        .agg(["mean", "sem"])
        .reset_index()
        .set_index("network")
    )

    # plot the data
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    for network in ["hybrid", "pc", "amort"]:
        ax.plot(
            prog_rec.loc[network]["batch"],
            prog_rec.loc[network]["mean"],
            label=labels[network],
            color=colours[network],
        )
        ax.fill_between(
            prog_rec.loc[network]["batch"],
            y1=prog_rec.loc[network]["mean"] - prog_rec.loc[network]["sem"],
            y2=prog_rec.loc[network]["mean"] + prog_rec.loc[network]["sem"],
            alpha=0.5,
            color=colours[network],
        )
        # put x tick labels
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
    # add legend and labels
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Batch")
    ax.set_ylabel("Reconstruction Error")
    # set limits
    ax.set_ylim(0.00, 0.90)
    fig.savefig(
        "plots/exp_1_norm/progenitor_reconstruction.png", bbox_inches="tight", dpi=DPI
    )

    # Now plot the twins
    twin_acc_data = pd.DataFrame()
    for twin in ["normal", "swapped"]:
        for s in seeds:
            csv_files = glob.glob(
                f"results/exp_1_norm/{twin}_twin/{s}/label_accuracy/*.csv"
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
    twin_acc_data.to_csv("results/exp_1_norm/twin_accuracy.csv", index=False)
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
        # put x tick labels
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xticks)
        if i == 0:
            ax[i].set_ylabel("Classification Accuracy")
        # put title for each subplot
        ax[i].set_title(labels[network])
        ax[i].set_xlabel("Batch")
        ax[i].set_ylim(-0.01, 1)
        ax[i].set_yticks(np.arange(0, 1.25, 0.25))
    ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.savefig("plots/exp_1_norm/twin_accuracy.png", bbox_inches="tight", dpi=DPI)

    # plot reconstruction error of the twins
    twin_rec_data = pd.DataFrame()

    for twin in ["normal", "swapped"]:
        for s in seeds:
            csv_files = glob.glob(
                f"results/exp_1_norm/{twin}_twin/{s}/reconstruction/*.csv"
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

    twin_rec_data.to_csv("results/exp_1_norm/twin_reconstruction.csv", index=False)
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
        # put x tick labels
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xticks)
        if i == 0:
            ax[i].set_ylabel("Reconstruction Error")
        # put title for each subplot
        ax[i].set_title(labels[network])
        ax[i].set_xlabel("Batch")
        ax[i].set_ylim(0.05, 0.10)
    ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.savefig(
        "plots/exp_1_norm/twin_reconstruction.png", bbox_inches="tight", dpi=DPI
    )


if __name__ == "__main__":
    plot_exp_1(list(range(8)))
