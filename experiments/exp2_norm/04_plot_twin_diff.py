"""Creates decent plots for experiment 2"""

from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get a viridis colour map
cmap = plt.get_cmap("viridis")
# define colour palette for the three networks
colours = {
    "hybrid": cmap(0.0),
    "pc": cmap(0.4),
    "amort": cmap(0.8),
    "1": cmap(0.0),
    "2": cmap(0.4),
    "3": cmap(0.8),
}
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
twin_cmap = plt.get_cmap("Dark2")
twin_colours = {
    "normal": twin_cmap(0),
    "swapped": twin_cmap(1),
}


def plot_exp_2(epochs: List[int]):
    """Plot the results of experiment 2 across delusion epochs"""

    # The plan is to calculate a difference score between normal and swapped
    # twins and plot it as a function of the length of the
    # epochs used to train the twin from experiment 1.

    twin_acc_data = pd.DataFrame()
    for ee in epochs:
        dat = pd.read_csv(f"results/exp_2_norm/twin_accuracy_{ee}.csv")
        # select max epoch
        dat = dat[dat["epoch"] == dat["epoch"].max()]
        # select last iteration
        dat = dat.groupby(["seed", "class", "network", "twin"]).last().reset_index()
        # get average accuracy per network and twin
        dat = dat.groupby(["seed", "network", "twin"])["accuracy"].mean().reset_index()
        # calculate twin difference across seeds
        twin_diff = dat.pivot(
            index=["seed", "network"], columns="twin", values="accuracy"
        )
        twin_diff["diff"] = twin_diff["normal"] - twin_diff["swapped"]
        # get mean and se across seeds
        twin_diff = twin_diff.groupby("network").agg(["mean", "sem"]).reset_index()
        # add to the twin data
        twin_diff["del_epoch"] = ee
        twin_acc_data = pd.concat([twin_acc_data, twin_diff])

    # sort by del_epoch
    twin_acc_data = twin_acc_data.sort_values("del_epoch")
    # calculate batch number
    twin_acc_data["batch"] = (twin_acc_data["del_epoch"] + 1) * 56
    # zero out the zeroeth batch
    twin_acc_data.loc[twin_acc_data["del_epoch"] == 0, "batch"] = 0
    # group by network
    twin_acc_data = twin_acc_data.set_index("network")
    # save
    twin_acc_data.to_csv("results/exp_2_norm/twin_diff_del_epochs.csv", index=False)

    # plot the twin diff data
    # The plan here is to plot a single
    # Within each subplot, each network gets a different colour
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    batches = twin_acc_data.batch.unique()
    xticks = [batches[b] for b in [0, 2, 4, 6]]
    # plot accuracy as a function of batch number with standard error
    # networks are represented by different colors
    for network in ["hybrid", "pc", "amort"]:
        ax.plot(
            twin_acc_data.loc[network]["batch"],
            twin_acc_data.loc[network]["diff"]["mean"],
            label=labels[network],
            color=colours[network],
        )
        ax.fill_between(
            twin_acc_data.loc[network]["batch"],
            y1=twin_acc_data.loc[network]["diff"]["mean"]
            - twin_acc_data.loc[network]["diff"]["sem"],
            y2=twin_acc_data.loc[network]["diff"]["mean"]
            + twin_acc_data.loc[network]["diff"]["sem"],
            alpha=0.5,
            color=colours[network],
        )
    # put x tick labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    # add legend and labels
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Delusional Period Duration (Batches)")
    ax.set_ylabel("Accuracy Difference \n(Control - Delusional)")
    # set limits
    ax.set_ylim(-0.01, 1)
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    fig.savefig("plots/exp_2_norm/twin_acc_diff_del_epochs.svg", bbox_inches="tight")

    # now do the same, but for prediction errors
    twin_rec_data = pd.DataFrame()
    for ee in epochs:
        dat = pd.read_csv(
            f"results/exp_2_norm/twin_amort_label_error_{ee}.csv"
        )  # last epoch of self-sustaining period
        # remove "mean" layer
        dat = dat[dat["layer"] != "mean"]
        # select last epoch
        dat = dat[dat["epoch"] == dat["epoch"].max()]
        # calculate twin difference across seeds and layers
        twin_diff = dat.pivot(index=["seed", "layer"], columns="twin", values="error")
        twin_diff["diff"] = twin_diff["swapped"] - twin_diff["normal"]
        # get mean and se across seeds
        twin_diff = twin_diff.groupby("layer").agg(["mean", "sem"]).reset_index()
        # add to the twin data
        twin_diff["del_epoch"] = ee
        twin_rec_data = pd.concat([twin_rec_data, twin_diff])

    # sort by del_epoch
    twin_rec_data = twin_rec_data.sort_values("del_epoch")
    # calculate batch number
    twin_rec_data["batch"] = (twin_rec_data["del_epoch"] + 1) * 56
    # zero out the zeroeth epoch
    twin_rec_data.loc[twin_rec_data["del_epoch"] == 0, "batch"] = 0
    # group by network
    twin_rec_data = twin_rec_data.set_index("layer")
    # save
    twin_rec_data.to_csv(
        "results/exp_2_norm/twin_error_diff_del_epochs.csv", index=False
    )

    # plot the twin diff data
    # Within each subplot, each network gets a different colour
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    # plot accuracy as a function of batch number with standard error
    # networks are represented by different colors
    for layer in ["1", "2", "3"]:
        ax.plot(
            twin_rec_data.loc[layer]["batch"],
            twin_rec_data.loc[layer]["diff"]["mean"],
            label=f"Layer {layer}",
            color=colours[layer],
        )
        ax.fill_between(
            twin_rec_data.loc[layer]["batch"],
            y1=twin_rec_data.loc[layer]["diff"]["mean"]
            - twin_rec_data.loc[layer]["diff"]["sem"],
            y2=twin_rec_data.loc[layer]["diff"]["mean"]
            + twin_rec_data.loc[layer]["diff"]["sem"],
            alpha=0.5,
            color=colours[layer],
        )

    # put x tick labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    # add legend and labels
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Delusional Period Duration (Batches)")
    ax.set_ylabel("Pred. Error Difference \n(Delusional - Control)")
    # set limits
    ax.set_ylim(-0.005, 0.005)
    fig.savefig(
        "plots/exp_2_norm/twin_amort_error_diff_del_epochs.svg",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    EPOCHS = list(range(9, 50, 10)) + [0, 4]
    plot_exp_2(EPOCHS)
