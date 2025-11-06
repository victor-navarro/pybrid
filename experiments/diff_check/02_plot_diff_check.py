""" Creates decent plots for diff_check_mnist"""

import os
import glob
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pybrid import utils

# define DPI
DPI = 600

# get a viridis colour map
cmap = plt.get_cmap("viridis")
# define colour palette for the three networks
colours = {"hybrid": cmap(0.0), "pc": cmap(0.4), "amort": cmap(0.8)}
# define labels for the three networks
labels = {
    "hybrid": "Hybrid",
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



def plot_exps(seeds: List[int], experiments: List[str]):
    """Plot the results"""

    for exp in experiments:

        # plot classification accuracy of the progenitor model
        acc_data = pd.DataFrame()

        for s in seeds:
            # set batches per epoch by hand
            # this depends on the batch size and the dataset used
            # for EMNIST_c-MNIST, at batch size 512, there are 117 batches per epoch.
            # for EMNIST_c-MNIST, at batch size 64, there are 937 batches per epoch.

            # load the config
            config_file = os.path.join(f"results/{exp}/progenitor/{s}/config.json")
            cfg = utils.load_json_config(config_file)
            dataset_len = 24000 if "balanced" in cfg.data.dataset else 60000
            batches_per_epoch = dataset_len // cfg.optim.batch_size

            # get all csv files in the folder
            csv_files = glob.glob(f"results/{exp}/progenitor/{s}/label_accuracy/*.csv")
            # load the data and append epoch (leading digits in filename) as a column
            for f in csv_files:
                epoch = int(os.path.basename(f).split(".")[0])
                acc_data = pd.concat(
                    [
                        acc_data,
                        pd.read_csv(f).assign(epoch=epoch, seed=s, batch=(epoch + 1)*batches_per_epoch),
                    ]
                )

        acc_data.to_csv(f"results/{exp}/progenitor_accuracy.csv", index=False)
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
        # zero out sem if NaN
        prog_acc["sem"] = prog_acc["sem"].fillna(0)

        # plot the data
        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        ax.axhline(1 / 12, color="gray", linestyle="--")
        # plot accuracy as a function of batch number with standard error
        # networks are represented by different colors
        for network in ["hybrid", "pc", "amort"]:
            ax.plot(
                prog_acc.loc[network]["batch"],
                prog_acc.loc[network]["mean"],
                label=labels[network],
                color=colours[network],
            )
            # dots
            # ax.scatter(
            #     prog_acc.loc[network]["batch"],
            #     prog_acc.loc[network]["mean"],
            #     color=colours[network],
            #     s=5,
            # )
            ax.fill_between(
                prog_acc.loc[network]["batch"],
                y1=prog_acc.loc[network]["mean"] - prog_acc.loc[network]["sem"],
                y2=prog_acc.loc[network]["mean"] + prog_acc.loc[network]["sem"],
                alpha=0.5,
                color=colours[network],
            )
        # add legend and labels
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel("Batch")
        ax.set_ylabel("Classification Accuracy")
        # set limits
        ax.set_ylim(-0.01, 1)
        ax.set_yticks(np.arange(0, 1.0, 0.10))
        os.makedirs("plots/diff_check_mnist/", exist_ok=True)
        fig.savefig(
            f"plots/diff_check_plots/{exp}.png", bbox_inches="tight", dpi=300
        )


if __name__ == "__main__":
    exp_list = ["diff_check_mnist", "diff_check_mnist_emnist", "diff_check_balanced_emnist"]
    plot_exps(list(range(1)), exp_list)
