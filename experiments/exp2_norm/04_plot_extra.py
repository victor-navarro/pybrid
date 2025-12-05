"""Creates decent plots for experiment 2"""

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
twin_cmap = plt.get_cmap("Dark2")
twin_colours = {
    "normal": twin_cmap(0),
    "swapped": twin_cmap(1),
}


def plot_exp_2(seeds: List[int]):
    """Extra plots for experiment 2"""

    # A plot of hybrid accuracy across iterations

    for mod in ["swapped_twin", "normal_twin"]:
        acc_data = pd.DataFrame()
        for s in seeds:
            acc_data = pd.concat(
                [
                    acc_data,
                    pd.read_csv(
                        f"results/exp_2_norm/{mod}/49/{s}/label_accuracy_more_iters/49.csv"
                    ).assign(seed=s),
                ]
            )
        # on a seed by seed basis
        avg_acc_data = (
            acc_data.groupby(["network", "seed", "iteration"])["accuracy"]
            .mean()
            .reset_index()
        )
        avg_acc_data["class"] = "AVG"

        # join
        acc_data = pd.concat([acc_data, avg_acc_data])

        # now get mean and se
        agg_acc_data = (
            acc_data.groupby(["network", "iteration", "class"])["accuracy"]
            .agg(["mean", "sem"])
            .reset_index()
            .set_index("network")
        )
        max_iters = agg_acc_data["iteration"].max()

        # plot only the data for the hybrid and pc networks
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        classes = agg_acc_data["class"].unique()
        # put AVG last
        classes = np.append(classes[classes != "AVG"], "AVG")
        cmap = plt.get_cmap("tab20")
        for i, net in enumerate(["hybrid", "pc"]):
            # add 1/12 chance line
            ax[i].axhline(1 / 12, color="grey", linestyle="--")
            data = agg_acc_data.loc[net]
            for ci, c in enumerate(classes):
                if c == "AVG":
                    col = "black"
                else:
                    col = cmap(ci)
                class_data = data[data["class"] == c]
                ax[i].plot(
                    class_data["iteration"],
                    class_data["mean"],
                    label=c,
                    color=col,
                )
                # add ribbon for se
                ax[i].fill_between(
                    class_data["iteration"],
                    class_data["mean"] - class_data["sem"],
                    class_data["mean"] + class_data["sem"],
                    alpha=0.5,
                    color=col,
                )
                # add amortization point if hybrid
                if net == "hybrid":
                    ax[i].plot(
                        0,
                        class_data[class_data["iteration"] == 0]["mean"],
                        "o",
                        label=c,
                        color=col,
                    )
            ax[i].set_title(labels[net])
            ax[i].set_ylim(-0.01, 1.01)
            ax[i].set_xlim(-1, max_iters)
            ax[i].set_xlabel("Inference Step")
            if i == 0:
                ax[i].set_ylabel("Classification Accuracy")

        ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Save the plot
        fig.savefig(
            f"plots/exp_2_norm/{mod}_hybrid_pc_accuracy_49_49_more_iters.svg",
            bbox_inches="tight",
        )

        # Now do reconstruction
        rec_data = pd.DataFrame()
        for s in seeds:
            rec_data = pd.concat(
                [
                    rec_data,
                    pd.read_csv(
                        f"results/exp_2_norm/{mod}/49/{s}/reconstruction_more_iters/49.csv"
                    ).assign(seed=s),
                ]
            )
        # on a seed by seed basis
        avg_rec_data = (
            rec_data.groupby(["network", "seed", "iteration"])["rec_error"]
            .mean()
            .reset_index()
        )
        avg_rec_data["class"] = "AVG"

        # join
        rec_data = pd.concat([rec_data, avg_rec_data])

        # now get mean and se
        agg_rec_data = (
            rec_data.groupby(["network", "iteration", "class"])["rec_error"]
            .agg(["mean", "sem"])
            .reset_index()
            .set_index("network")
        )
        max_iters = agg_rec_data["iteration"].max()

        # plot only the data for the hybrid and pc networks
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        classes = agg_rec_data["class"].unique()
        # put AVG last
        classes = np.append(classes[classes != "AVG"], "AVG")
        cmap = plt.get_cmap("tab20")
        for i, net in enumerate(["hybrid", "pc"]):
            data = agg_rec_data.loc[net]
            for ci, c in enumerate(classes):
                if c == "AVG":
                    col = "black"
                else:
                    col = cmap(ci)
                class_data = data[data["class"] == c]
                ax[i].plot(
                    class_data["iteration"],
                    class_data["mean"],
                    label=c,
                    color=col,
                )
                # add ribbon for se
                ax[i].fill_between(
                    class_data["iteration"],
                    class_data["mean"] - class_data["sem"],
                    class_data["mean"] + class_data["sem"],
                    alpha=0.5,
                    color=col,
                )
                # add amortization point if hybrid
                if net == "hybrid":
                    ax[i].plot(
                        0,
                        class_data[class_data["iteration"] == 0]["mean"],
                        "o",
                        label=c,
                        color=col,
                    )
            ax[i].set_title(labels[net])
            ax[i].set_ylim(0.0, 0.1)
            ax[i].set_xlim(-1, max_iters)
            ax[i].set_xlabel("Inference Step")
            if i == 0:
                ax[i].set_ylabel("Reconstruction Error")

        ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Save the plot
        fig.savefig(
            f"plots/exp_2_norm/{mod}_hybrid_pc_reconstruction_49_49_more_iters.svg",
            bbox_inches="tight",
        )

    # A plot of hybrid accuracy across iterations, at the start and the end of experiment, with 0 delusional epochs

    for mod in ["normal_twin"]:
        for epoch in [0, 49]:
            acc_data = pd.DataFrame()
            for s in seeds:
                acc_data = pd.concat(
                    [
                        acc_data,
                        pd.read_csv(
                            f"results/exp_2_norm/{mod}/0/{s}/label_accuracy_more_iters/{epoch}.csv"
                        ).assign(seed=s),
                    ]
                )
            # on a seed by seed basis
            avg_acc_data = (
                acc_data.groupby(["network", "seed", "iteration"])["accuracy"]
                .mean()
                .reset_index()
            )
            avg_acc_data["class"] = "AVG"

            # join
            acc_data = pd.concat([acc_data, avg_acc_data])

            # now get mean and se
            agg_acc_data = (
                acc_data.groupby(["network", "iteration", "class"])["accuracy"]
                .agg(["mean", "sem"])
                .reset_index()
                .set_index("network")
            )
            max_iters = agg_acc_data["iteration"].max()

            # plot only the data for the hybrid and pc networks
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            classes = agg_acc_data["class"].unique()
            # put AVG last
            classes = np.append(classes[classes != "AVG"], "AVG")
            cmap = plt.get_cmap("tab20")
            for i, net in enumerate(["hybrid", "pc"]):
                # add 1/12 chance line
                ax[i].axhline(1 / 12, color="grey", linestyle="--")
                data = agg_acc_data.loc[net]
                for ci, c in enumerate(classes):
                    if c == "AVG":
                        col = "black"
                    else:
                        col = cmap(ci)
                    class_data = data[data["class"] == c]
                    ax[i].plot(
                        class_data["iteration"],
                        class_data["mean"],
                        label=c,
                        color=col,
                    )
                    # add ribbon for se
                    ax[i].fill_between(
                        class_data["iteration"],
                        class_data["mean"] - class_data["sem"],
                        class_data["mean"] + class_data["sem"],
                        alpha=0.5,
                        color=col,
                    )
                    # add amortization point if hybrid
                    if net == "hybrid":
                        ax[i].plot(
                            0,
                            class_data[class_data["iteration"] == 0]["mean"],
                            "o",
                            label=c,
                            color=col,
                        )
                ax[i].set_title(labels[net])
                ax[i].set_ylim(-0.01, 1.01)
                ax[i].set_xlim(-1, max_iters)
                ax[i].set_xlabel("Inference Step")
                if i == 0:
                    ax[i].set_ylabel("Classification Accuracy")

            ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))

            # Save the plot
            fig.savefig(
                f"plots/exp_2_norm/{mod}_hybrid_pc_accuracy_0_{epoch}_more_iters.svg",
                bbox_inches="tight",
            )

            # Now do reconstruction
            rec_data = pd.DataFrame()
            for s in seeds:
                rec_data = pd.concat(
                    [
                        rec_data,
                        pd.read_csv(
                            f"results/exp_2_norm/{mod}/0/{s}/reconstruction_more_iters/{epoch}.csv"
                        ).assign(seed=s),
                    ]
                )
            # on a seed by seed basis
            avg_rec_data = (
                rec_data.groupby(["network", "seed", "iteration"])["rec_error"]
                .mean()
                .reset_index()
            )
            avg_rec_data["class"] = "AVG"

            # join
            rec_data = pd.concat([rec_data, avg_rec_data])

            # now get mean and se
            agg_rec_data = (
                rec_data.groupby(["network", "iteration", "class"])["rec_error"]
                .agg(["mean", "sem"])
                .reset_index()
                .set_index("network")
            )
            max_iters = agg_rec_data["iteration"].max()

            # plot only the data for the hybrid and pc networks
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            classes = agg_rec_data["class"].unique()
            # put AVG last
            classes = np.append(classes[classes != "AVG"], "AVG")
            cmap = plt.get_cmap("tab20")
            for i, net in enumerate(["hybrid", "pc"]):
                data = agg_rec_data.loc[net]
                for ci, c in enumerate(classes):
                    if c == "AVG":
                        col = "black"
                    else:
                        col = cmap(ci)
                    class_data = data[data["class"] == c]
                    ax[i].plot(
                        class_data["iteration"],
                        class_data["mean"],
                        label=c,
                        color=col,
                    )
                    # add ribbon for se
                    ax[i].fill_between(
                        class_data["iteration"],
                        class_data["mean"] - class_data["sem"],
                        class_data["mean"] + class_data["sem"],
                        alpha=0.5,
                        color=col,
                    )
                    # add amortization point if hybrid
                    if net == "hybrid":
                        ax[i].plot(
                            0,
                            class_data[class_data["iteration"] == 0]["mean"],
                            "o",
                            label=c,
                            color=col,
                        )
                ax[i].set_title(labels[net])
                ax[i].set_ylim(0.0, 0.1)
                ax[i].set_xlim(-1, max_iters)
                ax[i].set_xlabel("Inference Step")
                if i == 0:
                    ax[i].set_ylabel("Reconstruction Error")

            ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))

            # Save the plot
            fig.savefig(
                f"plots/exp_2_norm/{mod}_hybrid_pc_reconstruction_0_{epoch}_more_iters.svg",
                bbox_inches="tight",
            )


if __name__ == "__main__":
    plot_exp_2(list(range(8)))
