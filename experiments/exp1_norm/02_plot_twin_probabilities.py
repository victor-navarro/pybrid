"""Plots twin label probabilities for experiment 1."""

from typing import List
import os
import pandas as pd
import matplotlib.pyplot as plt

RES_FOLDER = "results/exp_1_norm"
OUT_FOLDER = "plots/twin_label_probabilities/exp_1"
FIGSIZE = (3, 4)
CLASS = "4"
EPOCHS = [0, 9, 49]

# define labels
labels = {"hybrid": "Amort+Inf", "pc": "Inf", "amort": "Amort"}

# define colour dict
colours = {
    "4": "#166d20",
    "A": "#e97133",
    "Other": "#909090",  # default grey for other labels
}


def plot_exp(seeds: List[int]) -> None:
    """Twin probabilities for experiment 1."""
    for twin in ["normal_twin", "swapped_twin"]:
        all_data = pd.DataFrame()
        for seed in seeds:
            base_folder = os.path.join(OUT_FOLDER, str(seed))
            os.makedirs(base_folder, exist_ok=True)
            folder = os.path.join(RES_FOLDER, twin, str(seed), "label_probabilities")
            csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
            for csv_file in csv_files:
                # read csv file
                data = pd.read_csv(os.path.join(folder, csv_file))
                # add epoch
                data["epoch"] = int(csv_file.split(".")[0])
                # add seed
                data["seed"] = seed
                # filter
                data = data[data["epoch"].isin(EPOCHS)]  # filter for epochs
                data = data[data["class"] == CLASS]  # filter for class
                # concat to all_data
                all_data = pd.concat([all_data, data], ignore_index=True)

        # aggregate into mean and sem
        all_data = (
            all_data.groupby(["class", "iteration", "network", "label", "epoch"])
            .agg(
                mean=("label_avg", "mean"),
                sem=("label_avg", "sem"),
            )
            .reset_index()
        )

        # create figure
        fig, ax = plt.subplots(
            len(EPOCHS),
            2,
            figsize=FIGSIZE,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        for ei, epoch in enumerate(EPOCHS):
            for ni, network in enumerate(["hybrid", "pc"]):
                # set ylabel if first column
                if ni == 0:
                    ax[ei, ni].set_ylabel(f"Batch {(epoch+1) * 56}\nProbability")
                # set x label if last row
                if ei == len(EPOCHS) - 1:
                    ax[ei, ni].set_xlabel("Iteration")
                # set title for each subplot
                if ei == 0:
                    ax[ei, ni].set_title(labels[network])
                # set y limits
                ax[ei, ni].set_ylim(0, 1)
                # select data for the epoch and network
                pdat = all_data[
                    (all_data["epoch"] == epoch) & (all_data["network"] == network)
                ]
                # plot the data
                for label, group in pdat.groupby("label"):
                    zorder = 2 if label in ["4", "A"] else 1
                    lab = label if label in ["4", "A"] else "Other"
                    ax[ei, ni].plot(
                        group["iteration"],
                        group["mean"],
                        label=lab,
                        color=colours[str(lab)],
                        zorder=zorder,
                    )
                    # if hybrid, add points on first iteration
                    if network == "hybrid":
                        ax[ei, ni].scatter(
                            group[group["iteration"] == 0]["iteration"],
                            group[group["iteration"] == 0]["mean"],
                            color=colours[str(lab)],
                            s=50,
                            zorder=zorder,
                        )

                    # add ribbon for sem
                    ax[ei, ni].fill_between(
                        group["iteration"],
                        group["mean"] - group["sem"],
                        group["mean"] + group["sem"],
                        alpha=0.2,
                        color=colours[str(lab)],
                        zorder=zorder,
                    )

        hs, ls = ax[2, 1].get_legend_handles_labels()
        by_label = dict(zip(ls, hs))
        # reorder so other appears last
        by_label = {
            k: by_label[k] for k in sorted(by_label, key=lambda x: x == "Other")
        }
        ax[2, 1].legend(
            by_label.values(),
            by_label.keys(),
            title="Label",
            loc="best",
            fontsize=8,
        )
        plot_path = f"plots/exp_1_norm/label_probabilities_{twin}_{CLASS}.svg"
        fig.savefig(plot_path)
        plt.close(fig)


if __name__ == "__main__":
    plot_exp(seeds=list(range(8)))
