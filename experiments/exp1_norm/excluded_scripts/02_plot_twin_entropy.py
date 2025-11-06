"""Process the results of experiment 1."""

from typing import List
import os
import pandas as pd
import matplotlib.pyplot as plt
import pybrid.postprocessing as post

RES_FOLDER = "results/exp_1_norm"
OUT_FOLDER = "plots/twin_entropy/exp_1"

def process_exp(seeds: List[int]):
    """Process the results of experiment 1."""
    all_data = pd.DataFrame()
    for twin in ["normal_twin", "swapped_twin"]:
        for seed in seeds:
            os.makedirs(os.path.join(OUT_FOLDER, str(seed)), exist_ok=True)
            folder = os.path.join(RES_FOLDER, twin, str(seed), "label_entropy")
            csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
            for csv_file in csv_files:
                # read csv file
                data = pd.read_csv(os.path.join(folder, csv_file))
                # add information
                data["seed"] = seed
                data["epoch"] = int(csv_file.split(".")[0])
                data["twin"] = twin
                # append to all_data
                all_data = pd.concat([all_data, data], ignore_index=True)

    # now make the plots for each epoch
    for epoch in all_data["epoch"].unique():
        # get data for this epoch
        epoch_data = all_data[all_data["epoch"] == epoch]
        # plot target entropy across iterations
        fig, axes = post.plots.multiplot(
            data=epoch_data[epoch_data["type"] == "target"], 
            multi="network", 
            what = "entropy", by = "iteration", group = "twin"
            )
        # add two lines to the axes
        # the highest line is the entropy for a flat label distribution (i.e., log(12))
        # the lowest line is the entropy for the smooth label we used ([.97, 0.03, ...])
        [ax.axhline(y=2.4849, color="r", linestyle="--") for ax in axes]
        [ax.axhline(y=2.4297, color="r", linestyle="--") for ax in axes]
        # remove the legend for the first two axes
        [ax.legend().remove() for ax in axes[:2]]
        # set plot title
        fig.suptitle("Full entropy")
        # change fig size
        fig.set_size_inches(12, 6)
        plot_path = os.path.join(OUT_FOLDER, f"{epoch}_target_entropy.png")

        fig.savefig(plot_path)
        plt.close(fig)

        # plot non-target entropy across iterations
        fig, axes = post.plots.multiplot(
            data=epoch_data[epoch_data["type"] == "non-target"], 
            multi="network", 
            what = "entropy", by = "iteration", group = "twin"
            )
        # For this one we just add one line (flat label distribution of 11 classes; log(11))
        [ax.axhline(y=2.3978, color="r", linestyle="--") for ax in axes]
        # remove the legend for the first two axes
        [ax.legend().remove() for ax in axes[:2]]
        # set plot title
        fig.suptitle("Non-target entropy")
        # change fig size
        fig.set_size_inches(12, 6)
        plot_path = os.path.join(OUT_FOLDER, str(seed), f"{epoch}_non-target_entropy.png")
        fig.savefig(plot_path)
        plt.close(fig)

if __name__ == "__main__":
    process_exp(seeds=list(range(1)))
