"""Process the results of experiment 1."""

from typing import List
import os
import pandas as pd
import matplotlib.pyplot as plt
import pybrid.postprocessing as post

RES_FOLDER = "results/exp_1_norm"
OUT_FOLDER = "plots/twin_label_ranks/exp_1"

def process_exp(seeds: List[int]):
    """Process the results of experiment 1."""
    for twin in ["normal_twin", "swapped_twin"]:
        for seed in seeds:
            base_folder = os.path.join(OUT_FOLDER, str(seed))
            os.makedirs(base_folder, exist_ok=True)
            folder = os.path.join(RES_FOLDER, twin, str(seed), "label_ranks")
            csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
            for csv_file in csv_files:
                # read csv file
                data = pd.read_csv(os.path.join(folder, csv_file))
                # get epoch
                epoch = int(csv_file.split(".")[0])
                for class_label in data["class"].unique():
                    # get data for this class
                    fig, axes = post.plots.multiplot(
                        data=data[data["class"] == class_label], 
                        multi="network", 
                        what = "rank", by = "iteration", group = "rank_value"
                        )
                    # remove the legend for the first two axes
                    [ax.legend().remove() for ax in axes[:2]]

                    # set plot title
                    fig.suptitle(f"Class: {class_label}")
                    # change fig size
                    fig.set_size_inches(12, 6)
                    plot_path = os.path.join(base_folder, f"{epoch}_{twin}_{class_label}_label_ranks.png")
                    fig.savefig(plot_path)
                    plt.close(fig)

if __name__ == "__main__":
    process_exp(seeds=list(range(1)))
