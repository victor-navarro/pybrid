"""Plot error surface for experiment 1."""

from typing import List
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

RES_FOLDER = "results/exp_1_norm"
FIGSIZE = (3.25, 6.5)
CMAP = "Spectral_r"
MESH = True
CLASS = "4"
LAYER = "sum"
EPOCHS = [0, 9, 49]
REDUCTION = "tsne"  # or "pca"
NORMALIZE = False  # whether to normalize error to [0, 1] range
DPI = 300

# define labels
labels = {"hybrid": "Amort+Inf", "pc": "Inf", "amort": "Amort"}

# define colour dict
colours = {
    "4": "#166d20",
    "A": "#e97133",
    "Other": "#909090",  # default grey for other labels
}

# define min and max of energy to deal with weird tails
NRG_MIN = 0.30
NRG_MAX = 0.35
# covers about 90% of the data


def plot_exp(seeds: List[int]) -> None:
    """Process the results of experiment 1."""
    for twin in ["normal_twin", "swapped_twin"]:
        all_data = pd.DataFrame()
        for seed in seeds:
            folder = os.path.join(RES_FOLDER, twin, str(seed), "sampled_label_error")
            csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
            for csv_file in csv_files:
                # read csv file
                data = pd.read_csv(os.path.join(folder, csv_file), low_memory=False)
                # add epoch
                data["epoch"] = int(csv_file.split(".")[0])
                # add seed
                data["seed"] = seed
                # filter
                data = data[data["epoch"].isin(EPOCHS)]  # filter for epochs
                data = data[data["class"] == CLASS]  # filter for class
                data = data[data["layer"] == LAYER]  # filter for layer
                # concat to all_data
                all_data = pd.concat([all_data, data], ignore_index=True)

        # print some quantiles
        print(f"0.05 quantile: {all_data['error'].quantile(0.05)}")
        print(f"0.95 quantile: {all_data['error'].quantile(0.95)}")

        # first, we get the first two dimensions of a PCA on the labels
        # get unique labels
        dim_cols = [col for col in all_data.columns if col.startswith("label_dim_")]
        unique_data = all_data[all_data["epoch"] == min(EPOCHS)]
        unique_labels = unique_data[dim_cols]
        # transform to numpy array
        unique_labels = unique_labels.to_numpy()
        if REDUCTION == "tsne":
            # apply t-SNE
            tsne = TSNE(n_components=2, perplexity=50, random_state=42)
            red_labels = tsne.fit_transform(unique_labels)
        elif REDUCTION == "pca":
            # apply PCA
            pca = PCA(n_components=2)
            red_labels = pca.fit_transform(unique_labels)
        else:
            raise ValueError("Reduction method not recognized.")

        red_df = pd.DataFrame(red_labels, columns=["d1", "d2"])
        # add label names
        red_df["name"] = unique_data["name"].values
        # add columns to all_data
        all_data = all_data.merge(red_df, left_on="name", right_on="name", how="left")

        # now we create the plots
        fig, ax = plt.subplots(len(EPOCHS), 1, figsize=FIGSIZE, layout="constrained")
        # get min and max of pca space
        pca_min = all_data[["d1", "d2"]].min().min() - 0.1
        pca_max = all_data[["d1", "d2"]].max().max() + 0.1

        for ei, epoch in enumerate(EPOCHS):
            # plot the PCA results as a mesh in 3D
            epoch_data = all_data[all_data["epoch"] == epoch].copy()
            # normalize error if needed
            if NORMALIZE:
                min_error = 0
                max_error = 1
                epoch_data["error"] = (
                    epoch_data["error"] - epoch_data["error"].min()
                ) / (epoch_data["error"].max() - epoch_data["error"].min())

            else:
                min_error = NRG_MIN
                max_error = NRG_MAX

            if MESH:
                step = (pca_max - pca_min) / 30
                grid = [
                    np.arange(pca_min, pca_max, step),
                    np.arange(pca_min, pca_max, step),
                ]
                bin_dat = stats.binned_statistic_2d(
                    epoch_data["d1"],
                    epoch_data["d2"],
                    values=epoch_data["error"],
                    statistic="mean",
                    bins=grid,
                )
                z = bin_dat.statistic
                # mask
                z = np.ma.masked_invalid(z)
                x, y = np.meshgrid(grid[0], grid[1])
                mesh = ax[ei].pcolor(
                    x, y, z.T, vmin=min_error, vmax=max_error, cmap=CMAP
                )
            else:
                # plot the points
                mesh = ax[ei].scatter(
                    epoch_data["d1"],
                    epoch_data["d2"],
                    c=epoch_data["error"],
                    cmap=CMAP,
                    vmin=min_error,
                    vmax=max_error,
                    s=50,
                    linewidth=0.5,
                )

            # select class labels from the epoch_data
            class_dat = epoch_data[~epoch_data["name"].str.startswith("rand-")]
            # now annotate the class labels on the plot
            for _, row in class_dat.iterrows():
                txt = ax[ei].text(
                    row["d1"],
                    row["d2"],
                    row["name"],
                    color="white",
                    fontsize=16,
                    zorder=10,
                    ha="center",
                )
                txt.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=2, foreground="black"),
                        path_effects.Normal(),
                    ]
                )
            # remove the axes ticks
            ax[ei].set_xticks([])
            ax[ei].set_yticks([])
            # set ylabel
            ax[ei].set_ylabel(f"Batch {(epoch+1)*56}")

        cbar = fig.colorbar(mesh, ax=ax[1], orientation="vertical")
        cbar.set_label("Total Error")

        plot_path = f"plots/exp_1_norm/sampled_label_error_{twin}_{CLASS}.png"
        fig.savefig(plot_path, dpi=DPI)
        plt.close(fig)


if __name__ == "__main__":
    plot_exp(seeds=list(range(1)))
