"""Process the results of experiment 1."""

from typing import List
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import pybrid.postprocessing as post
from pybrid import utils


TEST_ITERS = 100
GET_ENTROPY = True


def process_exp(output_dir: str, seeds: List[int]):
    """Process the results of experiment 1."""
    prog_folder = os.path.join(output_dir, "exp_1_norm/progenitor")
    normal_folder = os.path.join(output_dir, "exp_1_norm/normal_twin")
    swapped_folder = os.path.join(output_dir, "exp_1_norm/swapped_twin")

    seed_folders = []
    for seed in seeds:
        for folder in [prog_folder, normal_folder, swapped_folder]:
            seed_folders.append(os.path.join(folder, str(seed)))

    if GET_ENTROPY:
        for folder in seed_folders:
            # find model files
            mfs, epochs = utils.get_model_files(folder)
            for mf, epoch in tqdm(zip(mfs, epochs), desc="Getting entropy"):

                # get entropy
                ent_df = post.get_label_entropy(
                    folder, pkl_name=mf, test_iters=TEST_ITERS
                )
                # set csv path
                csv_path = os.path.join(folder, f"label_entropy/{epoch}.csv")
                # make directory
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                # save as csv
                ent_df.to_csv(csv_path, index=False)
                # plot target entropy across iterations
                fig, axes = post.plots.multiplot(
                    data=ent_df[ent_df["type"] == "target"], 
                    multi="network", 
                    what = "entropy", by = "iteration"
                    )
                # add two lines to the axes
                # the highest line is the entropy for a flat label distribution (i.e., log(12))
                # the lowest line is the entropy for the smooth label we used ([.97, 0.03, ...])
                [ax.axhline(y=2.4849, color="r", linestyle="--") for ax in axes]
                [ax.axhline(y=2.4297, color="r", linestyle="--") for ax in axes]
                # set plot title
                fig.suptitle("Full entropy")
                # change fig size
                fig.set_size_inches(12, 6)
                plot_path = os.path.join(folder, f"label_entropy/{epoch}_target_entropy.png")

                fig.savefig(plot_path)
                plt.close(fig)

                # plot non-target entropy across iterations
                fig, axes = post.plots.multiplot(
                    data=ent_df[ent_df["type"] == "non-target"], 
                    multi="network", 
                    what = "entropy", by = "iteration"
                    )
                # For this one we just add one line (flat label distribution of 11 classes; log(11))
                [ax.axhline(y=2.3978, color="r", linestyle="--") for ax in axes]
                # set plot title
                fig.suptitle("Non-target entropy")
                # change fig size
                fig.set_size_inches(12, 6)
                plot_path = os.path.join(folder, f"label_entropy/{epoch}_non-target_entropy.png")

                fig.savefig(plot_path)
                plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    os.makedirs("results", exist_ok=True)
    process_exp(output_dir="results", seeds=list(range(8)))
