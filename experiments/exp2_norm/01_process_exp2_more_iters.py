"""Process the results of experiment 2."""

from typing import List
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import pybrid.postprocessing as post
from pybrid import utils


TEST_ITERS = 1000
GET_ACCURACY = True
DO_RECONSTRUCTION = True


def process_exp(output_dir: str, seeds: List[int]):
    """Process some epochs of experiment 2 with more iterations."""
    # define the epochs (just focus on 0 and 49)
    epochs = [0, 49]
    for epoch in epochs:

        normal_folder = os.path.join(output_dir, "normal_twin", str(epoch))
        swapped_folder = os.path.join(output_dir, "swapped_twin", str(epoch))

        seed_folders = []
        for seed in seeds:
            for folder in [normal_folder, swapped_folder]:
                seed_folders.append(os.path.join(folder, str(seed)))

        if GET_ACCURACY:
            for folder in seed_folders:
                # find model files
                mfs, epochs = utils.get_model_files(folder)
                for mf, epoch in tqdm(zip(mfs, epochs), desc="Getting accuracy"):
                    # get accuracy
                    acc_df = post.get_label_accuracy(
                        folder, pkl_name=mf, test_iters=TEST_ITERS
                    )
                    # set csv path
                    csv_path = os.path.join(
                        folder, f"label_accuracy_more_iters/{epoch}.csv"
                    )
                    # make directory
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    # save as csv
                    acc_df.to_csv(csv_path, index=False)
                    # plot accuracy across iterations
                    fig, _ = post.plots.plot_iterations(csv_path)
                    plot_path = os.path.join(
                        folder, f"label_accuracy_more_iters/{epoch}.png"
                    )
                    fig.savefig(plot_path)
                    plt.close(fig)

                # now plot accuracy across epochs
                fig, _ = post.plots.plot_epochs(
                    os.path.join(folder, "label_accuracy_more_iters/")
                )
                fig.savefig(
                    os.path.join(
                        folder, "label_accuracy_more_iters/accuracy_epochs.png"
                    )
                )
                plt.close(fig)

        if DO_RECONSTRUCTION:
            for folder in seed_folders:
                mfs, epochs = utils.get_model_files(folder)
                for mf, epoch in tqdm(zip(mfs, epochs), desc="Getting rec error"):
                    # get reconstruction error
                    rec_df = post.get_reconstruction_error(
                        folder, pkl_name=mf, test_iters=TEST_ITERS
                    )
                    csv_path = os.path.join(
                        folder, f"reconstruction_more_iters/{epoch}.csv"
                    )
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    rec_df.to_csv(csv_path, index=False)
                    # plot reconstruction error across iterations
                    fig, _ = post.plots.plot_iterations(csv_path, metric="rec_error")
                    plot_path = os.path.join(
                        folder, f"reconstruction_more_iters/{epoch}.png"
                    )
                    fig.savefig(plot_path)
                    plt.close(fig)

                # now plot reconstruction error across epochs
                fig, _ = post.plots.plot_epochs(
                    os.path.join(folder, "reconstruction_more_iters/"),
                    metric="rec_error",
                )
                fig.savefig(
                    os.path.join(
                        folder, "reconstruction_more_iters/rec_error_epochs.png"
                    )
                )
                plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    os.makedirs("results", exist_ok=True)
    process_exp(output_dir="results/exp_2_norm", seeds=list(range(8)))
