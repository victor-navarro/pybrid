"""Process the results of experiment 1."""

from typing import List
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import pybrid.postprocessing as post
from pybrid import utils


TEST_ITERS = 100
GET_ACCURACY = True
DO_RECONSTRUCTION = True
EXTRACT_FEATURES = True


def process_exp(output_dir: str, seeds: List[int]):
    """Process the results of experiment 1."""
    normal_folder = os.path.join(output_dir, "exp_1_norm_e1/normal_twin")
    swapped_folder = os.path.join(output_dir, "exp_1_norm_e1/swapped_twin")

    seed_folders = []
    for seed in seeds:
        for folder in [normal_folder, swapped_folder]:
            seed_folders.append(os.path.join(folder, str(seed)))

    if GET_ACCURACY:
        for folder in seed_folders:
            # find model files
            mfs, batches = utils.get_model_files(folder)
            for mf, batch in tqdm(zip(mfs, batches), desc="Getting accuracy"):
                # get accuracy
                acc_df = post.get_label_accuracy(
                    folder, pkl_name=mf, test_iters=TEST_ITERS
                )
                # set csv path
                csv_path = os.path.join(folder, f"label_accuracy/{batch}.csv")
                # make directory
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                # save as csv
                acc_df.to_csv(csv_path, index=False)
                # plot accuracy across iterations
                fig, _ = post.plots.plot_iterations(csv_path)
                plot_path = os.path.join(folder, f"label_accuracy/{batch}.png")
                fig.savefig(plot_path)
                plt.close(fig)

            # now plot accuracy across batches
            fig, _ = post.plots.plot_epochs(os.path.join(folder, "label_accuracy/"))
            fig.savefig(os.path.join(folder, "label_accuracy/accuracy_epochs.png"))
            plt.close(fig)

    if DO_RECONSTRUCTION:
        for folder in seed_folders:
            mfs, batches = utils.get_model_files(folder)
            for mf, batch in tqdm(zip(mfs, batches), desc="Getting rec error"):
                # get reconstruction error
                rec_df = post.get_reconstruction_error(
                    folder, pkl_name=mf, test_iters=TEST_ITERS
                )
                csv_path = os.path.join(folder, f"reconstruction/{batch}.csv")
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                rec_df.to_csv(csv_path, index=False)
                # plot reconstruction error across iterations
                fig, _ = post.plots.plot_iterations(csv_path, metric="rec_error")
                plot_path = os.path.join(folder, f"reconstruction/{batch}.png")
                fig.savefig(plot_path)
                plt.close(fig)

            # now plot reconstruction error across batches
            fig, _ = post.plots.plot_epochs(
                os.path.join(folder, "reconstruction/"), metric="rec_error"
            )
            fig.savefig(os.path.join(folder, "reconstruction/rec_error_epochs.png"))
            plt.close(fig)

    if EXTRACT_FEATURES:
        for folder in seed_folders:
            mfs, batches = utils.get_model_files(folder)
            for mf, batch in tqdm(zip(mfs, batches), desc="Getting features"):
                feats = post.get_features(folder, pkl_name=mf)
                # save features as pkls
                feats_path = os.path.join(folder, f"features/{batch}.pkl")
                utils.save_pkl(feats, feats_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    os.makedirs("results", exist_ok=True)
    process_exp(output_dir="results", seeds=list(range(1)))
