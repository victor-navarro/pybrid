"""Process the results of experiment 1."""

from typing import List
import os
import logging
from tqdm import tqdm
import pybrid.postprocessing as post
from pybrid import utils


TEST_ITERS = 100


def process_exp(output_dir: str, seeds: List[int]):
    """Process the results of experiment 1."""
    normal_folder = os.path.join(output_dir, "exp_1_norm/normal_twin")
    swapped_folder = os.path.join(output_dir, "exp_1_norm/swapped_twin")

    seed_folders = []
    for seed in seeds:
        for folder in [normal_folder, swapped_folder]:
            seed_folders.append(os.path.join(folder, str(seed)))

    for folder in seed_folders:
        # find model files
        mfs, epochs = utils.get_model_files(folder)
        for mf, epoch in tqdm(zip(mfs, epochs), desc="Getting label error"):
            # set csv path
            csv_path = os.path.join(folder, f"amort_label_error/{epoch}.csv")
            # get energy
            nrg_df = post.get_amort_labels_error(
                folder, pkl_name=mf, test_iters=TEST_ITERS,
            )
            # add epoch
            nrg_df["epoch"] = epoch
            # make directory
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            # save as csv
            nrg_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    os.makedirs("results", exist_ok=True)
    process_exp(output_dir="results", seeds=list(range(8)))
