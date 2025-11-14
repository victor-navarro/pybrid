"""Process the results of experiment 1."""

from typing import List
import os
import logging
from tqdm import tqdm
import pybrid.postprocessing as post
from pybrid import utils


TEST_ITERS = 100
GET_ERROR = True


def process_exp(output_dir: str, seeds: List[int]) -> None:
    """Get error from sampled labels for experiment 1.

    This error is like the amortized label error, but with sampled labels instead.

    """
    prog_folder = os.path.join(output_dir, "exp_1_norm/progenitor")
    normal_folder = os.path.join(output_dir, "exp_1_norm/normal_twin")
    swapped_folder = os.path.join(output_dir, "exp_1_norm/swapped_twin")

    seed_folders = []
    for seed in seeds:
        for folder in [prog_folder, normal_folder, swapped_folder]:
            seed_folders.append(os.path.join(folder, str(seed)))

    if GET_ERROR:
        for folder in seed_folders:
            # don't do the progenitor
            if "progenitor" in folder:
                continue
            # find model files
            mfs, epochs = utils.get_model_files(folder)
            for mf, epoch in tqdm(zip(mfs, epochs), desc="Getting label error"):
                # set csv path
                csv_path = os.path.join(folder, f"sampled_label_error/{epoch}.csv")
                # get energy
                nrg_df = post.get_sampled_labels_error(
                    folder,
                    pkl_name=mf,
                    test_iters=TEST_ITERS,
                    nsamples=2400,
                    sample_strategy="normal",
                )
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
    # just do the first seed
    process_exp(output_dir="results", seeds=list(range(1)))
