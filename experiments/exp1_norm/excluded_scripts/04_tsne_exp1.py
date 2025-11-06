"""Perform RSA on experiment 1."""

from typing import List
import os
import logging
from pybrid.postprocessing import do_tsne


DO_TSNE = True


def rsa_exp(output_dir: str, seeds: List[int]):
    """Process the results of experiment 1."""
    prog_folder = os.path.join(output_dir, "exp_1/progenitor")
    normal_folder = os.path.join(output_dir, "exp_1/normal_twin")
    swapped_folder = os.path.join(output_dir, "exp_1/swapped_twin")
    pretty_names = ["0", "1", "2", "3", "4", "5", "A", "B", "C", "D", "E", "S"]

    feature_folders = [prog_folder, normal_folder, swapped_folder]
    model_names = ["Progenitor", "Normal", "Swapped"]

    if DO_TSNE:
        do_tsne(
            feature_folders=feature_folders,
            seeds=seeds,
            model_names=model_names,
            class_names=pretty_names,
            output_dir=os.path.join(output_dir, "exp_1/tsne"),
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    rsa_exp(output_dir="results", seeds=list(range(8)))
