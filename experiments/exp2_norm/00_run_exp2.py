""" Runs experiment 2 for CEMNIST. """

import os
from typing import List
import logging
from pybrid.scripts import resume_training
from pybrid import utils


TRAIN = True


def run_exp(output_dir: str, seeds: List[int]):
    """Run experiment 2.

    The twin models are trained with with amortized, unfrozen class-labels.
    They resume from the twins trained in exp1 (across different epochs)

    """
    base_folders = [
        os.path.join(output_dir, "exp_1_norm/normal_twin"),
        os.path.join(output_dir, "exp_1_norm/swapped_twin"),
    ]
    twin_folders = [
        os.path.join(output_dir, "exp_2_norm/normal_twin"),
        os.path.join(output_dir, "exp_2_norm/swapped_twin"),
    ]
    # define epochs to resume from
    epochs = [4] + list(range(9, 50, 10))

    if TRAIN:
        for seed in seeds:
            logging.info("Seed: %d", seed)
            for ti, twin in enumerate(["normal", "swapped"]):
                logging.info("Training %s twin", twin)
                for epoch in epochs:
                    # load base configuration for twin
                    cfg = utils.load_json_config(
                        os.path.join(base_folders[ti], str(seed), "config.json")
                    )
                    # determine model pickle
                    model_pkl = os.path.join(
                        base_folders[ti], str(seed), f"model_{epoch}.pkl"
                    )
                    # change to unfrozen mode
                    cfg.model.freeze_top = False
                    # retrain
                    resume_training(
                        cfg=cfg,
                        model_pkl=model_pkl,
                        output_dir=os.path.join(twin_folders[ti], str(epoch)),
                    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    os.makedirs("results", exist_ok=True)
    run_exp(output_dir="results", seeds=list(range(8)))
