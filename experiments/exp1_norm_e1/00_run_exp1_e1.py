"""Runs experiment 1 followup."""

import os
from typing import List
import logging
from pybrid.scripts_batches import main, run_twins
from pybrid.config import DefaultConfig
from pybrid.utils import get_mapping


def run_exp(output_dir: str, seeds: List[int]):
    """Run experiment 1.

    The progenitor model is trained with supervised, frozen class-labels.

    The twin models are trained with with amortized, frozen class-labels.
    """
    # NOTE: Each twin experiment involves the training of a progenitor model
    # and two twin models.
    # The progenitor model is trained on the original dataset.
    # The normal twin model is trained on the original dataset.
    # The swapped twin model is trained on a dataset where the superordinate labels
    # of the c-EMNIST dataset are swapped.

    prog_folder = os.path.join(output_dir, "exp_1_norm/progenitor")
    normal_folder = os.path.join(output_dir, "exp_1_norm_e1/normal_twin")
    swapped_folder = os.path.join(output_dir, "exp_1_norm_e1/swapped_twin")

    for seed in seeds:
        logging.info("Seed: %d", seed)
        cfg = DefaultConfig()
        cfg.exp.log_dir = prog_folder
        cfg.exp.seed = seed

        cfg.exp.num_epochs = 100  # number of epochs
        cfg.exp.test_every = 10  # test every 5 epochs
        cfg.exp.save_models = True  # save model pickle on each epoch

        # change the dataset
        cfg.data.dataset = "c-EMNIST_balanced"
        # select only some classes within the dataset
        cfg.data.dataset_classes = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 28]

        # configure network architecture
        cfg.model.nodes = [12, 500, 500, 784]
        cfg.model.amort_nodes = [784, 500, 500, 12]
        cfg.model.model_class = "DoubleAmortModel"

        # set flag to reduce the one-hot labels in the dataset
        cfg.data.shrink_classes = True

        # get superordinate labels
        cfg.data.train_sord = get_mapping("balanced", "alphanumerical")
        cfg.data.test_sord = get_mapping("balanced", "alphanumerical")
        cfg.data.trap_sord = get_mapping("balanced", "alphanumerical_rev")

        # reduce number of test iterations
        cfg.infer.num_test_iters = 100

        # global batch size
        cfg.optim.batch_size = 512
        # weight normalization
        cfg.optim.normalize_weights = True

        cfg.model.train_amort = True
        # test later
        cfg.exp.test_pc = False
        cfg.exp.test_amort = False
        cfg.exp.test_hybrid = False

        logging.info("Training progenitor model")
        # main(cfg, save_batches=False)

        twin_cfg = cfg.copy()
        # change some options
        twin_cfg.exp.num_epochs = 1
        # change to unsupervised mode
        twin_cfg.model.supervised = False
        logging.info("Training twins")
        run_twins(
            cfg=twin_cfg,
            progenitor_pkl=os.path.join(prog_folder, str(seed), "final_model.pkl"),
            normal_dir=normal_folder,
            swapped_dir=swapped_folder,
            save_batches=True,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    os.makedirs("results", exist_ok=True)
    run_exp(output_dir="results", seeds=list(range(1))) # only one seed
