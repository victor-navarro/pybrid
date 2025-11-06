""" Runs check for CEMNIST. """

import os
from typing import List
import logging
from pybrid.scripts import main
from pybrid.config import DefaultConfig
from pybrid.utils import get_mapping


def run_exp(output_dir: str, seeds: List[int]):
    """DUMMY EXPERIMENT FOR DEBUG CHECKS"""

    prog_folder = os.path.join(output_dir, "dummy/progenitor")

    for seed in seeds:
        logging.info("Seed: %d", seed)
        cfg = DefaultConfig()
        cfg.exp.log_dir = prog_folder
        cfg.exp.seed = seed

        cfg.exp.num_epochs = 8  # number of epochs
        cfg.exp.test_every = 1  # test every 5 epochs
        cfg.exp.save_models = True  # save model pickle on each epoch

        # change the dataset
        cfg.data.dataset = "c-EMNIST_balanced"
        # select only some classes within the dataset
        cfg.data.dataset_classes = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 28]
        cfg.model.n_amort_nets = 2

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
        cfg.optim.batch_size = 64

        # weight normalization
        cfg.optim.normalize_weights = True
        cfg.model.train_amort = True
        
        cfg.exp.test_pc = True
        cfg.exp.test_amort = True
        cfg.exp.test_hybrid = True

        logging.info("Training progenitor model")
        main(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    os.makedirs("results", exist_ok=True)
    run_exp(output_dir="results", seeds=list(range(1)))
