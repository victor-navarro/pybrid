""" A module with functions to summarise experiments """

import os
import logging
from typing import Optional, List
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pybrid import utils
from pybrid import tests
from pybrid import datasets

from pybrid.config import DefaultConfig


def postprocess_runs(
    folder: str,
    test_iters: int = 100,
    pkl_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    test_reconstruction: bool = False,
):
    """Get additional summary statistics for an experiment(s)."""
    seed_folders = [
        s for s in os.listdir(folder) if os.path.isdir(os.path.join(folder, s))
    ]
    metrics_dict = {}
    confusion_dict = {}
    reconstruction_dict = {}
    logging.info("Postprocessing runs")
    # go through all the seed folders
    for seed in seed_folders:
        logging.info("Processing seed %s", seed)
        # get the seed folder
        seed_folder = os.path.join(folder, seed)
        # set config file
        config_file = os.path.join(seed_folder, "config.json")
        # load configuration
        cfg = utils.load_json_config(config_file)
        # set seed
        utils.seed(cfg.exp.seed)

        # set metrics file
        metrics_file = os.path.join(seed_folder, "metrics.json")
        # load metrics
        metrics = utils.load_json(metrics_file)

        # initialize metrics_dict if necessary
        if len(metrics_dict.keys()) == 0:
            for k in metrics.keys():
                metrics_dict[k] = []

        # append metrics
        for k, v in metrics.items():
            metrics_dict[k].append(v)

        # do model evaluation
        # set seed pkl
        if pkl_name is None:
            seed_pkl = os.path.join(seed_folder, "final_model.pkl")
        else:
            seed_pkl = os.path.join(seed_folder, pkl_name)
        # assert that the seed pkl exists
        assert os.path.exists(seed_pkl), "model pkl does not exist"
        # load model
        model = utils.load_pkl(seed_pkl)
        # set number of test iterations
        cfg.infer.num_test_iters = test_iters

        # get confusion matrices
        logging.info("Generating confusion matrices")
        # get model predictions
        confusion_pkl = os.path.join(seed_folder, "confusion_matrix.pkl")
        preds = tests.test_predictions(model, cfg, iterwise=False, seed=0)
        # save to pkl
        utils.save_pkl(preds, confusion_pkl)

        # initialize confusion_dict if necessary
        if len(confusion_dict.keys()) == 0:
            # add labels key once
            confusion_dict["labels"] = preds["labels"].tolist()
            # for the other keys, initialize as lists
            for k in preds.keys():
                if "labels" not in k:
                    confusion_dict[k] = []

        # append
        for k, v in preds.items():
            if "labels" not in k:
                confusion_dict[k].append(v)

        if test_reconstruction:
            # get reconstruction error
            logging.info("Calculating reconstruction error")
            reconstruction_json = os.path.join(
                seed_folder, "reconstruction_errors.json"
            )
            # get epochs of models in folder
            mod_files = [f for f in os.listdir(seed_folder) if f.startswith("model_")]
            mod_epochs = [int(f.split("_")[1].split(".")[0]) for f in mod_files]
            mod_epochs.sort()

            errors = reconstruction_error(
                cfg=cfg,
                epochs=mod_epochs,
            )
            # save to json
            utils.save_json(errors, reconstruction_json)

            # initialize reconstruction_dict if necessary
            if len(reconstruction_dict.keys()) == 0:
                for k in errors.keys():
                    reconstruction_dict[k] = []

            # append
            for k, v in errors.items():
                reconstruction_dict[k].append(v)

    # stack and summarise metrics
    for k, v in metrics_dict.items():
        metrics_dict[k] = np.stack(v)
        mu = np.mean(metrics_dict[k], axis=0)
        std = np.std(metrics_dict[k], axis=0)
        n = len(metrics_dict[k])
        metrics_dict[k] = {
            "mean": mu.tolist(),
            "se": (std / np.sqrt(n)).tolist(),
            "std": std.tolist(),
            "n": n,
        }

    # save as json
    if output_dir is None:
        output_dir = folder
    utils.save_json(metrics_dict, os.path.join(output_dir, "metrics_summary.json"))

    metric_keys = [
        "hybrid_acc",
        "pc_acc",
        "amort_acc",
        "pc_losses",
        "pc_errs",
        "amort_losses",
        "amort_errs",
        "init_errs",
        "final_errs",
    ]
    logging.info("Plotting general metrics")
    met_plot = plot_metrics(metrics_dict, keys=metric_keys)
    met_plot.savefig(os.path.join(output_dir, "metrics_plot.png"))

    # stack and summarise confusion matrices
    for k, v in confusion_dict.items():
        if "labels" not in k:
            confusion_dict[k] = np.stack(v)
            mu = np.mean(confusion_dict[k], axis=0)
            std = np.std(confusion_dict[k], axis=0)
            n = len(confusion_dict[k])
            confusion_dict[k] = {
                "mean": mu,
                "se": (std / np.sqrt(n)),
                "std": std,
                "n": n,
            }
    # save as pkl
    utils.save_pkl(confusion_dict, os.path.join(output_dir, "confusion_summary.pkl"))

    if test_reconstruction:
        # stack and summarise reconstruction errors
        for k, v in reconstruction_dict.items():
            reconstruction_dict[k] = np.stack(v)
            mu = np.mean(reconstruction_dict[k], axis=0)
            std = np.std(reconstruction_dict[k], axis=0)
            n = len(reconstruction_dict[k])
            reconstruction_dict[k] = {
                "mean": mu.tolist(),
                "se": (std / np.sqrt(n)).tolist(),
                "std": std.tolist(),
                "n": n,
            }
        # add batch_idx
        reconstruction_dict["batch_idx"] = metrics_dict["batch_idx"]
        # save as json
        utils.save_json(
            reconstruction_dict, os.path.join(output_dir, "reconstruction_summary.json")
        )

        logging.info("Plotting reconstruction errors")
        reconstruction_keys = ["hybrid_error", "pc_error", "amort_error", "label_error"]
        rec_plot = plot_metrics(reconstruction_dict, keys=reconstruction_keys)
        rec_plot.savefig(os.path.join(output_dir, "reconstruction_plot.png"))


def plot_metrics(metrics: dict, keys: Optional[list] = None):
    """Plot metrics.

    Args:
        metrics (dict): Dictionary of metrics.
        keys (list): List of keys to plot.
    """
    if keys is None:
        keys = list(metrics.keys())
    # plot general metrics
    plot_side = int(np.ceil(np.sqrt(len(keys))))
    batch_ids = metrics["batch_idx"]["mean"]

    fig, axes = plt.subplots(plot_side, plot_side, layout="constrained")
    for ax, k in zip(axes.flat, keys):
        ax.set_title(f"{k}")
        # plot line with means and error bars
        ax.errorbar(
            batch_ids,
            metrics[k]["mean"],
            yerr=metrics[k]["se"],
            fmt="o",
        )
    return fig
