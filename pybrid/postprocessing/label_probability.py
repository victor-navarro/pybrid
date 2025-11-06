""" Module to obtain label probabilities from a model and plot them."""

import os
from typing import Optional
import pandas as pd
import numpy as np
import torch
from pybrid import utils, datasets
from pybrid.tests import test_model


def get_label_probability(
    model_folder: str,
    test_iters: int = 100,
    pkl_name: Optional[str] = None,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Gets label probability on a test set for a model.

    Args:
        model_folder (str): Folder containing the model.
        test_iters (int): Number of test iterations.
        pkl_name (Optional[str]): Name of the pkl file.
        batch_size (int): Batch size for testing.

    Returns:
        A pandas DataFrame with class labels, iterations, and probabilities.
    """
    # set config file
    config_file = os.path.join(model_folder, "config.json")
    # load configuration
    cfg = utils.load_json_config(config_file)
    # set seed
    utils.seed(cfg.exp.seed)

    # do model evaluation
    # set seed pkl
    if pkl_name is None:
        seed_pkl = os.path.join(model_folder, "final_model.pkl")
    else:
        seed_pkl = os.path.join(model_folder, pkl_name)
    # assert that the seed pkl exists
    assert os.path.exists(seed_pkl), "model pkl does not exist"
    # load model
    model = utils.load_pkl(seed_pkl)
    # get test set
    ds = datasets.get_dataset(cfg)
    test_dataset = ds[1]
    # make dataloader
    test_loader = datasets.get_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    # determine unique classes
    class_labels = test_dataset.classes
    class_labels = [class_labels[c] for c in cfg.data.dataset_classes]

    # initialize empty dataframe
    full_df = pd.DataFrame()

    # loop through batches
    for imgs, labels, contexts in test_loader:
        if cfg.model.train_amort:
            # model has amortised inference, so need to test hybrid and amort components
            acts = test_model(
                imgs, contexts, model, layer=0, num_iters=test_iters, use_amort=True
            )
            labels = labels.cpu().numpy()
            # score activations using utility function
            hybrid_probs = _agg_acts(acts, labels)
            hybrid_probs["network"] = "hybrid"
            # add to full dataframe
            full_df = pd.concat([full_df, hybrid_probs])

            # test amortization
            acts = test_model(
                imgs,
                contexts,
                model,
                layer=0,
                num_iters=1,
                use_amort=True,
                use_infer=False,
            )

            amort_probs = _agg_acts(acts, labels)
            amort_probs["network"] = "amort"
            full_df = pd.concat([full_df, amort_probs])

        # test pc component
        acts = test_model(imgs, contexts, model, layer=0, num_iters=test_iters)
        pc_probs = _agg_acts(acts, labels)
        pc_probs["network"] = "pc"
        full_df = pd.concat([full_df, pc_probs])

    # get sums again
    full_df = full_df.groupby(["class", "iteration", "network", "label"]).sum().reset_index()
    # calculate average label activation
    full_df["label_avg"] = full_df["label_sum"] / full_df["trials"]
    # relabel class
    full_df["class"] = full_df["class"].apply(lambda x: class_labels[x])
    # relabel label
    full_df["label"] = full_df["label"].apply(lambda x: class_labels[x])
    # calculate batches
    return full_df


def _agg_acts(acts: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    # tons of work just to avoid a softmax
    # get classes in batch
    labels = np.argmax(labels, -1)
    classes = np.unique(labels)
    choices = np.argmax(acts, -1)
    # create empty dataframe
    full_df = pd.DataFrame()
    for c in classes:
        # get the data for the class
        class_dat = choices[labels == c, :]
        # now loop through the labels
        for l in range(acts.shape[-1]):
            df = pd.DataFrame({"label_sum": (class_dat == l).sum(0), "iteration": np.arange(acts.shape[1])}) # sum across trials
            df["class"] = c
            df["label"] = l
            # add number of trials
            df["trials"] = class_dat.shape[0]
            # add to full_df
            full_df = pd.concat([full_df, df], ignore_index=True)

    return full_df
