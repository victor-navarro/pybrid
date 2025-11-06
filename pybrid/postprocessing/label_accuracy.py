""" Module to obtain label accuracies from a model and plot them."""

import os
from typing import Optional
import pandas as pd
import numpy as np
import torch
from pybrid import utils, datasets
from pybrid.tests import test_model


def get_label_accuracy(
    model_folder: str,
    test_iters: int = 100,
    pkl_name: Optional[str] = None,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Gets label accuracy on a test set for a model.

    Args:
        model_folder (str): Folder containing the model.
        test_iters (int): Number of test iterations.
        pkl_name (Optional[str]): Name of the pkl file.
        batch_size (int): Batch size for testing.

    Returns:
        A pandas DataFrame with class labels, iterations, and accuracies.
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
            # score activations using utility function
            hybrid_accs = _agg_acts(acts, labels)
            hybrid_accs["network"] = "hybrid"
            # add to full dataframe
            full_df = pd.concat([full_df, hybrid_accs])

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

            amort_accs = _agg_acts(acts, labels)
            amort_accs["network"] = "amort"
            full_df = pd.concat([full_df, amort_accs])

        # test pc component
        acts = test_model(imgs, contexts, model, layer=0, num_iters=test_iters)
        pc_accs = _agg_acts(acts, labels)
        pc_accs["network"] = "pc"
        full_df = pd.concat([full_df, pc_accs])

    # get sums
    full_df = full_df.groupby(["class", "iteration", "network"]).sum().reset_index()
    # calculate accuracy
    full_df["accuracy"] = full_df["correct"] / full_df["trials"]
    # relabel class
    full_df["class"] = full_df["class"].apply(lambda x: class_labels[x])
    # calculate batches
    return full_df


def _agg_acts(acts: torch.Tensor, labels: torch.Tensor) -> pd.DataFrame:
    # get accuracy
    pred_classes = np.argmax(acts, axis=2)
    batch_classes = np.argmax(labels.cpu().numpy(), axis=1)
    # compare each predicted class with the true class
    accs = pred_classes == batch_classes[:, None]
    # make into a dataframe
    df = pd.DataFrame(accs)
    df["class"] = batch_classes
    # melt the dataframe
    df = df.melt(id_vars="class", var_name="iteration", value_name="correct")
    # aggregate to counts of correct on a class by class basis
    # also add total trials per class
    agg_df = df.groupby(["class", "iteration"]).sum().reset_index()
    # add trials per class
    agg_df["trials"] = (
        df.groupby(["class", "iteration"]).count().reset_index()["correct"]
    )
    return agg_df
