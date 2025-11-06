""" Module to obtain label ranks from a model and plot them."""

import os
from typing import Optional
import pandas as pd
import numpy as np
import torch
from pybrid import utils, datasets
from pybrid.tests import test_model


def get_label_ranks(
    model_folder: str,
    test_iters: int = 100,
    pkl_name: Optional[str] = None,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Gets label rank on a test set for a model.

    Args:
        model_folder (str): Folder containing the model.
        test_iters (int): Number of test iterations.
        pkl_name (Optional[str]): Name of the pkl file.
        batch_size (int): Batch size for testing.

    Returns:
        A pandas DataFrame with class labels, iterations, and ranks.
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

    # loop through batches
    amort_acts = []
    hybrid_acts = []
    pc_acts = []
    full_labels = []
    for imgs, labels, contexts in test_loader:
        full_labels.append(labels.cpu().numpy())
        if cfg.model.train_amort:
            # model has amortised inference, so need to test hybrid and amort components
            hybrid_acts.append(test_model(
                imgs, contexts, model, layer=0, num_iters=test_iters, use_amort=True
            ))
            # test amortization
            amort_acts.append(test_model(
                imgs,
                contexts,
                model,
                layer=0,
                num_iters=1,
                use_amort=True,
                use_infer=False,
            ))

        # test pc component
        pc_acts.append(test_model(imgs, contexts, model, layer=0, num_iters=test_iters))

    # get average ranks for each network
    hybrid_ranks = _get_ranks(
        np.concatenate(hybrid_acts), np.concatenate(full_labels)
    )
    hybrid_ranks["network"] = "hybrid"
    amort_ranks = _get_ranks(
        np.concatenate(amort_acts), np.concatenate(full_labels)
    )
    amort_ranks["network"] = "amort"
    pc_ranks = _get_ranks(
        np.concatenate(pc_acts), np.concatenate(full_labels)
    )
    pc_ranks["network"] = "pc"

    rank_df = pd.concat([hybrid_ranks, amort_ranks, pc_ranks], ignore_index=True)
    
    # check some counts
    nclasses = labels.shape[1]
    # for example, the amortization network is expected to have 12*12 entries
    # because there's only one iteration, 12 classes, and 12 ranks
    assert(rank_df[(rank_df["network"] == "amort")].shape[0] == nclasses**2)
    # similarly, the both the hybrid and pc networks are expected to have 12*12*100 entries
    # because there are 100 iterations
    assert(rank_df[(rank_df["network"] == "hybrid")].shape[0] == nclasses**2*test_iters)
    assert(rank_df[(rank_df["network"] == "pc")].shape[0] == nclasses**2*test_iters)

    # relabel class
    rank_df["class"] = rank_df["class"].apply(lambda x: class_labels[x])
    # also relabel rank_value with class_label
    rank_df["rank_value"] = rank_df["rank_value"].apply(
        lambda x: class_labels[x]
    )
    
    return rank_df


def _get_ranks(acts: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    # first, we take the average activation for each class
    labels = np.argmax(labels, axis=-1)
    
    avg_acts = []
    for i in range(acts.shape[-1]):
        # get the average activations for this class
        avg_acts.append(np.mean(acts[labels == i, :, :], axis=0))
    avg_acts = np.array(avg_acts)
    # now we have the array of average activations across iterations
    # get rank
    ranks = np.argsort(-1*avg_acts, -1) # sort in descending order
    
    # initialize empty dataframe
    full_df = pd.DataFrame()
    # cycle through classes
    for i in range(acts.shape[-1]):
        df = pd.DataFrame(ranks[i])
        # each row is an iteration
        # each column is a rank
        # add class and iteration
        df["class"] = i
        df["iteration"] = np.arange(acts.shape[1])
        # melt
        df = df.melt(id_vars=["class", "iteration"], var_name="rank", value_name="rank_value")
        # add to full_df
        full_df = pd.concat([full_df, df], ignore_index=True)
    return full_df
