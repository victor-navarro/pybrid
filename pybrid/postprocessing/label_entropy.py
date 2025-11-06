""" Module to obtain label entropies from a model and plot them."""

import os
from typing import Optional
import pandas as pd
import numpy as np
import torch
from pybrid import utils, datasets
from pybrid.tests import test_model


def get_label_entropy(
    model_folder: str,
    test_iters: int = 100,
    pkl_name: Optional[str] = None,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Gets label entropy on a test set for a model.

    Args:
        model_folder (str): Folder containing the model.
        test_iters (int): Number of test iterations.
        pkl_name (Optional[str]): Name of the pkl file.
        batch_size (int): Batch size for testing.

    Returns:
        A pandas DataFrame with class labels, iterations, and entropies.
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
            hybrid_ents = _get_ents(torch.tensor(acts), labels)
            hybrid_ents["network"] = "hybrid"
            # add to full dataframe
            full_df = pd.concat([full_df, hybrid_ents])

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

            amort_ents = _get_ents(torch.tensor(acts), labels)
            amort_ents["network"] = "amort"
            full_df = pd.concat([full_df, amort_ents])

        # test pc component
        acts = test_model(imgs, contexts, model, layer=0, num_iters=test_iters)
        pc_ents = _get_ents(torch.tensor(acts), labels)
        pc_ents["network"] = "pc"
        full_df = pd.concat([full_df, pc_ents])

    # get sums
    full_df = full_df.groupby(["class", "type", "iteration", "network"]).sum().reset_index()
    # calculate entropy
    full_df["entropy"] = full_df["entropy"] / full_df["trials"]
    # relabel class
    full_df["class"] = full_df["class"].apply(lambda x: class_labels[x])
    return full_df


def _get_ents(acts: torch.Tensor, labels: torch.Tensor, eps: float = 1e-10) -> pd.DataFrame:
    # get probabilities
    pred_probs = torch.softmax(acts, dim=-1).cpu().numpy()
    batch_classes = np.argmax(labels.cpu().numpy(), axis=1)
    # calculate upper entropy bound
    # get the number of classes
    num_classes = pred_probs.shape[-1]
    # get entropies
    ents = pred_probs * np.log(pred_probs + eps)
    ents = -ents.sum(axis=-1)
    # now get the non-target entropies
    # create an index mask
    mask = np.arange(num_classes) != batch_classes[:, None]
    # repeat along iteration dimension
    mask = np.repeat(mask[:, None, :], pred_probs.shape[1], axis=1)
    # select non-target probabilities
    non_target_probs = pred_probs[mask].reshape(pred_probs.shape[0], -1, num_classes - 1)
    # re normalise probabilities
    non_target_probs = non_target_probs / np.sum(non_target_probs, axis=-1, keepdims=True)
    # get entropies
    non_target_ents = non_target_probs * np.log(non_target_probs + eps)
    non_target_ents = -non_target_ents.sum(axis=-1)
   
        
    # make into a dataframe
    df_ents = pd.DataFrame(ents)
    df_ents["class"] = batch_classes
    # melt the dataframe
    df_ents = df_ents.melt(id_vars="class", var_name="iteration", value_name="entropy")
    df_ents["type"] = "target"

    # make into a dataframe
    df_tents = pd.DataFrame(non_target_ents)
    df_tents["class"] = batch_classes
    # melt the dataframe
    df_tents = df_tents.melt(id_vars="class", var_name="iteration", value_name="entropy")
    df_tents["type"] = "non-target"

    # combine
    df_ents = pd.concat([df_ents, df_tents])

    # aggregate to average entropy as a function of class and iteration
    agg_df = df_ents.groupby(["class", "type", "iteration"]).sum().reset_index()
    # add trials per class
    agg_df["trials"] = (
        df_ents.groupby(["class", "type", "iteration"]).count().reset_index()["entropy"]
    )
    return agg_df
