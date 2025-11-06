""" Module to obtain label error from a model."""

import os
from typing import Optional, List, Literal
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from pybrid import utils, datasets
from pybrid.tests import test_model
from scipy.stats import qmc


def get_sampled_labels_error(
    model_folder: str,
    test_iters: int = 100,
    pkl_name: Optional[str] = None,
    batch_size: int = 512,
    nsamples: int = 1000,
    sample_strategy: Literal["uniform", "latinhyper", "normal"] = "uniform",
) -> pd.DataFrame:
    """Gets label error on a test set for a model.

    Args:
        model_folder (str): Folder containing the model.
        test_iters (int): Number of test iterations.
        pkl_name (Optional[str]): Name of the pkl file.
        batch_size (int): Batch size for testing.
        nsamples (int): Number of label samples to draw.
        sample_strategy (Literal): Strategy for sampling labels. Options are "uniform", "latinhyper", and "normal".

    Returns:
        A pandas DataFrame.

    Note: The logic behind this analysis is to compute the (model) total error induced by different labels. 
    We explore the label space at random, sampling points. The error is
    computed by running the model clamped on the labels for a fixed number of inference steps (starting from amortised states). The error is then the mean absolute error of the model's predictions at each layer.
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
    # get inference set
    infer_imgs, infer_labels, infer_contexts = utils.get_infer_set(test_loader)
    
    # determine unique classes
    class_labels = test_dataset.classes
    class_labels = [class_labels[c] for c in cfg.data.dataset_classes]

    # create grid of labels
    # get min and max label values
    min_label = 0.03
    max_label = 0.97
    # sample labels
    if sample_strategy == "latinhyper":
        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=infer_labels.shape[1])
        labels = torch.tensor(sampler.random(n=nsamples), dtype=infer_labels.dtype)
    elif sample_strategy == "uniform":
        labels = torch.rand(nsamples, len(class_labels))
    elif sample_strategy == "normal":
        # this one is more convoluted; we use the class labels as a means 
        samp_per_class = nsamples // len(class_labels)
        # sample a common normal distribution
        sigma = 0.5
        # now add mean and stack
        labels = torch.cat([torch.randn(samp_per_class, len(class_labels), device=infer_labels.device)*sigma + c for c in infer_labels], dim=0)
        
    labels = labels * (max_label - min_label) + min_label
    labels = labels.to(device=cfg.model.device)
    names = [f"rand-{n}" for n in list(range(len(labels)))]
    # also add the actual labels into the mix
    labels = torch.cat([labels, infer_labels], dim=0)
    names += class_labels

    # initialize empty dataframe
    full_df = pd.DataFrame()
    layer_labs = ["3", "2", "1", "sum"]
    # loop through classes
    for c, class_label in enumerate(class_labels):
        # get the class dataset
        class_dataset = label_dataset(infer_imgs[c], labels, names)
        # get a loader for the class
        class_loader = DataLoader(class_dataset, batch_size=batch_size, shuffle=False)
        # go through the loader
        for imgs, labs, nams in class_loader:
            # train the model
            model.train_batch(
                imgs,
                labs,
                amort_net_i=99, # dummy
                num_iters=test_iters,
                use_amort=False,
            )
            # now get the mean absolute error
            layer_errors = [model.errs[i+1].abs().mean(1) for i in range(model.num_layers)]
            layer_errors.append(torch.stack(layer_errors, dim=0).sum(0))
            # add to full dataframe
            for i, layer_error in enumerate(layer_errors):
                # create a dataframe for the layer
                df = pd.DataFrame({
                    "class": class_label,
                    "layer": layer_labs[i],
                    "name": nams,
                    "error": layer_error.cpu().numpy()
                })
                # append the label columns
                for ld in range(labs.shape[1]):
                    df[f"label_dim_{ld}"] = labs[:, ld].cpu().numpy()
                # add to full_df
                full_df = pd.concat([full_df, df], ignore_index=True)

    # calculate batches
    return full_df

def get_amort_labels_error(
    model_folder: str,
    test_iters: int = 100,
    pkl_name: Optional[str] = None,
    batch_size: int = 512,
    reduction: Literal["mean", "weighted_mean", "sum"] = "mean"
) -> pd.DataFrame:
    """Gets label error on a test set for a model.

    Args:
        model_folder (str): Folder containing the model.
        test_iters (int): Number of test iterations.
        pkl_name (Optional[str]): Name of the pkl file.
        batch_size (int): Batch size for testing.

    Returns:
        A pandas DataFrame.

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
    test_dataset = datasets.get_dataset(cfg)[1]
    # make dataloader
    test_loader = datasets.get_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False
    )    
    layer_labs = ["3", "2", "1", reduction]
    
    # go through the loader
    layer_errors = [[] for _ in range(model.num_layers)]
    with torch.no_grad():
        for imgs, labs, ctxs in test_loader:
            # get amortisation labels
            amort_labels = model.forward(imgs, ctxs)
            # train the model
            model.train_batch(
                imgs,
                amort_labels,
                ctxs,
                num_iters=test_iters,
                use_amort=True,
            )
            # now append mean absolute error
            for i in range(model.num_layers):
                layer_errors[i].append(model.errs[i+1].abs())

    # calculate mean over the dataset
    mean_errors = [torch.cat(x).mean().cpu() for x in layer_errors]

    # calculate reduction
    if reduction == "mean":
        mean_errors.append(torch.stack(mean_errors).mean()) 
    elif reduction == "weighted_mean":
        weights = 1/torch.tensor(cfg.model.nodes[1:])
        weights = weights / weights.sum()
        mean_errors.append((torch.stack(mean_errors)*weights).mean())
    elif reduction == "sum":
        mean_errors.append(torch.stack(mean_errors).sum())
    
    mean_errors = [e.item() for e in mean_errors]

    # add to full dataframe
    # create a dataframe for the layer
    df = pd.DataFrame({
        "layer": layer_labs,
        "error": mean_errors
    })
    return df

# a simple dataset class to hold the labels
class label_dataset(torch.utils.data.Dataset):
    """A simple dataset class to hold the labels."""

    def __init__(self, image: torch.Tensor, labels: torch.Tensor, names: List[str]):
        self.image = image
        self.labels = labels
        self.names = names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.image, self.labels[idx], self.names[idx]