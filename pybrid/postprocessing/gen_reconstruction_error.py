""" A function to extract the reconstruction error for a given model. """

import os
from typing import Optional
import torch
import pandas as pd
from pybrid import datasets, utils
from pybrid.tests import test_model


def get_gen_reconstruction_error(
    model_folder: str,
    test_iters: int = 100,
    pkl_name: Optional[str] = None,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Calculate reconstruction error for a given model.

    Args:
        folder: Folder containing the configuration and model files. (will be used as output)
        pkl_name: Name of the pickle file (optional)

    Returns:
        A pandas DataFrame with class labels, iterations, and reconstruction errors.


    Note:
        The reconstruction error is defined as the absolute pixelwise error between
        the original image and the reconstructed image. This reconstruction
        error is calculated differently for the hybrid, pc, and amort networks.

        For the hybrid and pc networks, the reconstruction error is calculated
        at the end of iteration loop. For the amort network, the reconstruction
        error is calculated by passing the label predictions through the pc network.

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
            acts = torch.tensor(acts).to(model.device)
            hybrid_preds = model.backward(acts)
            # score activations using utility function
            hybrid_rec = _agg_errs(torch.abs(imgs.unsqueeze(1) - hybrid_preds), labels)
            hybrid_rec["network"] = "hybrid"
            # add to full dataframe
            full_df = pd.concat([full_df, hybrid_rec])

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
            acts = torch.tensor(acts).to(model.device)
            # do a backward pass to get image
            amort_preds = model.backward(acts)
            amort_rec = _agg_errs(torch.abs(imgs.unsqueeze(1) - amort_preds), labels)
            amort_rec["network"] = "amort"
            full_df = pd.concat([full_df, amort_rec])

        # test pc
        acts = test_model(imgs, contexts, model, layer=0, num_iters=test_iters)
        acts = torch.tensor(acts).to(model.device)
        pc_preds = model.backward(acts)
        pc_rec = _agg_errs(torch.abs(imgs.unsqueeze(1) - pc_preds), labels)
        pc_rec["network"] = "pc"
        full_df = pd.concat([full_df, pc_rec])

    # get sums
    full_df = full_df.groupby(["class", "iteration", "network"]).sum().reset_index()
    # calculate average error per image
    full_df["rec_error"] = full_df["total_error"] / full_df["trials"]
    # relabel class
    full_df["class"] = full_df["class"].apply(lambda x: class_labels[x])
    return full_df


# convenience function to aggregate reconstruction errors per classes with batch support
def _agg_errs(errs: torch.Tensor, labels: torch.Tensor) -> pd.DataFrame:
    batch_classes = labels.argmax(1).cpu().numpy()
    # compute mean error per image
    mean_err = errs.mean(2).cpu().numpy()
    # sum across unique batch_classes
    err_list = [mean_err[batch_classes == c].sum(0) for c in set(batch_classes)]
    # count number of trials per class
    trials = [sum(batch_classes == c) for c in set(batch_classes)]
    # make into a dataframe
    df = pd.DataFrame(err_list)
    df["class"] = list(set(batch_classes))
    # add trials per class
    df["trials"] = trials
    # melt the dataframe
    df = df.melt(
        id_vars=["class", "trials"], var_name="iteration", value_name="total_error"
    )
    return df
