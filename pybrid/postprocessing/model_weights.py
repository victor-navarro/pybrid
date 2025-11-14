"""A module to analyse weights from models"""

import itertools
from typing import List, Optional
import pandas as pd
import torch
from pybrid import utils

SUPPORTED_METRICS = ["l2-norm", "cos"]


def compare_models(
    model_pkls: List[str],
    model_names: Optional[List[str]] = None,
    compare_weights: bool = True,
    compare_biases: bool = False,
    metrics: List[str] = ["l2-norm", "cos"],
) -> pd.DataFrame:
    """Measure model weight differences"""
    assert all(
        [m in SUPPORTED_METRICS for m in metrics]
    ), f"metric not supported, use any of {SUPPORTED_METRICS}"

    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(model_pkls))]
    # get a list of all the model weights
    inf_vals = []
    amort_vals = []
    for model_pkl in model_pkls:
        # load the model
        model = utils.load_pkl(model_pkl)
        model_feats = []
        for l in range(model.num_layers):
            feats = None
            if compare_weights:
                feats = model.layers[l].weights.cpu().flatten()
            if compare_biases:
                if feats is None:
                    feats = model.layers[l].bias.cpu().flatten()
                else:
                    feats = torch.cat(
                        [feats, model.layers[l].bias.cpu().flatten()], dim=0
                    )
            model_feats.append(feats)
        inf_vals.append(model_feats)
        # now the amortization network(s)
        model_feats = [[] for _ in range(len(model.amort_nets))]
        for i in range(len(model.amort_nets)):
            net_feats = []
            for l in range(len(model.amort_nets[i])):
                feats = None
                if compare_weights:
                    feats = model.amort_nets[i][l].weights.cpu().flatten()
                if compare_biases:
                    if feats is None:
                        feats = model.amort_nets[i][l].bias.cpu().flatten()
                    else:
                        feats = torch.cat(
                            [feats, model.amort_nets[i][l].bias.cpu().flatten()], dim=0
                        )
                net_feats.append(feats)
            model_feats[i] = net_feats
        amort_vals.append(model_feats)

    # now we can go through the comparisons
    pairs = list(itertools.combinations(list(range(len(model_names))), 2))
    # define their names
    pair_names = [f"{model_names[pair[0]]}:{model_names[pair[1]]}" for pair in pairs]
    # create a dataframe to store the results
    results = pd.DataFrame()
    with torch.no_grad():
        cos = torch.nn.CosineSimilarity(dim=0)
        for pi, pair in enumerate(pairs):
            m1 = inf_vals[pair[0]]
            m2 = inf_vals[pair[1]]
            # compare the weights
            for metric in metrics:
                if metric == "l2-norm":
                    diffs = [(w1 - w2).norm().item() for w1, w2 in zip(m1, m2)]
                elif metric == "cos":
                    diffs = [cos(w1, w2).item() for w1, w2 in zip(m1, m2)]
                # create a new dataframe to store the results
                temp_res = pd.DataFrame(
                    {
                        "value": diffs,
                        "metric": metric,
                        "layer": list(range(len(m1))),
                        "comparison": pair_names[pi],
                        "network": "inference",
                    }
                )
                results = pd.concat([results, temp_res], ignore_index=True)
            # this cycles through the number of amortization networks
            for i in range(len(amort_vals[0])):
                m1 = amort_vals[pair[0]][i]
                m2 = amort_vals[pair[1]][i]
                # compare the weights
                for metric in metrics:
                    if metric == "l2-norm":
                        diffs = [(w1 - w2).norm().item() for w1, w2 in zip(m1, m2)]
                    elif metric == "cos":
                        diffs = [cos(w1, w2).item() for w1, w2 in zip(m1, m2)]
                    # create a new dataframe to store the results
                    temp_res = pd.DataFrame(
                        {
                            "value": diffs,
                            "metric": metric,
                            "layer": list(reversed(range(len(m1)))),  # reverse levels
                            "comparison": pair_names[pi],
                            "network": f"amort_{i}",
                        }
                    )
                    results = pd.concat([results, temp_res], ignore_index=True)
    return results
