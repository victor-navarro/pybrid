""" Animate t-SNE of the progenitor network. """

# VN: this requires R with gganimate installed

import os
import pandas as pd
from pybrid import utils

SEED = 0
LAYERS = [0, 1, 2, 3]
NETWORKS = ["hybrid", "pc", "amort"]

OUT_FOLDER = "results/exp_1/tsne/0/animations/progenitor"
os.makedirs(OUT_FOLDER, exist_ok=True)


def filter_tsne_data(net: str):
    """Filters the tsne data for the given network."""
    picked_net = [f == net for f in tsne_data["model_name"]]
    # mark all non-final_model entries
    non_cemnist = [not "final_model" in f for f in tsne_data["pkl_name"]]

    # merge filters
    filters = [p and n for p, n in zip(picked_net, non_cemnist)]

    # filter pkl_names
    pkl_names = [
        tsne_data["pkl_name"][i]
        for i in range(len(tsne_data["pkl_name"]))
        if filters[i]
    ]
    tsne_centroids = tsne_data["tsne_centroids"][filters]
    class_names = [
        tsne_data["class_names"][i]
        for i in range(len(tsne_data["class_names"]))
        if filters[i]
    ]
    epoch = [int(os.path.basename(f).split(".")[0]) for f in pkl_names]
    pd_data = pd.DataFrame(
        {
            "class_names": class_names,
            "epoch": epoch,
            "tsne_1": tsne_centroids[:, 0],
            "tsne_2": tsne_centroids[:, 1],
        }
    )
    return pd_data


for net in NETWORKS:
    for lay in LAYERS:
        tsne_pkl = f"results/exp_1/tsne/{SEED}/{net}_layer_{lay}_tsne_data.pkl"
        tsne_data = utils.load_pkl(tsne_pkl)
        filtered_data = filter_tsne_data("Progenitor")
        data_fname = os.path.join(OUT_FOLDER, f"tsne_data_{net}_layer_{lay}.csv")
        filtered_data.to_csv(data_fname, index=False)
        gif_fname = f"tsne_animation_{net}_layer_{lay}.gif"
        # build and execute the Rscript call
        rcall = f"Rscript experiments/misc/animate_progenitor_tsne.R {data_fname} {gif_fname} {OUT_FOLDER}"
        os.system(rcall)
