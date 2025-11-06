""" Plot the t-SNE of the twins against the last progenitor. """

import os
import pandas as pd
from pybrid import utils


def filter_tsne_data(data: dict):
    """Filters the tsne data for the given network."""
    # get epochs
    pd_data = pd.DataFrame(
        {
            "class_names": data["class_names"],
            "tsne_1": data["tsne_centroids"][:, 0],
            "tsne_2": data["tsne_centroids"][:, 1],
            "model_name": data["model_name"],
            "pkl_name": data["pkl_name"],
        }
    )
    # filter entries with final_model in pkl_name
    pd_data = pd_data[~pd_data["pkl_name"].str.contains("final_model")]
    # calculate epoch from pkl_name
    pd_data["epoch"] = pd_data["pkl_name"].apply(
        lambda x: int(os.path.basename(x).split(".")[0])
    )
    # filter last epoch for progenitor or all epochs for any of the twins
    pd_data = pd_data[
        (
            (pd_data["epoch"] == pd_data["epoch"].max())
            & (pd_data["model_name"] == "Progenitor")
        )
        | (pd_data["model_name"] != "Progenitor")
    ]

    return pd_data


SEED = 0
LAYERS = [0, 1, 2, 3]
NETWORKS = ["hybrid", "pc", "amort"]

OUT_FOLDER = "results/exp_1/tsne/0/animations/twins"
os.makedirs(OUT_FOLDER, exist_ok=True)

for net in NETWORKS:
    for lay in LAYERS:
        tsne_pkl = f"results/exp_1/tsne/{SEED}/{net}_layer_{lay}_tsne_data.pkl"
        tsne_data = utils.load_pkl(tsne_pkl)
        filtered_data = filter_tsne_data(tsne_data)
        data_fname = os.path.join(OUT_FOLDER, f"tsne_data_{net}_layer_{lay}.csv")
        filtered_data.to_csv(data_fname, index=False)
        gif_fname = f"tsne_animation_{net}_layer_{lay}.gif"
        # build and execute the Rscript call
        rcall = f"Rscript experiments/misc/animate_twins_tsne.R {data_fname} {gif_fname} {OUT_FOLDER}"
        os.system(rcall)
