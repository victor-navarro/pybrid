""" A module to analyse representations from models"""

import os
import logging
import itertools
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
from pybrid import utils
from pybrid.postprocessing.extraction import HybridFeatures


def do_rsa(
    feature_pkls: List[str],
    seeds: List[int],
    model_names: Optional[List[str]] = None,
    pkl_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    spearman: bool = False,
):
    """Performs classwise RSA on features from different models.

    Args:
        feature_pkls (List[str]): List of feature folders.
        seeds (List[int]): List of seeds.
        model_names (Optional[List[str]]): List of model names.
        pkl_names (Optional[List[str]]): List of pkl names.
        class_names (Optional[List[str]]): List of class names.
        output_dir (Optional[str]): Output directory

    """
    assert len(feature_pkls) > 1, "Need at least two models to compare"
    if model_names is None:
        model_names = [str(i) for i in range(len(feature_pkls))]
    if output_dir is None:
        output_dir = "rsa_results"

    all_corrs = None
    logging.info("Performing classwise RSA")

    # check pkl_names
    if pkl_names is None:
        pkl_names = ["final_model.pkl"] * len(feature_pkls)
    else:
        assert len(pkl_names) == len(
            feature_pkls
        ), "pkl_names must match model folders"

    for seed in seeds:
        logging.info("Processing seed %d", seed)
        seed_roots = [os.path.join(f, str(seed)) for f in feature_pkls]
        seed_folder = os.path.join(output_dir, "rsa", str(seed))
        os.makedirs(seed_folder, exist_ok=True)
        # read feature pkls
        features = []
        for i, f in enumerate(seed_roots):
            features.append(utils.load_pkl(os.path.join(f, "features", pkl_names[i])))

        # get RDMs for each model, networks, and layers
        for i, f in enumerate(features):
            f.get_distances_classwise()
            # save RDMs as a plot
            fig = f.visualize_classwise_distances(labels=class_names)
            fig.set_size_inches(8, 6)
            fig.savefig(os.path.join(seed_folder, f"{model_names[i].lower()}_rdm.png"))
            plt.close(fig)

        # now the complicated bit
        # we want to get the correlations between the RDMs, on a model and layer basis
        corrs = []
        nets = []
        layers = []
        for net in ["hybrid", "pc", "amort"]:
            # loop through the layers
            for l in range(features[0].n_layers):
                # get the classwise_distances for each model
                layer_distances = [f.classwise_distances[net][l] for f in features]
                # stack and get correlation matrix
                layer_distances = np.stack(layer_distances)
                if spearman:
                    corrs.append(spearmanr(layer_distances)[0])
                else:
                    corrs.append(np.corrcoef(layer_distances))

                nets.append(net)
                layers.append(l)

        # save as a dict; save dict as pkl
        corr_dict = {
            "corrs": np.stack(corrs),
            "nets": nets,
            "layers": layers,
            "model": model_names,
        }
        utils.save_pkl(corr_dict, os.path.join(seed_folder, "rsa_correlations.pkl"))

        # initialize all_corrs dict if necessary
        if all_corrs is None:
            all_corrs = corr_dict
            all_corrs["corrs"] = [corr_dict["corrs"]]

        else:
            # stack the correlation matrices
            all_corrs["corrs"].append(corr_dict["corrs"])

    # summarise correlation matrices in all_corrs with mean, sd, se, and n
    assert all_corrs is not None, "No correlation matrices found"

    # stack the correlation matrices
    all_corrs["corrs"] = np.stack(all_corrs["corrs"])
    all_corrs["mean"] = np.mean(all_corrs["corrs"], axis=0)
    all_corrs["sd"] = np.std(all_corrs["corrs"], axis=0)
    all_corrs["n"] = len(seeds)
    all_corrs["se"] = all_corrs["sd"] / all_corrs["n"]

    # save as pkl
    utils.save_pkl(all_corrs, os.path.join(output_dir, "rsa/rsa_summary.pkl"))


def do_tsne(
    feature_pkls: List[str],
    seeds: List[int],
    networks: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
):
    """Performs classwise tSNE on features from different models.

    Args:
        feature_pkls (List[str]): List of feature folders.
        seeds (List[int]): List of seeds.
        model_names (Optional[List[str]]): List of model names.
        pkl_names (Optional[List[str]]): List of pkl names.
        class_names (Optional[List[str]]): List of (pretty) class names.
        output_dir (Optional[str]): Output directory.
    """
    if model_names is None:
        model_names = [str(i) for i in range(len(feature_pkls))]
    if class_names is not None:
        nclasses = len(class_names)
    if networks is None:
        networks = ["hybrid", "pc", "amort"]
    if layers is None:
        layers = [0, 1, 2, 3]
    if output_dir is None:
        output_dir = "tsne_results"
    logging.info("Performing t-SNE on features")
    for seed in seeds:
        logging.info("Processing seed %d", seed)
        seed_roots = [os.path.join(f, str(seed)) for f in feature_pkls]
        seed_folder = os.path.join(output_dir, str(seed))
        os.makedirs(seed_folder, exist_ok=True)

        # The plan is to get all the class centroids together, and then do tSNE on them
        # We will have to go through the networks and the layers
        for network in networks:
            for layer in layers:
                logging.info("Performing t-SNE on %s layer %d", network, layer)
                tsne_data = {
                    "model_name": [],
                    "pkl_name": [],
                    "class_names": [],
                    "tsne_centroids": None,  # to be np.ndarray
                    "raw_centroids": None,  # to be np.ndarray,
                    "layer": layer,
                    "network": network,
                }
                # we'll have to read the features each time for memory reasons
                all_centroids = []  # to train the tSNE
                for i, r in enumerate(seed_roots):
                    # get the feature files in the feature folder
                    feature_files = os.listdir(r + "/features")
                    # now load the features
                    features = [
                        utils.load_pkl(os.path.join(r, "features", f))
                        for f in feature_files
                    ]
                    # get the centroids for each class
                    all_centroids.extend(
                        [get_centroids(f, layer, network) for f in features]
                    )
                    if class_names is None:
                        class_names = list(features[0].classes)
                        nclasses = len(class_names)
                    # add information
                    tsne_data["pkl_name"].extend(
                        [f for f in feature_files for _ in range(nclasses)]
                    )
                    tsne_data["class_names"].extend(class_names * len(feature_files))
                    # add to dict
                    tsne_data["model_name"].extend(
                        [model_names[i]] * (len(feature_files) * len(class_names))
                    )
                # get tSNE
                all_centroids = np.vstack(all_centroids)
                tsne = TSNE(n_components=2, random_state=0)
                tsne_centroids = tsne.fit_transform(all_centroids)
                tsne_data["tsne_centroids"] = tsne_centroids
                tsne_data["raw_centroids"] = all_centroids

                # save as pkl
                pkl_name = f"{network}_layer_{layer}_tsne_data.pkl"
                utils.save_pkl(tsne_data, os.path.join(seed_folder, pkl_name))


def do_cka(
    feature_pkls: List[str],
    model_names: Optional[List[str]] = None,
    debiased: bool = True,
) -> pd.DataFrame:
    """Performs CKA on features from different models.

    Args:
        feature_pkls (List[str]): List of feature folders.
        model_names (Optional[List[str]]): List of model names.
        debiased (bool): Use debiased CKA. Default is True.

    """
    assert len(feature_pkls) > 1, "Need at least two models to compare"
    if model_names is None:
        model_names = [str(i) for i in range(len(feature_pkls))]
    
    all_ckas = pd.DataFrame()
    logging.info("Performing CKA")

    # read feature pkls
    features = [utils.load_pkl(f) for f in feature_pkls]
    
    # now the complicated bit
    # we want to get the CKAs on a model and layer basis
    # initialise a dataframe with model1, model2, cka
    # model 1 and model 2 are the models being compared
    comps = []
    ckas = []
    nets = []
    layers = []

    # create unique combinations between two models in model_names
    combs = list(itertools.combinations(range(len(model_names)), 2))
    comp_names = [
        f"{model_names[m1]}:{model_names[m2]}" for m1, m2 in combs]

    for net in ["hybrid", "pc", "amort"]:
        for l in range(features[0].n_layers-1):
            for i, (m1, m2) in enumerate(combs):
                comps.append(comp_names[i])
                nets.append(net)
                layers.append(l)
                if m1 == m2:
                    ckas.append(1.0)
                else:
                    ckas.append(
                        np.clip(
                            cka_linear_features(
                                features[m1][net][l],
                                features[m2][net][l],
                                debiased=debiased,
                            ),
                            0.0,
                            1.0,
                        )
                    )

        # save as a csv
        cka_df = pd.DataFrame(
            {
                "comparison": comps,
                "cka": ckas,
                "network": nets,
                "layer": layers,
            }
        )
        # add to all_ckas
        all_ckas = pd.concat([all_ckas, cka_df])
    
    return all_ckas

def get_centroids(feats: HybridFeatures, layer: int, net: str):
    """Utility function to get feature centroids from network and layer in HybridFeatures object."""
    # get the layer centroid for each class
    centroids = np.zeros((len(feats.classes), feats.hybrid[layer].shape[1]))
    for i, c in enumerate(feats.classes):
        idx = np.where(feats.labels == c)[0]
        centroids[i] = np.mean(feats[net][layer][idx], axis=0)
    return centroids


def _debiased_dot_product_similarity_helper(
    xty: np.floating,
    sum_squared_rows_x: np.ndarray,
    sum_squared_rows_y: np.ndarray,
    squared_norm_x: np.floating,
    squared_norm_y: np.floating,
    n: int,
):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
        xty
        - n / (n - 2.0) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
    )


def cka_linear_features(
    features_x: np.ndarray, features_y: np.ndarray, debiased: bool = True
) -> np.floating:
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.
        Args:
            features_x: A num_examples x num_features matrix of features.
            features_y: A num_examples x num_features matrix of features.
            debiased: Use unbiased estimator of dot product similarity. CKA may still be
            biased. Note that this estimator may be negative.

        Returns:
            The value of CKA between X and Y.

        Note: This is borrowed from the demo code for Kornblith's paper on CKA.

    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))
    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum("ij,ij->i", features_x, features_x)
        sum_squared_rows_y = np.einsum("ij,ij->i", features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity,
            sum_squared_rows_x,
            sum_squared_rows_y,
            squared_norm_x,
            squared_norm_y,
            n,
        )
        normalization_x = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_x**2,
                sum_squared_rows_x,
                sum_squared_rows_x,
                squared_norm_x,
                squared_norm_x,
                n,
            )
        )
        normalization_y = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_y**2,
                sum_squared_rows_y,
                sum_squared_rows_y,
                squared_norm_y,
                squared_norm_y,
                n,
            )
        )

    return dot_product_similarity / (normalization_x * normalization_y)
