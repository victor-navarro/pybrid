import os
from typing import Optional, List
import logging
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from pybrid import utils, datasets
from pybrid.config import DefaultConfig

_FEAT_DICT_KEYS = ["hybrid", "pc", "amort", "labels"]


class Extractor(torch.nn.Module):
    """
    Instatiates a network to be used as feature extractor.
    """

    def __init__(self, model_cfg: DefaultConfig, model_pkl: str):
        super().__init__()

        self.model_cfg: DefaultConfig = model_cfg
        self.num_iters = model_cfg.infer.num_test_iters
        self.init_std = model_cfg.infer.init_std
        self.fixed_preds = model_cfg.infer.fixed_preds_test
        self.use_amort = model_cfg.model.train_amort
        self.thresh = model_cfg.infer.test_thresh
        self.delta_thresh = model_cfg.infer.delta_thresh

        # load a model from file
        print(f"Loading model from {model_pkl}...")
        self.model = utils.load_pkl(model_pkl)
        self.model_file = model_pkl
        self.model_type = model_cfg.model.model_class

    def extract(self, batch: tuple):
        """
        Extract features from a batch of data.

        Args:
            batch (tuple): a tuple of torch tensors.
            network (BaseModel): the network to be used for extraction.
        """
        common_args = {
            "num_iters": self.num_iters,
            "fixed_preds": self.fixed_preds,
            "thresh": self.thresh,
            "delta_thresh": self.delta_thresh,
        }
        hybrid_feats, pc_feats, amort_feats = None, None, None

        if self.model_type == "DoubleAmortModel":
            img_batch, label_batch, amort_net_i = batch
            # reset model
            self.model.reset()
            # get amortization features if needed
            if self.use_amort:
                # set image
                self.model.set_img_batch_amort(img_batch)
                # propagate
                self.model.forward_mu(amort_net_i)
                # copy features
                amort_feats = self._get_features()

            else:
                # randomly initialize
                self.model.reset_mu(img_batch.size(0), self.init_std)

            # set image batch in the pc network
            self.model.set_img_batch(img_batch)

            # do the iteration process
            _, _ = self.model.test_updates(**common_args)
            # copy features
            if self.use_amort:
                # these features are from a hybrid model
                hybrid_feats = self._get_features()
            else:
                # these features are from a pc network
                pc_feats = self._get_features()

            if self.use_amort:
                # now need to get features for the pc network
                self.model.reset()
                # randomly initialize
                self.model.reset_mu(img_batch.size(0), self.init_std)
                self.model.set_img_batch(img_batch)
                self.model.test_updates(**common_args)

                pc_feats = self._get_features()

        if self.model_type == "asym":
            img_batch, label_batch = batch
            # TODO
            raise NotImplementedError("Asymmetrical models are not yet supported.")

        label_batch = label_batch.argmax(dim=1).cpu().numpy()
        return hybrid_feats, pc_feats, amort_feats, label_batch

    def _get_features(self):
        """
        Get the features from the model.
        """
        feats = [self.model.mus[l].cpu().numpy() for l in range(self.model.num_layers)]
        # last layer is the prediction of the input, not the state
        # we get it as a prediction from the layer above
        feats += [self.model.layers[-1](self.model.mus[-2]).cpu().numpy()]

        return feats


class HybridFeatures:
    """
    A class to store features extracted from a model.
    """

    def __init__(self, features: dict):
        # assert that the keys are correct
        assert set(features.keys()) == set(_FEAT_DICT_KEYS)
        self.hybrid = features["hybrid"]
        self.pc = features["pc"]
        self.amort = features["amort"]
        self.labels = features["labels"]
        self.classes: np.ndarray = np.unique(self.labels)
        self.n_classes: int = len(self.classes)
        self.n_layers = len(self.hybrid)
        self.distances = None
        self.distance_correlations = None
        self.classwise_distances = None
        self.distances_classwise = None
        self.distance_correlations_classwise = None

    def len(self):
        return len(self.labels)

    def __str__(self):
        return f"<Features> With {len(self.labels)} samples across {len(self.labels.unique())} classes."

    def __getitem__(self, network):
        return getattr(self, network)

    def visualize(self, idx, layer):
        """
        Visualize the features at a given index and layer.
        """
        # intialize figure
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # get the features
        h, p, a, l = (
            self.hybrid[layer],
            self.pc[layer],
            self.amort[layer],
            self.labels,
        )

        side = np.ceil(np.sqrt(h.shape[1])).astype(int)
        shape = (side, side)

        # plot the features
        # pad with zeros and reshape
        h_img = np.pad(h[idx], (0, side**2 - h[idx].shape[0]), mode="constant")
        h_img = h_img.reshape(shape)
        ax[0].imshow(h_img)
        ax[0].set_title("Hybrid")

        # now do the same for the other two networks
        p_img = np.pad(p[idx], (0, side**2 - p[idx].shape[0]), mode="constant")
        p_img = p_img.reshape(shape)
        ax[1].imshow(p_img)
        ax[1].set_title("PC")

        a_img = np.pad(a[idx], (0, side**2 - a[idx].shape[0]), mode="constant")
        a_img = a_img.reshape(shape)
        ax[2].imshow(a_img)
        ax[2].set_title("Amort")

        # Use class as title
        plt.suptitle(f"Class: {l[idx]}")
        return fig

    def get_distances(self):
        """
        Get the pairwise distances between the features.

        Args:
            features (Features): the features to be used.

        Returns:
            dict: a dictionary of distances.
        """
        distances = {"hybrid": [], "pc": [], "amort": []}
        for l in range(self.n_layers):
            # get the features for this layer
            h, p, a = (
                self.hybrid[l],
                self.pc[l],
                self.amort[l],
            )
            # get the distances
            distances["hybrid"].append(pdist(h))
            distances["pc"].append(pdist(p))
            distances["amort"].append(pdist(a))

        self.distances = distances

    def get_distances_classwise(self):
        """
        Get the pairwise distances between the features, classwise.

        Args:
            features (Features): the features to be used.

        Returns:
            dict: a dictionary of distances.
        """
        logging.info("Calculating classwise distances...")
        # get the unique classes
        classes = np.unique(self.labels)
        distances = dict()
        for c in classes:
            distances[c] = {"hybrid": [], "pc": [], "amort": []}
            # get the indices for this class
            idx = np.where(self.labels == c)[0]
            for l in range(self.n_layers):
                # get the features for this layer
                h, p, a, l = (self.hybrid[l], self.pc[l], self.amort[l], self.labels)
                # get the distances
                distances[c]["hybrid"].append(pdist(h[idx]))
                distances[c]["pc"].append(pdist(p[idx]))
                distances[c]["amort"].append(pdist(a[idx]))

        self.distances_classwise = distances

    def get_classwise_distances(self):
        """
        Get the pairwise distances between the classes, for each layer and network.
        """
        logging.info("Calculating classwise distances...")
        # get the unique classes
        distances = dict()
        for net in ["hybrid", "pc", "amort"]:
            distances[net] = []
            for l in range(self.n_layers):
                # get the layer centroid for each class
                centroids = np.zeros((len(self.classes), self.hybrid[l].shape[1]))
                for i, c in enumerate(self.classes):
                    idx = np.where(self.labels == c)[0]
                    centroids[i] = np.mean(self[net][l][idx], axis=0)
                # calculate the pairwise distances between centroids
                distances[net].append(pdist(np.array(centroids)))
        self.classwise_distances = distances

    def visualize_classwise_distances(
        self,
        networks: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ):
        """
        Visualize the classwise distances between the classes, per network and layer.
        """
        if self.classwise_distances is None:
            self.get_classwise_distances()
        assert self.classwise_distances is not None

        if labels is None:
            labels = [str(s) for s in self.classes]
        assert labels is not None
        if networks is None:
            networks = ["hybrid", "pc", "amort"]
        else:
            # assert that the networks are valid
            assert all([n in ["hybrid", "pc", "amort"] for n in networks])

        fig, axes = plt.subplots(
            len(networks), self.n_layers, figsize=(self.n_layers + 1, len(networks))
        )
        # if networks is lenght 1, then axes is not a list of lists
        if len(networks) == 1:
            axes = np.expand_dims(axes, axis=0)
        for l in range(self.n_layers):
            for n, net in enumerate(networks):
                axes[n, l].imshow(
                    squareform(self.classwise_distances[net][l]), cmap="viridis"
                )
                if n == 0:
                    axes[n, l].set_title(f"Layer {l}")
                if l == 0:
                    axes[n, l].set_ylabel(net)

                axes[n, l].set_xticks(range(len(labels)))
                axes[n, l].set_yticks(range(len(labels)))
                axes[n, l].set_xticklabels(labels)
                axes[n, l].set_yticklabels(labels)
                plt.setp(
                    axes[n, l].get_xticklabels(),
                    rotation=90,
                    ha="right",
                    rotation_mode="anchor",
                )
                plt.setp(
                    axes[n, l].get_yticklabels(),
                    rotation=0,
                    ha="right",
                    rotation_mode="anchor",
                )

        return fig

    def correlate_distances(self):
        """
        Create a correlation matrix for each layer, between all networks
        """
        if self.distances is None:
            self.get_distances()

        # initialize a correlation matrix
        corrmat = np.zeros((len(self.hybrid), 3, 3))
        for l in range(len(self.hybrid)):
            # get the distances for this layer
            h, p, a = (
                self.distances["hybrid"][l],
                self.distances["pc"][l],
                self.distances["amort"][l],
            )
            corrmat[l] = np.corrcoef([h, p, a])

        self.distance_correlations = corrmat

    def correlate_distances_classwise(self):
        """
        Create a correlation matrix for each layer, between all models, classwise
        """
        if self.distances_classwise is None:
            self.get_distances_classwise()

        # initialize a correlation matrix
        corrmat = dict()
        for c, dists in self.distances_classwise.items():
            corrmat[c] = np.zeros((len(self.hybrid), 3, 3))
            for l in range(len(self.hybrid)):
                # get the distances for this layer
                h, p, a = (
                    dists["hybrid"][l],
                    dists["pc"][l],
                    dists["amort"][l],
                )
                corrmat[c][l] = np.corrcoef([h, p, a])
        self.distance_correlations_classwise = corrmat

    def visualize_correlations(self):
        """
        Visualize the correlation matrix for the distances.
        """
        if self.distance_correlations is None:
            self.correlate_distances()
        assert self.distance_correlations is not None

        fig, ax = plt.subplots(1, len(self.hybrid), figsize=(15, 5))
        for l in range(len(self.hybrid)):
            ax[l].imshow(self.distance_correlations[l], cmap="viridis", vmin=-1, vmax=1)
            ax[l].set_title(f"Layer {l}")
            # set x and y axis labels
            ax[l].set_xticks([0, 1, 2], ["Hybrid", "PC", "Amort"])
            ax[l].set_yticks([0, 1, 2], ["Hybrid", "PC", "Amort"])
            # rotate the tick labels for the x axis
            plt.setp(
                ax[l].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor"
            )

        return fig

    def visualize_correlations_classwise(self):
        """
        Visualize the correlation matrix for the distances, classwise.
        """
        if self.distance_correlations_classwise is None:
            self.correlate_distances_classwise()
        assert self.distance_correlations_classwise is not None
        nclasses = len(self.distance_correlations_classwise)
        fig, axes = plt.subplots(
            len(self.hybrid),
            nclasses,
            sharex=True,
            sharey=True,
            layout="constrained",
        )

        # find vmin and vmax from all the correlation matrices
        vmin = np.min(
            [np.min(c) for c in self.distance_correlations_classwise.values()]
        )
        vmax = 1

        # now plot the correlation matrix, each row is a class, and each column is a layer
        for c, corr in self.distance_correlations_classwise.items():
            for l in range(len(self.hybrid)):
                axes[l, c].imshow(corr[l], cmap="viridis", vmin=vmin, vmax=vmax)
                axes[l, c].set_ylabel(f"Layer {l}")
                # set x labels
                axes[l, c].set_yticks([0, 1, 2], ["H", "PC", "A"])
                axes[l, c].set_xticks([0, 1, 2], ["H", "PC", "A"])
                # rotate the tick labels for the x axis
                plt.setp(
                    axes[l, c].get_xticklabels(),
                    rotation=90,
                    ha="right",
                    rotation_mode="anchor",
                )
                # the y label should be the class
                if l == 0:
                    axes[l, c].set_title(f"Class {c}")

        # add colorbar to the right side of the figure
        fig.colorbar(
            axes[0, 0].imshow(corr[0], vmin=vmin, vmax=vmax),
            ax=axes,
            orientation="horizontal",
        )

        return fig


def get_features(
    folder: str,
    pkl_name: Optional[str] = None,
    loader: Optional[torch.utils.data.DataLoader] = None,
) -> HybridFeatures:
    """
    Get features from the extractor.

    Args:


    Returns:

    """
    # load the configuration
    config = utils.load_json_config(os.path.join(folder, "config.json"))
    # get the pkl_path
    if pkl_name is None:
        pkl_name = "final_model.pkl"

    # load the extractor
    extractor = Extractor(config, os.path.join(folder, pkl_name))

    # get loader if not provided
    if loader is None:
        _, test_dataset = datasets.get_dataset(config)
        loader = datasets.get_dataloader(test_dataset, batch_size=512, shuffle=False)

    features = {"hybrid": [], "pc": [], "amort": [], "labels": []}
    for batch in tqdm(loader):
        h, p, a, l = extractor.extract(batch)
        features["hybrid"].append(h)
        features["pc"].append(p)
        features["amort"].append(a)
        features["labels"].append(l)

    # stack the features
    for k in features.keys():
        # labels are simple
        if k == "labels":
            features[k] = np.concatenate(features[k])
        else:
            # everything else is complicated, because there are many layers within each batch
            features[k] = [
                np.concatenate([f[l] for f in features[k]])
                for l in range(len(features[k][0]))
            ]

    return HybridFeatures(features)


def plot_classwise_tsne(
    features: List[HybridFeatures],
    model_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    output_folder: str = "results",
):
    """ "
    Perform tSNE on the classwise features and plot them.

    Args:
        features (List[HybridFeatures]): a list of features from different models.
        model_names (List[str]): a list of model names.
        class_names (List[str]): a list of class names.
        output_folder (str): the output folder to save the plots.

    """
    n_layers = features[0].n_layers
    n_classes = features[0].n_classes
    class_labels = features[0].labels

    if model_names is None:
        model_names = [str(i) for i in range(len(features))]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    # we loop per layer and class, all three models are embedded in the same space
    # we also loop per subnetwork

    for net in ["hybrid", "pc", "amort"]:
        # intialize figure without axes or labels
        fig, axes = plt.subplots(
            n_layers,
            n_classes,
            figsize=(14, 4),
        )

        print(f"Getting tSNE for all models, subnetwork {net}")
        for layer in range(n_layers):
            # get the features of all three models together
            all_features = np.concatenate([f[net][layer] for f in features], axis=0)
            # intialize the tsne embedder
            tsne = TSNE(n_components=2, random_state=0)
            # fit the tsne
            print(f"Getting tSNE for Layer {layer}")
            tsne_features = tsne.fit_transform(all_features)

            for i, c in enumerate(range(n_classes)):
                # now get each model's embedding from the tsne_features
                for mi, m in enumerate(model_names):
                    # get the model tsne_features
                    model_features = tsne_features[
                        mi * len(class_labels) : (mi + 1) * len(class_labels)
                    ]
                    # now get the features of the class
                    class_features = model_features[class_labels == c]
                    # average the features
                    class_features = np.mean(class_features, axis=0)
                    # plot
                    axes[layer, i].scatter(
                        class_features[:, 0], class_features[:, 1], label=m, alpha=0.5
                    )
                    if layer == 0:
                        axes[layer, i].set_title(class_names[i])
                    # remove labels from both axes
                    axes[layer, i].set_xticks([])
                    axes[layer, i].set_yticks([])
                    # add y title to the leftmost plot
                    if i == 0:
                        axes[layer, i].set_ylabel(f"Layer {layer}")

        # add legend to the right side of the plot
        axes[0, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Model")

        # save the figure
        fig.savefig(f"{output_folder}/tsne_features_{net}.png")


def plot_classwise_rsa(
    features: List[HybridFeatures],
    model_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    output_folder: str = "results",
):
    """
    Perform RSA on the features and plot them.

    Args:
        features (List[HybridFeatures]): a list of features from different models.
        model_names (List[str]): a list of model names.
        class_names (List[str]): a list of class names.
        output_folder (str): the output folder to save the plots.
    """

    n_layers = features[0].n_layers
    n_classes = features[0].n_classes
    if model_names is None:
        model_names = [str(i) for i in range(len(features))]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    # do it for each network
    for net in ["hybrid", "pc", "amort"]:
        for i, f in enumerate(features):
            f.get_distances_classwise()
            f.visualize_classwise_distances(labels=class_names)
            # make the figure a little bigger
            plt.gcf().set_size_inches(8, 6)
            plt.savefig(
                f"{output_folder}/{model_names[i]}_distances_classwise_{net}.png"
            )

        # intialize figure (one subplot per layer)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        # Now the fun part
        corrs = []
        # loop through the layers
        for l in range(n_layers):
            # get the classwise_distances for each model
            layer_distances = [f.classwise_distances[net][l] for f in features]
            # stack and get correlation matrix
            layer_distances = np.stack(layer_distances)
            # get the correlation matrix
            corrs.append(np.corrcoef(layer_distances))

        # calculate highest and lowest correlations in corrs
        max_corr = np.max([np.max(c) for c in corrs])
        min_corr = np.min([np.min(c) for c in corrs])

        # plot the correlation matrices
        for i, c in enumerate(corrs):
            im = axes[i].imshow(c, vmin=min_corr, vmax=max_corr)
            axes[i].set_title(f"Layer {i}")
            # put the actual values in the plot
            for j in range(3):
                for k in range(3):
                    axes[i].text(
                        j, k, f"{c[j, k]:.2f}", ha="center", va="center", color="w"
                    )
            # put model_names on both x and y axes
            axes[i].set_xticks(np.arange(3))
            axes[i].set_yticks(np.arange(3))
            axes[i].set_xticklabels(model_names)
            axes[i].set_yticklabels(model_names)

            # add colorbar to the bottom
        fig.colorbar(im, ax=axes, orientation="horizontal")
        fig.savefig(f"{output_folder}/RSA_correlations_classwise_{net}.png")
