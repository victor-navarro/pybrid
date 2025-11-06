""" A module to plot results from postprocessing functions"""

import os
from typing import Tuple, List, Optional, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_iterations(
    csv_path: str, metric: str = "accuracy"
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots label accuracy data across iterations.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        A tuple containing fig and ax objects
    """
    # load the data
    data = pd.read_csv(csv_path)
    # identify number of networks
    networks = data["network"].unique()
    max_iters = data["iteration"].max()
    fig, ax = plt.subplots(
        1, len(networks), figsize=(6 * len(networks), 6), layout="constrained"
    )
    if len(networks) == 1:
        ax = [ax]
    for i, network in enumerate(networks):
        network_data = data[data["network"] == network]
        for class_label in network_data["class"].unique():
            class_data = network_data[network_data["class"] == class_label]
            if network == "amort":
                ax[i].plot(
                    class_data["iteration"],
                    class_data[metric],
                    "o",
                    label=class_label,
                )
            else:
                ax[i].plot(
                    class_data["iteration"],
                    class_data[metric],
                    label=class_label,
                )

        # put legend outside of plot
        ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax[i].set_title(network)
        ax[i].set_xlabel("Iteration")
        ax[i].set_ylabel(metric)
        ax[i].set_ylim(-0.01, 1.01)
        ax[i].set_xlim(-1, max_iters)

    return fig, ax

def plot_epochs(folder: str, metric: str = "accuracy") -> Tuple[plt.Figure, plt.Axes]:
    """Plots label accuracies at the end of inference,
    for all epochs of a model in a folder."""
    # get all csv files in the folder
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    full_data = pd.DataFrame()
    for f in csv_files:
        # get epoch number from filename
        epoch = int(f.split(".")[0])
        data = pd.read_csv(os.path.join(folder, f))
        # filter to contain only the last iteration per class/network
        data = data.groupby(["network", "class"]).last().reset_index()
        data["epoch"] = epoch
        full_data = pd.concat([full_data, data])
    # sort the data by class, epoch
    full_data = full_data.sort_values(by=["class", "epoch"])
    # identify number of networks
    networks = full_data["network"].unique()
    max_epochs = full_data["epoch"].max()
    fig, ax = plt.subplots(
        1, len(networks), figsize=(6 * len(networks), 6), layout="constrained"
    )
    if len(networks) == 1:
        ax = [ax]
    for i, network in enumerate(networks):
        network_data = full_data[full_data["network"] == network]
        for class_label in network_data["class"].unique():
            class_data = network_data[network_data["class"] == class_label]
            ax[i].plot(
                class_data["epoch"],
                class_data[metric],
                label=class_label,
            )
        # add average line
        epoch_means = (
            network_data[["epoch", metric]].groupby("epoch").mean().reset_index()
        )
        ax[i].plot(
            epoch_means["epoch"], epoch_means[metric], label="AVG", color="black"
        )
        # put legend outside of plot
        ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax[i].set_title(network)
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel(metric)
        ax[i].set_ylim(-0.01, 1.01)
        ax[i].set_xlim(-1, max_epochs)
    return fig, ax


def plot_variable(
    data: pd.DataFrame,
    what: str,
    by: str,
    group: Optional[List[str]] = None, 
    ax: Optional[plt.Axes] = None,
    cmap: str = "tab20",
):
    """Plot variable.

    Args:
        data (pd.DataFrame): DataFrame containing the  results.
        by (str | None): Column name to group by. If None, use the index.
        group: (List[str] | None): List of column names to group by. If None, use the index.
        what (str): Column name to plot. Default is "accuracy".

    Returns:
        A tuple of (fig, ax) where fig is the figure and ax is the axes.
    """

    assert str(by) in data.columns, f"{by} not in data columns"
    assert str(what) in data.columns, f"{what} not in data columns"
    if group is not None:
        if isinstance(group, str):
            group = [group]
        assert all([str(g) in data.columns for g in group]), f"{group} not in data columns"
    else:
        group = []

    if ax is None:
        # try to get the current axes
        ax = plt.gca()
        # if no axes, create a new one
        if ax is None:
            _, ax = plt.subplots(layout="constrained")

    # calculate mean and sem for each group
    data_grouped = data.groupby([str(by)] + group)[what].agg(["mean", "sem"]).reset_index()
    # group the data by group
    if len(group) > 0:
        data_grouped = data_grouped.groupby(group)
        # get colors
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(data_grouped)))
        # plot the data
        for i, (name, group_data) in enumerate(data_grouped):
            # plot the mean as points and sem as ribbon
            ax.plot(
                group_data[by],
                group_data["mean"],
                "-o",
                label=name[0],
                color=colors[i],
            )
            ax.fill_between(
                group_data[by],
                group_data["mean"] - group_data["sem"],
                group_data["mean"] + group_data["sem"],
                alpha=0.2,
                color=colors[i],
            )

            # add legend
            ax.legend(loc="best", bbox_to_anchor=(1, 1))
            
    else:
        # plot the mean and sem
        ax.plot(
            data_grouped[by],
            data_grouped["mean"],
            "-o",
            label=data_grouped[by].name,
        )
        ax.fill_between(
            data_grouped[by],
            data_grouped["mean"] - data_grouped["sem"],
            data_grouped["mean"] + data_grouped["sem"],
            alpha=0.2,
        )
    # add vertical line at 0.5
    ax.set_ylabel(what.capitalize())
    ax.set_xlabel(by.capitalize())

    return ax

def multiplot(
    data: pd.DataFrame,
    multi: str,
    plot_fun: Callable = plot_variable,
    **kwargs
):
    """Plot multiple variables in a single figure.

    Args:
        data (pd.DataFrame): DataFrame containing the  results.
        multi (str): Column name to create panels from.
        plot_fun (Callable): Function to plot the data. Default is plot_variable.
        **kwargs: Additional arguments to pass to the plot function.

    Returns:
        A tuple of (fig, ax) where fig is the figure and ax is the axes.
    """
    # check how many panels are needed
    n_panels = data[multi].nunique()
    fig, axes = plt.subplots(1, n_panels, layout="constrained", sharey=True)

    for i, (name, group_data) in enumerate(data.groupby(multi)):
        # call the plot function
        plot_fun(group_data, ax=axes[i], **kwargs)
        # set the title
        axes[i].set_title(name)

    return fig, axes
