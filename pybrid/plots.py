""" Plots for CEMNIST experiments. """

import os
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pybrid import utils


def plot_metrics(data, metrics=None, path=None, trapped=False):
    """
    Plot metrics from a dictionary of data.

    Args:
        data (dict): Dictionary of data.
        metrics (list): List of metrics to plot.
        path (str): Path to save the plot.
        trapped (bool): Whether the data is from a trapped experiment.
    """

    if metrics is None:
        if trapped:
            metrics = [
                "hybrid_acc",
                "pc_acc",
                "amort_acc",
                "pc_losses",
                "pc_errs",
                "amort_losses",
                "amort_errs",
                "init_errs",
                "final_errs",
                "trap_hybrid_acc",
                "trap_pc_acc",
                "trap_amort_acc",
                "trapped_hybrid_acc_diff",
                "trapped_pc_acc_diff",
                "trapped_amort_acc_diff",
            ]
        else:
            metrics = [
                "hybrid_acc",
                "pc_acc",
                "amort_acc",
                "pc_losses",
                "pc_errs",
                "amort_losses",
                "amort_errs",
                "init_errs",
                "final_errs",
            ]

    plot_side = int(np.ceil(np.sqrt(len(metrics))))
    _, axes = plt.subplots(plot_side, plot_side)

    batch_ids = data.batch_idx
    for ax, m in zip(axes.flat, metrics):
        ax.set_title(f"{m}")
        ax.plot(batch_ids, data[m], "o")
    plt.tight_layout()

    plt.savefig(path)
    plt.close("all")


def plot_json_metrics(jsonpath, metrics=None, path=None):
    """
    Plot performance metrics from a JSON file.

    Args:
        jsonpath (str): Path to the JSON file.
        metrics (list): List of metrics to plot.
        path (str): Path to save the plot.
    """
    data = utils.to_attr_dict(utils.load_json(jsonpath))
    if path is None:
        path = os.path.join(os.path.dirname(jsonpath), "performance_plot.png")
    plot_metrics(data, metrics, path)


def create_gif(
    png_folder,
    gif_path=None,
    pattern="",
    font_path="arial.ttf",
    font_size=20,
    font_fill="black",
):
    """
    Create a GIF from a folder of PNG files.

    Args:
        png_folder (str): Path to the folder containing PNG files.
        pattern (str): Pattern to match the PNG files.
        gif_path (str): Path to save the GIF.
        font_path (str): Path to the font file.
        font_size (int): Font size.
    """
    if gif_path is None:
        gif_path = os.path.join(png_folder, "output.gif")
    # Get list of PNG files in the folder
    pngs = [file for file in os.listdir(png_folder) if file.endswith(".png")]
    # Filter the files based on the pattern
    pngs = [file for file in pngs if pattern in file]
    png_files = sorted(
        [file for file in pngs],
        key=natural_sort_key,
    )

    # Sort the files by name
    # png_files.sort()

    images = []
    font = ImageFont.truetype(font_path, font_size)

    for png_file in png_files:
        # Open each PNG file
        img = Image.open(os.path.join(png_folder, png_file))
        draw = ImageDraw.Draw(img)

        # Get the filename without extension
        filename = os.path.splitext(png_file)[0]

        # Get the size of the text
        text_width, _ = draw.textsize(filename, font=font)

        # Calculate position to place the text (top right corner)
        x = img.width - text_width - 10
        y = 10

        # Draw the text on the image
        draw.text((x, y), filename, fill=font_fill, font=font)

        # Append the image to the list
        images.append(img)

    # Save the GIF
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
    )
    print("Successfully created", gif_path)


def natural_sort_key(s):
    """Key function for natural sorting."""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def animate(png_folder, webm_path=None, font_path="arial.ttf", font_size=20, **kwargs):
    if webm_path is None:
        webm_path = os.path.join(png_folder, "output.webm")
        gif_path = os.path.join(png_folder, "output.gif")
    else:
        gif_path = webm_path.replace(".webm", ".gif")
    create_gif(
        png_folder=png_folder,
        gif_path=gif_path,
        font_path=font_path,
        font_size=font_size,
        **kwargs,
    )
    print("Converting GIF to WEBM")
    subprocess.run(["ffmpeg", "-y", "-i", gif_path, webm_path])
    print("Successfully created", webm_path)


def compare_metrics(json_paths, labels=None, metrics=None, path=None, trapped=False):
    """
    Compare metrics from multiple JSON files.

    Args:
        json_paths (list): List of paths to the JSON files.
        labels (list): List of labels for the models in each JSON file.
        metrics (list): List of metrics to plot.
        path (str): Path to save the plot.
        trapped (bool): Whether the data is from a trapped experiment.
    """
    if labels is None:
        labels = [f"Model {i}" for i in range(len(json_paths))]

    if metrics is None:
        if trapped:
            metrics = [
                "hybrid_acc",
                "pc_acc",
                "amort_acc",
                "pc_losses",
                "pc_errs",
                "amort_losses",
                "amort_errs",
                "init_errs",
                "final_errs",
                "trap_hybrid_acc",
                "trap_pc_acc",
                "trap_amort_acc",
                "trapped_hybrid_acc_diff",
                "trapped_pc_acc_diff",
                "trapped_amort_acc_diff",
            ]
        else:
            metrics = [
                "hybrid_acc",
                "pc_acc",
                "amort_acc",
                "pc_losses",
                "pc_errs",
                "amort_losses",
                "amort_errs",
                "init_errs",
                "final_errs",
            ]

    plot_side = int(np.ceil(np.sqrt(len(metrics))))
    _, axes = plt.subplots(plot_side, plot_side, layout="constrained")

    for ax, m in zip(axes.flat, metrics):
        ax.set_title(f"{m}")
        for i, json_path in enumerate(json_paths):
            data = utils.to_attr_dict(utils.load_json(json_path))
            batch_ids = data.batch_idx
            ax.plot(batch_ids, data[m], "o", label=labels[i])

    # put legend
    axes.flat[-1].legend(loc="upper right")
    plt.savefig(path)
    plt.close("all")


def compare_twin_metrics(
    json_paths, labels=None, metrics=None, path=None, plot_dims=None, plot_size=(6, 10)
):
    """
    Compare metrics from multiple JSON files.

    Args:
        json_paths (list): List of paths to the JSON files.
        labels (list): List of labels for the models in each JSON file.
        metrics (list): List of metrics to plot.
        path (str): Path to save the plot.
        plot_dims (tuple): Dimensions of the plot grid.
        plot_size (tuple): Size of the plot in inches.
    """
    assert len(json_paths) == 2, "Only two models can be compared"

    if labels is None:
        labels = [f"Model {i}" for i in range(len(json_paths))]

    if metrics is None:
        metrics = [
            "hybrid_acc",
            "pc_acc",
            "amort_acc",
            "pc_losses",
            "pc_errs",
            "amort_losses",
            "amort_errs",
            "init_errs",
            "final_errs",
        ]

    if plot_dims is None:
        plot_side = int(np.ceil(np.sqrt(len(metrics))))
        plot_dims = (plot_side + 1, plot_side)

    _, axes = plt.subplots(*plot_dims, layout="constrained")

    metrics = metrics + [f"{m}_diff" for m in metrics]
    # initialize a dictionary to store the diff metrics
    metrics_diff = {m: 0 for m in metrics if m.endswith("_diff")}

    for ax, m in zip(axes.flat, metrics):
        ax.set_title(f"{m}")
        if not m.endswith("_diff"):
            for i, json_path in enumerate(json_paths):
                data = utils.to_attr_dict(utils.load_json(json_path))
                batch_ids = data.batch_idx
                ax.plot(batch_ids, data[m], "o", label=labels[i])
                # compute the difference according to i
                if i == 0:
                    metrics_diff[f"{m}_diff"] = data[m]
                else:
                    metrics_diff[f"{m}_diff"] = np.array(
                        metrics_diff[f"{m}_diff"]
                    ) - np.array(data[m])
        else:
            ax.plot(batch_ids, metrics_diff[m], "o", label=f"{labels[0]} - {labels[1]}")

    # put legend
    axes.flat[0].legend(loc="upper left")
    axes.flat[-1].legend(loc="upper right")
    # change the size of the plot
    plt.gcf().set_size_inches(*plot_size)
    plt.savefig(path)
    plt.close("all")
