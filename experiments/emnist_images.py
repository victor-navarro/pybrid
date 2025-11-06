""" Creates a mosaic of EMNIST images for the chosen classes."""

from typing import List
import os
import numpy as np
from PIL import Image
from pybrid.datasets import cEMNIST
from pybrid.utils import make_mosaic


def emnist_mosaic(
    classes: List[int],
    num_images: int,
    output_dir: str,
):
    """
    Create a mosaic of EMNIST images for the chosen classes.

    Args:
        classes: List of classes to include in the mosaic.
        num_images: Number of images per class.
        output_dir: Directory to save the mosaic.
    """

    # Set seed
    np.random.seed(0)

    # Load the dataset
    data = cEMNIST(train=True, split="balanced", labels=classes)

    # Pick images per class
    imgs = []
    for c in classes:
        idx = np.where(data.targets == c)[0]
        idx = np.random.choice(idx, num_images, replace=False)
        cimgs = data.data[idx].numpy()
        # transpose to (N, H, W) from (N, W, H)
        cimgs = cimgs.transpose(0, 2, 1)
        imgs.append(cimgs)

    imgs = np.vstack(imgs)

    # Create the mosaic
    mosaic = make_mosaic(
        imgs,
        ncol=len(classes),
        nrow=num_images,
        col_major=True,
    )

    # Save the mosaic
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "emnist_mosaic.png")
    Image.fromarray(mosaic).save(output_path)


if __name__ == "__main__":
    cs = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 28]
    emnist_mosaic(
        classes=cs,
        num_images=4,
        output_dir="plots",
    )
