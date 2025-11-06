""" A short script to generate images for the iteration loop for any given model pickle"""

import os
from typing import List, Optional
import torch
import imageio
import matplotlib.pyplot as plt
import numpy as np
from pybrid import utils
from pybrid import datasets
from pybrid.models import DoubleAmortModel

# load configuration
cfg = utils.load_json_config("results/exp_1/progenitor/0/config.json")

# define model pickle
MODEL_PKL = "results/exp_1/progenitor/0/model_9.pkl"
# load model
mod = utils.load_pkl(MODEL_PKL)

# load dataset
_, test_dataset, _ = datasets.get_dataset(cfg)
# get dataloader
test_loader = datasets.get_dataloader(test_dataset, cfg.optim.batch_size, False)
# get set of images for inference
img_set = utils.get_infer_set(test_loader)


# here's the function
def get_choices(
    model: DoubleAmortModel,
    infer_set: List[torch.Tensor],
    num_iters: int = 100,
    init_sd: float = 0.05,
):
    """
    Get images for each component of the model, for a given number of iterations
    """
    imgs, labels, nets = infer_set
    model.reset()
    # do amortization first
    amort_preds = model.forward(imgs, nets)

    # do pc next
    model.reset()
    model.reset_mu(imgs.size(0), init_sd)
    model.set_img_batch(imgs)
    pc_preds = []
    for _ in range(num_iters):
        # step the model
        model.test_updates(num_iters=1)
        pc_preds.append(model.mus[0])

    # now do hybrid
    model.reset()
    model.set_img_batch_amort(imgs)
    model.forward_mu(nets)
    model.set_img_batch(imgs)
    hybrid_preds = []
    for _ in range(num_iters):
        # step the model
        model.test_updates(num_iters=1)
        hybrid_preds.append(model.mus[0])

    return hybrid_preds, pc_preds, [amort_preds], [labels]


# get images
hybrid_preds, pc_preds, amort_preds, preds = get_choices(mod, img_set)


# define a postprocessing function
def postprocess_choice(pred: torch.Tensor) -> np.ndarray:
    # just a softmax
    return torch.nn.functional.softmax(pred * 3.0, dim=1).detach().cpu().numpy()


# post process the images
hybrid_preds = [postprocess_choice(img) for img in hybrid_preds]
pc_preds = [postprocess_choice(img) for img in pc_preds]
amort_preds = [postprocess_choice(img) for img in amort_preds]
preds = [postprocess_choice(img) for img in preds]

# save images
OUT_FOLDER = "experiments/extra/iteration_preds/"
os.makedirs(OUT_FOLDER, exist_ok=True)


def save_prediction(
    preds: List[np.ndarray],
    target_class: int,
    folder: str,
    prefix: str,
    pretty_names: Optional[List[str]] = None,
):
    nclasses = len(preds[0])
    if pretty_names is None:
        pretty_names = [str(i) for i in range(nclasses)]

    # the idea is to create a matplotlib plot
    # with the predictions across time for the target class

    # get the predictions for the target class
    pred_list = [pred[target_class] for pred in preds]
    # transform to np.ndarray
    pred_list = np.array(pred_list)

    # create the plot
    fig, ax = plt.subplots(layout="constrained")
    ax.set_xlim(0, len(pred_list))
    ax.set_ylim(0, 1)

    # draw the lines for each class
    for i in range(nclasses):
        ax.plot(pred_list[:, i], label=pretty_names[i])

    # add axis labels
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Class Probability")

    # add legend to the right side of the plot
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Class")

    # resize the plot
    fig.set_size_inches(4, 3)

    # save the plot
    plt.savefig(f"{folder}/{prefix}_{target_class}.png", dpi=300)


class_names = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "A",
    "B",
    "C",
    "D",
    "E",
    "S",
]


# create images for pc and hybrid
save_prediction(hybrid_preds, 6, OUT_FOLDER, "hybrid", class_names)
save_prediction(pc_preds, 6, OUT_FOLDER, "pc", class_names)
