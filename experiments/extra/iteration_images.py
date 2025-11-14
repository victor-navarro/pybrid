"""A function to generate images across iterations for any given model pickle"""

import os
from typing import List, Optional
import torch
from PIL import Image
import numpy as np
from pybrid import utils
from pybrid import datasets
from pybrid.models import DoubleAmortModel


def iteration_images(
    model_pkl: str,
    model_cfg_path: str,
    out_folder: str,
    classes: Optional[List[int]] = None,
    steps: Optional[List[int]] = None,
    from_labels: bool = False,
    ncol: int = 1,
    nrow: int = 12,
):
    """Generate iteration images for a given model pickle.

    Arguments:
        model_pkl: Path to the model pickle.
        model_cfg_path: Path to the model config json.
        out_folder: Path to the output folder.
        classes: List of classes to show (if None, use all).
        steps: List of target iteration steps to show (if None, save all).
        from_labels: Whether to reconstruct from labels or locally (layer above the image).
        ncol: Number of columns in the mosaic.
        nrow: Number of rows in the mosaic.
    """

    # make the output folder
    os.makedirs(out_folder, exist_ok=True)
    # load configuration
    cfg = utils.load_json_config(model_cfg_path)
    # set the seed
    utils.seed(cfg.exp.seed)
    # define model pickle
    model = utils.load_pkl(model_pkl)
    # load dataset
    _, test_dataset = datasets.get_dataset(cfg)
    # get dataloader
    test_loader = datasets.get_dataloader(test_dataset, cfg.optim.batch_size, False)
    # get set of images for inference
    img_set = utils.get_infer_set(test_loader)

    # select the classes if specified
    if classes is not None:
        # filter the image set
        img_set = [i[classes] for i in img_set]
        nrow = len(classes)

    # get images
    if from_labels:
        hybrid_imgs, pc_imgs, amort_imgs, label_imgs, imgs = _get_label_images(
            model, img_set
        )
    else:
        hybrid_imgs, pc_imgs, amort_imgs, imgs = _get_local_images(model, img_set)

    # now make the mosaics
    img_mos = utils.make_mosaic(
        np.concatenate(imgs),
        nrow=nrow,
        ncol=ncol,
    )
    # save the image
    img_mos = Image.fromarray(img_mos)
    img_mos.save(os.path.join(out_folder, "image_set.png"))

    # make mosaic of label_imgs if they exist
    if from_labels:
        label_mos = utils.make_mosaic(
            np.concatenate(label_imgs),
            nrow=nrow,
            ncol=ncol,
        )
        # save the image
        label_mos = Image.fromarray(label_mos)
        label_mos.save(os.path.join(out_folder, "label_imgs.png"))

    # make the mosaic for the amortized images
    amort_mos = utils.make_mosaic(
        np.concatenate(amort_imgs),
        nrow=nrow,
        ncol=ncol,
    )
    # save the image
    amort_mos = Image.fromarray(amort_mos)
    amort_mos.save(os.path.join(out_folder, "amort_imgs.png"))

    # make the mosaics for the pc_imgs
    pc_mos = []
    if steps is None:
        steps = list(range(0, len(pc_imgs)))
    for i in steps:
        pc_mos.append(
            utils.make_mosaic(
                np.array(pc_imgs[i]),
                nrow=nrow,
                ncol=ncol,
            )
        )
    # make yet another mosaic with the iteration steps
    pc_mos = utils.make_mosaic(
        np.array(pc_mos),
        nrow=1,
        ncol=len(pc_mos),
    )
    # save the image
    pc_mos = Image.fromarray(pc_mos)
    pc_mos.save(os.path.join(out_folder, "pc_imgs.png"))

    # make the mosaics for the hybrid imgs
    hybrid_mos = []
    for i in steps:
        hybrid_mos.append(
            utils.make_mosaic(
                np.array(hybrid_imgs[i]),
                nrow=nrow,
                ncol=ncol,
            )
        )
    # make yet another mosaic with the iteration steps
    hybrid_mos = utils.make_mosaic(
        np.array(hybrid_mos),
        nrow=1,
        ncol=len(hybrid_mos),
    )
    # save the image
    hybrid_mos = Image.fromarray(hybrid_mos)
    hybrid_mos.save(os.path.join(out_folder, "hybrid_imgs.png"))


# here's a function that reconstructs images from labels
def _get_label_images(
    model: DoubleAmortModel,
    infer_set: List[torch.Tensor],
    num_iters: int = 100,
    init_sd: float = 0.05,
):
    imgs, labels, nets = infer_set
    model.reset()
    # do amortization first
    model.set_img_batch_amort(imgs)
    model.forward_mu(nets)
    amort_imgs = [utils.postprocess_prediction(model.backward(model.mus[0]))]

    # do pc next
    model.reset()
    model.reset_mu(imgs.size(0), init_sd)
    model.set_img_batch(imgs)
    pc_imgs = []
    pc_imgs.append(utils.postprocess_prediction(model.backward(model.mus[0])))
    for _ in range(num_iters - 1):
        # step the model
        model.test_updates(num_iters=1)
        pc_imgs.append(utils.postprocess_prediction(model.backward(model.mus[0])))

    # now do hybrid
    model.reset()
    model.set_img_batch_amort(imgs)
    model.forward_mu(nets)
    model.set_img_batch(imgs)
    hybrid_imgs = []
    hybrid_imgs.append(utils.postprocess_prediction(model.backward(model.mus[0])))
    for _ in range(num_iters - 1):
        # step the model
        model.test_updates(num_iters=1)
        hybrid_imgs.append(utils.postprocess_prediction(model.backward(model.mus[0])))

    # one with labels
    label_imgs = [utils.postprocess_prediction(model.backward(labels))]

    imgs = [utils.postprocess_prediction(img) for img in imgs]
    return hybrid_imgs, pc_imgs, amort_imgs, label_imgs, imgs


# here's a function that reconstructs images locally
def _get_local_images(
    model: DoubleAmortModel,
    infer_set: List[torch.Tensor],
    num_iters: int = 100,
    init_sd: float = 0.05,
):
    imgs, _, nets = infer_set
    model.reset()
    # do amortization first
    model.set_img_batch_amort(imgs)
    model.forward_mu(nets)
    amort_imgs = [utils.postprocess_prediction(model.layers[-1](model.mus[-2]))]

    # do pc next
    model.reset()
    model.reset_mu(imgs.size(0), init_sd)
    model.set_img_batch(imgs)
    pc_imgs = []
    pc_imgs.append(utils.postprocess_prediction(model.layers[-1](model.mus[-2])))
    for _ in range(num_iters - 1):
        # step the model
        model.test_updates(num_iters=1)
        pc_imgs.append(utils.postprocess_prediction(model.layers[-1](model.mus[-2])))

    # now do hybrid
    model.reset()
    model.set_img_batch_amort(imgs)
    model.forward_mu(nets)
    model.set_img_batch(imgs)
    hybrid_imgs = []
    hybrid_imgs.append(utils.postprocess_prediction(model.layers[-1](model.mus[-2])))
    for _ in range(num_iters):
        # step the model
        model.test_updates(num_iters=1)
        hybrid_imgs.append(
            utils.postprocess_prediction(model.layers[-1](model.mus[-2]))
        )

    imgs = [utils.postprocess_prediction(img) for img in imgs]
    return hybrid_imgs, pc_imgs, amort_imgs, imgs


if __name__ == "__main__":
    stps = [0, 49, 99]
    cls = [4]
    # FROM LABELS
    # BY BATCHES
    base_folders = {
        "normal": "results/exp_1_norm_e1/normal_twin/0/",
        "swapped": "results/exp_1_norm_e1/swapped_twin/0/",
    }
    model_pkls = {
        "0": "model_0.pkl",
        "1": "model_1.pkl",
        "15": "model_15.pkl",
        "29": "model_29.pkl",
        "43": "model_43.pkl",
        "56": "final_model.pkl",
    }
    for model_name, base_folder in base_folders.items():
        for batch, pkl in model_pkls.items():
            # set the model pkl
            pkl = os.path.join(base_folder, pkl)
            # set the config path
            cfg_path = os.path.join(base_folder, "config.json")
            # set the output folder
            plot_path = os.path.join(
                "plots/extra/iteration_images/from_label", model_name, f"batch_{batch}"
            )
            iteration_images(
                pkl,
                cfg_path,
                plot_path,
                steps=stps,
                classes=cls,
                from_labels=True,
            )

    # BY EPOCHS
    base_folders = {
        "normal": "results/exp_1_norm/normal_twin/0/",
        "swapped": "results/exp_1_norm/swapped_twin/0/",
    }
    model_pkls = {
        "1": "model_0.pkl",
        "10": "model_9.pkl",
        "20": "model_19.pkl",
        "30": "model_29.pkl",
        "40": "model_39.pkl",
        "50": "model_49.pkl",
    }
    for model_name, base_folder in base_folders.items():
        for batch, pkl in model_pkls.items():
            # set the model pkl
            pkl = os.path.join(base_folder, pkl)
            # set the config path
            cfg_path = os.path.join(base_folder, "config.json")
            # set the output folder
            plot_path = os.path.join(
                "plots/extra/iteration_images/from_label", model_name, f"epoch_{batch}"
            )
            iteration_images(
                pkl,
                cfg_path,
                plot_path,
                steps=stps,
                classes=cls,
                from_labels=True,
            )

    # LOCALLY
    # BY BATCHES
    base_folders = {
        "normal": "results/exp_1_norm_e1/normal_twin/0/",
        "swapped": "results/exp_1_norm_e1/swapped_twin/0/",
    }
    model_pkls = {
        "0": "model_0.pkl",
        "1": "model_1.pkl",
        "15": "model_15.pkl",
        "29": "model_29.pkl",
        "43": "model_43.pkl",
        "56": "final_model.pkl",
    }
    for model_name, base_folder in base_folders.items():
        for batch, pkl in model_pkls.items():
            # set the model pkl
            pkl = os.path.join(base_folder, pkl)
            # set the config path
            cfg_path = os.path.join(base_folder, "config.json")
            # set the output folder
            plot_path = os.path.join(
                "plots/extra/iteration_images/local", model_name, f"batch_{batch}"
            )
            iteration_images(
                pkl,
                cfg_path,
                plot_path,
                steps=stps,
                classes=cls,
                from_labels=False,
            )

    # BY EPOCHS
    base_folders = {
        "normal": "results/exp_1_norm/normal_twin/0/",
        "swapped": "results/exp_1_norm/swapped_twin/0/",
    }
    model_pkls = {
        "1": "model_0.pkl",
        "10": "model_9.pkl",
        "20": "model_19.pkl",
        "30": "model_29.pkl",
        "40": "model_39.pkl",
        "50": "model_49.pkl",
    }
    for model_name, base_folder in base_folders.items():
        for batch, pkl in model_pkls.items():
            # set the model pkl
            pkl = os.path.join(base_folder, pkl)
            # set the config path
            cfg_path = os.path.join(base_folder, "config.json")
            # set the output folder
            plot_path = os.path.join(
                "plots/extra/iteration_images/local", model_name, f"epoch_{batch}"
            )
            iteration_images(
                pkl,
                cfg_path,
                plot_path,
                steps=stps,
                classes=cls,
                from_labels=False,
            )
