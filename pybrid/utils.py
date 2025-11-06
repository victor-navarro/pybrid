import os
import shutil
from dataclasses import asdict, is_dataclass
from typing import List, Optional, Tuple, Literal
import subprocess
import logging
import random
import json
import pickle
import numpy as np
import torch
from PIL import Image
from torchvision.datasets.utils import extract_archive
from pybrid.config import DefaultConfig


_device = "cuda" if torch.cuda.is_available() else "cpu"


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def get_device():
    return _device


def setup_experiment(cfg: DefaultConfig):
    setup_logging()
    cfg.exp.log_dir, cfg.exp.img_dir = setup_logdirs(cfg.exp.log_dir, cfg.exp.seed)
    seed(cfg.exp.seed)
    logging.info("Starting experiment @ %s [%s]", cfg.exp.log_dir, get_device())
    save_json(cfg.to_dict(), cfg.exp.log_dir + "/config.json")
    return cfg


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def setup_logdirs(dir_name, seed):
    seed_dir_name = dir_name + "/" + str(seed)
    img_seed_dir_name = dir_name + "/" + str(seed) + "/imgs"
    os.makedirs(seed_dir_name, exist_ok=True)
    os.makedirs(img_seed_dir_name, exist_ok=True)
    return seed_dir_name, img_seed_dir_name


def to_attr_dict(_dict):
    attr_dict = AttrDict()
    for k, v in _dict.items():
        if isinstance(v, dict):
            v = to_attr_dict(v)
        attr_dict[k] = v
    return attr_dict


def setup_logdir(dir_name, seed):
    seed_dir_name = dir_name + "/" + str(seed)
    os.makedirs(seed_dir_name, exist_ok=True)
    return seed_dir_name


def seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_tensor(tensor):
    return tensor.to(get_device()).float()


def flatten_array(array):
    return torch.flatten(torch.cat(array, dim=1))


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w") as file:
        json.dump(obj, file)


def load_json(path) -> dict:
    with open(path) as file:
        return json.load(file)


def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def run_mnist_dl(data_dir: str, subdir: str = "MNIST"):
    cmd = f"""curl https://www.di.ens.fr/~lelarge/MNIST.tar.gz -o MNIST.tar.gz
             tar -zxvf MNIST.tar.gz
             mv MNIST {data_dir}/{subdir}
             rm MNIST.tar.gz
        """
    logging.info("Downloading MNIST into %s/MNIST...", data_dir)
    subprocess.run(cmd, shell=True, check=True)


def run_emnist_dl(data_dir: str, subdir: str = "EMNIST"):
    """Download the EMNIST dataset."""
    # check if EMNIST is already downloaded
    if os.path.exists(f"{data_dir}/{subdir}/raw/"):
        if os.listdir(f"{data_dir}/{subdir}/raw/") != []:
            logging.info("EMNIST already downloaded.")
            return
    os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)
    cmd = f"""curl https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip -o EMNIST.zip
             unzip EMNIST.zip
             mv gzip/ {data_dir}/{subdir}/gzip/
             rm EMNIST.zip
        """
    logging.info("Downloading EMNIST into %s/EMNIST...", data_dir)
    subprocess.run(cmd, shell=True, check=True)

    gzip_folder = os.path.join(data_dir, subdir, "gzip")
    raw_folder = os.path.join(data_dir, subdir, "raw")
    os.makedirs(raw_folder, exist_ok=True)
    for gzip_file in os.listdir(gzip_folder):
        if gzip_file.endswith(".gz"):
            extract_archive(
                os.path.join(gzip_folder, gzip_file),
                raw_folder,
            )
    shutil.rmtree(gzip_folder)


def get_act_fn(act_fn):
    if act_fn == "linear":
        return Linear()
    elif act_fn == "relu":
        return ReLU()
    elif act_fn == "tanh":
        return Tanh()
    else:
        raise ValueError(f"invalid act fn {act_fn}")


class Activation(object):
    def forward(self, inp):
        raise NotImplementedError

    def deriv(self, inp):
        raise NotImplementedError

    def __call__(self, inp):
        return self.forward(inp)


class Linear(Activation):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return set_tensor(torch.ones((1,)))


class ReLU(Activation):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        out = self(inp)
        out[out > 0] = 1.0
        return out


class Tanh(Activation):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0


def get_mapping(
    split: Literal["alphanumerical", "alphanumerical_rev"] = "alphanumerical",
    map_type: Literal["by_class", "balanced"] = "balanced",
) -> List[int]:
    """Return a superordinate mapping for EMNIST classes

    Args:
        split (str): Split type, either "by_class" or "balanced".
        map_type (str): Mapping type, either "alphanumerical" or "alphanumerical_rev".

    Returns:
        List[int]: Superordinate mapping for EMNIST classes.
    """

    sord = []
    if map_type == "alphanumerical":
        if split == "by_class":
            sord = [0 for i in range(10)] + [1 for i in range(26 * 2)]
        if split == "balanced":
            sord = [0 for i in range(10)] + [1 for i in range(37)]
    if map_type == "alphanumerical_rev":
        if split == "by_class":
            sord = [1 for i in range(10)] + [0 for i in range(26 * 2)]
        if split == "balanced":
            sord = [1 for i in range(10)] + [0 for i in range(37)]

    return sord


def load_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        m = pickle.load(f)
    return m


def json_to_csv(json_path, csv_path):
    """
    Convert a JSON file to a CSV file.

    Args:
        json_path (str): Path to the JSON file.
        csv_path (str): Path to save the CSV file.
    """
    data = load_json(json_path)
    with open(csv_path, "w") as f:
        f.write(",".join(data.keys()) + "\n")
        for i in range(len(data["batch_idx"])):
            f.write(",".join([str(data[k][i]) for k in data.keys()]) + "\n")


def load_cfg(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return to_attr_dict(cfg)


def load_json_config(cfg_path: str) -> DefaultConfig:
    """
    Load a JSON configuration file and return a DefaultConfig object.

    Args:
        cfg_path (str): Path to the JSON configuration file.
    """
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)
    # instatiate DefaultConfig object
    cfg = DefaultConfig()
    # assign values to the object while taking care of nesting
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                setattr(getattr(cfg, k), k1, v1)
        else:
            setattr(cfg, k, v)
    return cfg


def make_mosaic(
    imgs: np.ndarray,
    nrow: int,
    ncol: int,
    padding: int = 2,
    pad_value: int = 0,
    col_major: bool = False,
) -> np.ndarray:
    """Create a mosaic with images.

    Args:
        imgs (np.ndarray): Images to create the mosaic.
        nrow (int): Number of rows in the mosaic.
        ncol (int): Number of columns in the mosaic.
        padding (int): Padding between images.
        pad_value (int): Padding value.

    Returns:
        np.ndarray: Mosaic of images.
    """
    ndims = imgs.ndim
    nimgs = imgs.shape[0]
    # create black images if there are not enough images to cover the mosaic
    if nimgs < nrow * ncol:
        imgs = np.concatenate(
            [imgs, np.full((nrow * ncol - nimgs, *imgs.shape[1:]), pad_value)]
        )
    imshape = imgs.shape[1:3]
    mshape = (
        nrow * imshape[0] + (nrow - 1) * padding,
        ncol * imshape[1] + (ncol - 1) * padding,
    )
    if ndims == 4:
        mshape = mshape + (3,)

    mosaic = np.full(
        mshape,
        pad_value,
        dtype=np.uint8,
    )

    for i in range(nimgs):
        if col_major:
            r = i % nrow
            c = i // nrow
        else:
            r = i // ncol
            c = i % ncol
        mosaic[
            r * imshape[0] + r * padding : (r + 1) * imshape[0] + r * padding,
            c * imshape[1] + c * padding : (c + 1) * imshape[1] + c * padding,
        ] = imgs[i]
    return mosaic


def get_infer_set(dataloader: torch.utils.data.DataLoader):
    """Get a set of images for inference

    Arguments:
        dataloader {torch.utils.data.DataLoader} -- the data loader to get the images from

    Returns:
        List[torch.Tensor] -- a list of images, labels, and contexts
    """
    # determine the target classes based on dimensionality of the labels
    # TODO: This only works with c-EMNIST for now, sorry.
    _, label_batch, _ = next(iter(dataloader))
    target_classes = torch.arange(label_batch.shape[1])
    # now find the indices of the target classes in the test loader
    # initialize empty lists
    imgs = []
    contexts = []
    labels = []
    for tc in target_classes:
        for img_batch, label_batch, context_batch in dataloader:
            # find the first occurrence of the target class in label_batch
            idx = torch.where(label_batch.argmax(dim=1) == tc)
            if len(idx[0]) > 0:
                imgs.append(img_batch[idx[0][0]])
                contexts.append(context_batch[idx[0][0]])
                labels.append(label_batch[idx[0][0]])
                break
    return [torch.stack(imgs), torch.stack(labels), torch.stack(contexts)]


def postprocess_prediction(
    imgs: torch.Tensor, mean: float = 0.1307, std: float = 0.3081
) -> np.ndarray:
    """Postprocess img predictions from a cEMNIST model to be in the correct format."""
    ims = imgs.cpu().numpy()
    ims = ims * std + mean
    # clip to [0, 1]
    ims = np.clip(ims, 0, 1)
    # go to uint8 and rotate
    ims = (ims * 255).reshape(-1, 28, 28).astype(np.uint8).transpose(0, 2, 1)
    return ims


def save_img(img: np.ndarray, path: str, size: Optional[Tuple] = None):
    """Save an image to a file.

    Args:
        img (np.ndarray): Image to save.
        path (str): Path to save the image.
    """
    i = Image.fromarray(img)
    if size is not None:
        i = i.resize(size)
    i.save(path)


def get_model_files(folder: str) -> Tuple[List[str], List[int]]:
    """Return model files per epoch"""
    model_files = [f for f in os.listdir(folder) if f.startswith("model_")]
    assert len(model_files) > 0, "No model files found in folder"
    epochs = [int(f.split("_")[1].split(".")[0]) for f in model_files]
    # sort both
    epochs, model_files = zip(*sorted(zip(epochs, model_files)))
    return model_files, epochs
