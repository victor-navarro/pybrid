"""Datasets module"""

import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import datasets, transforms
from pybrid import utils
from pybrid.config import DefaultConfig


class MNIST(datasets.MNIST):
    def __init__(
        self,
        train,
        path="./data",
        size=None,
        scale=None,
        normalize=False,
        noisify=False,
        labels=None,
        shrink_classes=False,
        noise_mean=0,
        noise_std=1,
    ):
        transform = _get_transform(
            normalize=normalize,
            mean=(0.1307),
            std=(0.3081),
            noisify=noisify,
            noise_mean=noise_mean,
            noise_std=noise_std,
        )
        super().__init__(path, download=False, transform=transform, train=train)
        self.scale = scale
        if size is not None:
            self._reduce(size)
        if labels is not None:
            self._split(labels, shrink_classes)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        target = _one_hot(target)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]

    def _split(self, labels, shrink_classes):
        idxs = torch.empty(0).long()
        for label in labels:
            idxs = torch.cat((idxs, (self.targets == label).nonzero().squeeze()))
        self.data = self.data[idxs]
        self.targets = self.targets[idxs]
        if shrink_classes:
            self.n_classes = len(labels)
            # now remap the old targets to the remaining targets, in order of appearance
            new_targets = torch.zeros_like(self.targets)
            for i, label in enumerate(labels):
                new_targets[self.targets == label] = i
            self.targets = new_targets


class cMNIST(datasets.MNIST):
    def __init__(
        self,
        train,
        path="./data",
        size=None,
        scale=None,
        normalize=False,
        sordmap=None,
        split_context=False,
        noisify=False,
        labels=None,
        shrink_classes=False,
        noise_mean=0,
        noise_std=1,
    ):
        transform = _get_transform(
            normalize=normalize,
            mean=(0.1307),
            std=(0.3081),
            noisify=noisify,
            noise_mean=noise_mean,
            noise_std=noise_std,
        )

        download_mnist(path, "cMNIST")
        super().__init__(path, download=False, transform=transform, train=train)
        self.scale = scale
        self.n_classes = len(self.classes)
        self.sordmap = sordmap
        self.split_context = split_context
        if size is not None:
            self._reduce(size)
        if labels is not None:
            self._split(labels, shrink_classes)
        if self.sordmap is not None:
            self.n_sord_classes = len(set(self.sordmap))

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        context = None
        if self.sordmap is not None and self.split_context:
            context = self.sordmap[target]

        if (self.sordmap is not None) and (not self.split_context):
            sord = _one_hot(self.sordmap[target], self.n_sord_classes)
            # append superordinate one-hot label
            data = torch.cat((data, _to_vector(sord)))

        target = _one_hot(target, self.n_classes)
        if self.scale is not None:
            target = _scale(target, self.scale)
        if context is None:
            return data, target
        return data, target, context

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]

    def _split(self, labels, shrink_classes):
        idxs = torch.empty(0).long()
        for label in labels:
            idxs = torch.cat((idxs, (self.targets == label).nonzero().squeeze()))
        self.data = self.data[idxs]
        self.targets = self.targets[idxs]
        if shrink_classes:
            self.n_classes = len(labels)
            if self.sordmap is not None:
                # remake the sordmap with the remaining targets
                self.sordmap = [self.sordmap[label] for label in labels]
            # now remap the old targets to the remaining targets, in order of appearance
            new_targets = torch.zeros_like(self.targets)
            for i, label in enumerate(labels):
                new_targets[self.targets == label] = i
            self.targets = new_targets


class cEMNIST(datasets.EMNIST):
    def __init__(
        self,
        train,
        path="./data/EMNIST",
        size=None,
        scale=None,
        normalize=False,
        noisify=False,
        labels=None,
        split="byclass",
        sordmap=None,
        shrink_classes=False,
        split_context=False,
        noise_mean=0,
        noise_std=1,
    ):
        transform = _get_transform(
            normalize=normalize,
            mean=(0.1307),
            std=(0.3081),
            noisify=noisify,
            noise_mean=noise_mean,
            noise_std=noise_std,
        )
        # run download manually; path is broken in nist.gov
        download_emnist(path, "cEMNIST")
        super().__init__(
            path,
            download=False,
            transform=transform,
            train=train,
            split=split,
        )
        self.scale = scale
        self.n_classes = len(self.classes)
        self.sordmap = sordmap
        self.split_context = split_context
        if size is not None:
            self._reduce(size)
        if labels is not None:
            self._split(labels, shrink_classes)
        if self.sordmap is not None:
            self.n_sord_classes = len(set(self.sordmap))

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        context = None
        if self.sordmap is not None and self.split_context:
            context = self.sordmap[target]

        if (self.sordmap is not None) and (not self.split_context):
            sord = _one_hot(self.sordmap[target], self.n_sord_classes)
            # append superordinate one-hot label
            data = torch.cat((data, _to_vector(sord)))

        target = _one_hot(target, self.n_classes)
        if self.scale is not None:
            target = _scale(target, self.scale)
        if context is None:
            return data, target
        return data, target, context

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]

    def _split(self, labels, shrink_classes):
        idxs = torch.empty(0).long()
        for label in labels:
            idxs = torch.cat((idxs, (self.targets == label).nonzero().squeeze()))
        self.data = self.data[idxs]
        self.targets = self.targets[idxs]
        if shrink_classes:
            self.n_classes = len(labels)
            if self.sordmap is not None:
                # remake the sordmap with the remaining targets
                self.sordmap = [self.sordmap[label] for label in labels]
            # now remap the old targets to the remaining targets, in order of appearance
            new_targets = torch.zeros_like(self.targets)
            for i, label in enumerate(labels):
                new_targets[self.targets == label] = i
            self.targets = new_targets


class SVHN(datasets.SVHN):
    def __init__(self, train, path="./data", size=None, scale=None, normalize=False):
        if normalize:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(path, download=True, transform=transform, train=train)
        self.scale = scale
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        target = _one_hot(target)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


class CIFAR10(datasets.CIFAR10):
    def __init__(self, train, path="./data", size=None, scale=None, normalize=False):
        if normalize:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(path, download=True, transform=transform, train=train)
        self.scale = scale
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        target = _one_hot(target)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


class CIFAR100(datasets.CIFAR100):
    def __init__(self, train, path="./data", size=None, scale=None, normalize=False):
        if normalize:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(path, download=True, transform=transform, train=train)
        self.scale = scale
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        target = _one_hot(target, n_classes=100)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, train, path="./data", size=None, normalize=False):
        transform = _get_transform(normalize=normalize, mean=(0.5), std=(0.5))
        super().__init__(path, download=True, transform=transform, train=train)
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        target = _one_hot(target)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


def download_emnist(data_dir: str = "./data/", subdir: str = "EMNIST"):
    utils.run_emnist_dl(data_dir, subdir)


def get_dataloader(
    dataset, batch_size, shuffle=True, **kwargs
) -> List[data.DataLoader]:
    """Get a dataloader for a dataset."""
    dataloader = data.DataLoader(
        dataset, batch_size, shuffle=shuffle, drop_last=True, **kwargs
    )
    return list(map(_preprocess_batch, dataloader))
    # return dataloader


def accuracy(pred_labels, true_labels):
    batch_size = pred_labels.size(0)
    correct = 0
    for b in range(batch_size):
        if torch.argmax(pred_labels[b, :]) == torch.argmax(true_labels[b, :]):
            correct += 1
    return correct / batch_size


def plot_imgs(img_preds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plot_side = int(np.ceil(np.sqrt(img_preds.shape[0])))
    imgs = img_preds.cpu().detach().numpy()
    imgs = [np.reshape(imgs[i, 0:784], [28, 28]).T for i in range(imgs.shape[0])]
    _, axes = plt.subplots(plot_side, plot_side)
    axes = axes.flatten()
    for i, img in enumerate(imgs):
        axes[i].imshow(img, cmap="gray")
    for h in range(i + 1, plot_side**2):
        axes[h].imshow(np.zeros_like(img), cmap="gray")
    plt.savefig(path)
    plt.close("all")


def _preprocess_batch(batch):
    batch[0] = utils.set_tensor(batch[0])
    batch[1] = utils.set_tensor(batch[1])
    if len(batch) == 3:
        batch[2] = utils.set_tensor(batch[2])
        return (batch[0], batch[1], batch[2])
    return (batch[0], batch[1])


def _get_transform(
    normalize=True, mean=(0.5), std=(0.5), noisify=False, noise_mean=1, noise_std=0.1
):
    transform = [transforms.ToTensor()]
    if normalize:
        transform = transform + [transforms.Normalize(mean=mean, std=std)]
    if noisify:
        transform = transform + [
            transforms.Lambda(
                lambda x: x + torch.randn_like(x) * noise_std + noise_mean
            )
        ]
    return transforms.Compose(transform)


def _one_hot(labels, n_classes=10):
    arr = torch.eye(n_classes)
    return arr[labels]


def _scale(targets, factor):
    return targets * factor + 0.5 * (1 - factor) * torch.ones_like(targets)


def _to_vector(batch):
    batch_size = batch.size(0)
    return batch.reshape(batch_size, -1).squeeze()


def get_dataset(cfg: DefaultConfig):
    """Returns a training and testing dataset based on the configuration."""
    train_dataset, test_dataset = None, None
    # back compatibility with data_dir
    if "data_dir" not in cfg.data.__dict__.keys():
        cfg.data.data_dir = "./data/"
    if cfg.data.dataset == "cEMNIST":
        train_dataset = cEMNIST(
            train=True,
            scale=cfg.data.label_scale,
            size=cfg.data.train_size,
            normalize=cfg.data.normalize,
            sordmap=cfg.data.train_sord,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )
        test_dataset = cEMNIST(
            train=False,
            scale=cfg.data.label_scale,
            size=cfg.data.test_size,
            normalize=cfg.data.normalize,
            sordmap=cfg.data.test_sord,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )

    if cfg.data.dataset == "cEMNIST_balanced":
        train_dataset = cEMNIST(
            train=True,
            scale=cfg.data.label_scale,
            size=cfg.data.train_size,
            normalize=cfg.data.normalize,
            split="balanced",
            sordmap=cfg.data.train_sord,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )
        test_dataset = cEMNIST(
            train=False,
            scale=cfg.data.label_scale,
            size=cfg.data.test_size,
            normalize=cfg.data.normalize,
            split="balanced",
            sordmap=cfg.data.test_sord,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )

    if cfg.data.dataset == "c-EMNIST_balanced":
        train_dataset = cEMNIST(
            train=True,
            scale=cfg.data.label_scale,
            size=cfg.data.train_size,
            normalize=cfg.data.normalize,
            split="balanced",
            sordmap=cfg.data.train_sord,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            split_context=True,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )
        test_dataset = cEMNIST(
            train=False,
            scale=cfg.data.label_scale,
            size=cfg.data.test_size,
            normalize=cfg.data.normalize,
            split="balanced",
            sordmap=cfg.data.test_sord,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            split_context=True,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )

    if cfg.data.dataset == "EMNIST_balanced":
        train_dataset = cEMNIST(
            train=True,
            scale=cfg.data.label_scale,
            size=cfg.data.train_size,
            normalize=cfg.data.normalize,
            split="balanced",
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )
        test_dataset = cEMNIST(
            train=False,
            scale=cfg.data.label_scale,
            size=cfg.data.test_size,
            normalize=cfg.data.normalize,
            split="balanced",
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )
    if cfg.data.dataset == "EMNIST_c-MNIST":
        train_dataset = cEMNIST(
            train=True,
            scale=cfg.data.label_scale,
            size=cfg.data.train_size,
            normalize=cfg.data.normalize,
            split="mnist",
            sordmap=cfg.data.train_sord,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            split_context=True,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )
        test_dataset = cEMNIST(
            train=False,
            scale=cfg.data.label_scale,
            size=cfg.data.train_size,
            normalize=cfg.data.normalize,
            split="mnist",
            sordmap=cfg.data.train_sord,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            split_context=True,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )

    if cfg.data.dataset == "c-MNIST":
        train_dataset = cMNIST(
            train=True,
            scale=cfg.data.label_scale,
            size=cfg.data.train_size,
            normalize=cfg.data.normalize,
            sordmap=cfg.data.train_sord,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            split_context=True,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )
        test_dataset = cMNIST(
            train=False,
            scale=cfg.data.label_scale,
            size=cfg.data.test_size,
            normalize=cfg.data.normalize,
            sordmap=cfg.data.train_sord,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            split_context=True,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )

    if cfg.data.dataset == "MNIST":
        train_dataset = MNIST(
            train=True,
            scale=cfg.data.label_scale,
            size=cfg.data.train_size,
            normalize=cfg.data.normalize,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )
        test_dataset = MNIST(
            train=False,
            scale=cfg.data.label_scale,
            size=cfg.data.test_size,
            normalize=cfg.data.normalize,
            labels=cfg.data.dataset_classes,
            shrink_classes=cfg.data.shrink_classes,
            noisify=cfg.data.noisify,
            noise_mean=cfg.data.noise_mean,
            noise_std=cfg.data.noise_std,
            path=cfg.data.data_dir,
        )

    return train_dataset, test_dataset


def download_mnist(data_dir: str = "data", subdir: str = "cMNIST"):
    new_path = os.path.join(data_dir, subdir)
    if not os.path.exists(new_path):
        utils.run_mnist_dl(data_dir, subdir)
