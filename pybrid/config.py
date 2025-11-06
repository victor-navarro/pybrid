"""Defines a default configuration dict for experiments"""

import copy
from typing import Literal, Optional, List
import torch

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DefaultExp:
    """A class for the default experiment configuration"""

    def __init__(self):

        self.log_dir: str = "results/test"
        self.img_dir: str = "results/test/imgs"
        self.seed: int = 0
        self.num_epochs: int = 20
        self.test_hybrid: bool = False
        self.test_pc: bool = False
        self.test_amort: bool = False
        self.gen_label_images: bool = True
        self.gen_infer_images: bool = True
        self.test_every: int = 1
        self.save_models: bool = True


class DefaultData:
    """A class for the default data configuration"""

    def __init__(self):
        self.dataset: str = "c-EMNIST_balanced"
        self.train_size: Optional[int] = None
        self.test_size: Optional[int] = None
        self.label_scale: float = 0.94
        self.normalize: bool = True
        self.train_sord: Optional[List[int]] = None
        self.test_sord: Optional[List[int]] = None
        self.trap_sord: Optional[List[int]] = None
        self.dataset_classes: Optional[List[int]] = None
        self.shrink_classes: bool = False
        self.noisify: bool = False
        self.noise_mean: float = 0.0
        self.noise_std: float = 1.0
        self.data_dir: str = "./data"


class DefaultInfer:
    """A class for the default inference configuration"""

    def __init__(self):
        self.mu_dt: float = 0.01
        self.num_train_iters: int = 100
        self.num_test_iters: int = 500
        self.fixed_preds_train: bool = False
        self.fixed_preds_test: bool = False
        self.train_thresh: Optional[float] = None
        self.test_thresh: Optional[float] = None
        self.init_std: float = 0.01
        self.no_backward: bool = False
        self.delta_thresh: Optional[float] = None


class DefaultModel:
    """A class for the default model configuration"""

    def __init__(self):
        self.nodes: List[int] = [62, 500, 500, 784]
        self.amort_nodes: List[int] = [784, 500, 500, 62]
        self.train_amort: bool = True
        self.use_bias: bool = True
        self.kaiming_init: bool = False
        self.act_fn: Literal["tanh"] = "tanh"
        self.device: str = _DEVICE
        self.model_class: Literal["DoubleAmortModel"] = "DoubleAmortModel"
        self.n_amort_nets: int = 2
        self.model_pkl: Optional[str] = None
        self.supervised: bool = True
        self.freeze_top: bool = True


class DefaultOptim:
    """A class for the default optimization configuration"""

    def __init__(self):
        self.name: Literal["Adam"] = "Adam"
        self.lr: float = 1e-4
        self.amort_lr: float = 1e-4
        self.batch_size: int = 64
        self.test_batch_size: Optional[int] = None
        self.batch_scale: bool = True
        self.grad_clip: int = 50
        self.weight_decay: Optional[float] = None
        self.normalize_weights: bool = False


# Defines a class for the default configuration


class DefaultConfig:
    """A class for the default configuration"""

    def __init__(self):
        self.exp: DefaultExp = DefaultExp()
        self.data: DefaultData = DefaultData()
        self.infer: DefaultInfer = DefaultInfer()
        self.model: DefaultModel = DefaultModel()
        self.optim: DefaultOptim = DefaultOptim()

    def copy(self):
        """
        Returns a deep copy of the configuration
        """
        return copy.deepcopy(self)

    def to_dict(self):
        """
        Returns a dictionary representation of the configuration
        """
        return {
            "exp": self.exp.__dict__,
            "data": self.data.__dict__,
            "infer": self.infer.__dict__,
            "model": self.model.__dict__,
            "optim": self.optim.__dict__,
        }

    def __repr__(self):
        return str(self.to_dict())
