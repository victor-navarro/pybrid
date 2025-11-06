# %%
""" Measure weight sparsity of a model, across epochs"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pybrid.utils import load_pkl


# %%
def get_abs_std(mod, layer: int):
    return mod.layers[layer].weights.abs().std().item()


def get_std(mod, layer: int):
    return mod.layers[layer].weights.std().item()


def get_l2norm(mod, layer: int):
    return mod.layers[layer].weights.norm().item()


def get_l1norm(mod, layer: int):
    return mod.layers[layer].weights.norm(p=1).item()


def get_mean(mod, layer: int):
    return mod.layers[layer].weights.mean().item()


def get_abs_mean(mod, layer: int):
    return mod.layers[layer].weights.abs().mean().item()


def get_quantile(mod, layer: int, q: float = 0.5):
    return mod.layers[layer].weights.abs().quantile(q).item()


def get_nthreshold(mod, layer: int, threshold: float = 1e-2):
    return (mod.layers[layer].weights.abs() < threshold).sum().item() / mod.layers[
        layer
    ].weights.numel()


def get_l1l2ratio(mod, layer: int, normalize: bool = True):
    ratio = (
        mod.layers[layer].weights.norm(p=1).item()
        / mod.layers[layer].weights.norm().item()
    )
    if normalize:
        return (ratio - 1) / ((mod.layers[layer].weights.numel() ** 0.5) - 1)
    return ratio


metrics = {
    "abs_std": get_abs_std,
    "std": get_std,
    "l2norm": get_l2norm,
    "l1norm": get_l1norm,
    "mean": get_mean,
    "abs_mean": get_abs_mean,
    "median": get_quantile,
    "nthreshold(1e-2)": get_nthreshold,
    "l1l2ratio": get_l1l2ratio,
}


# %%

history = [
    "../../results/exp_1_norm/progenitor/0",
    "../../results/exp_1_norm/normal_twin/0",
    "../../results/exp_2_norm/normal_twin/49/0",
]
LAYERS = range(3)

results = pd.DataFrame(columns=["bi", "epoch", "layer", "value", "measure"])
for bi, base_dir in enumerate(history):
    models = os.listdir(base_dir)
    models = [m for m in models if m.startswith("model") and m.endswith(".pkl")]
    for model in models:
        m = load_pkl(os.path.join(base_dir, model))
        for layer in LAYERS:
            epoch = int(model.split("_")[1].split(".")[0])
            for measure, func in metrics.items():
                value = func(m, layer)
                results.loc[results.shape[0] + 1] = pd.Series(
                    {
                        "bi": bi,
                        "epoch": epoch,
                        "layer": layer,
                        "value": value,
                        "measure": measure,
                    }
                )


# %%
results = results.sort_values(by=["bi", "epoch", "layer", "measure"])
results["epoch"][results["bi"] == 1] += 100
results["epoch"][results["bi"] == 2] += 150
results

# %%
# plot the results
umeasures = results["measure"].unique()
nmeasures = len(results["measure"].unique())
side = int(np.ceil(nmeasures**0.5))
fig, axes = plt.subplots(side, side, layout="constrained")
for i, ax in enumerate(axes.flatten()):
    if i < nmeasures:
        measure = umeasures[i]
        ax.axvline(100, color="gray", linestyle="--")
        ax.axvline(150, color="gray", linestyle="--")
        for layer in results["layer"].unique():
            data = results[
                (results["measure"] == measure) & (results["layer"] == layer)
            ]
            ax.plot(data["epoch"], data["value"], label=f"layer {layer}")
        ax.set_title(measure)
    else:
        ax.axis("off")
axes.flatten()[side * 2 - 1].legend(
    loc="right", bbox_to_anchor=(2.1, 0.5), title="Layer"
)
plt.show()
fig.set_size_inches(8, 4)
fig.savefig(
    "../../results/exp_2_norm_tests/network_sparsity.png", bbox_inches="tight", dpi=300
)

# %%
