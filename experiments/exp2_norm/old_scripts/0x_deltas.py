# %%
""" Check the delta of each layer across inference"""
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from pybrid.datasets import get_dataset, get_dataloader
from pybrid import utils
from pybrid.models import DoubleAmortModel

# %%
TEST_BATCHES = 4
TEST_BATCH_SIZE = 512
TEST_ITERS = 200
DPI = 600
BASE_DIR = "../../results/exp_2_norm/normal_twin/0/0"
# Epochs to compare
EPOCHS = [0, 9, 49]

# %%
# load model config
cfg = utils.load_json_config(os.path.join(BASE_DIR, "config.json"))
cfg.data.data_dir = "../../data"
# get dataset
ds = get_dataset(cfg)
# get class labels
class_labels = ds[1].classes
class_labels = [class_labels[c] for c in cfg.data.dataset_classes]
# get loaders
train_loader = get_dataloader(ds[0], batch_size=TEST_BATCH_SIZE)[:TEST_BATCHES]
test_loader = get_dataloader(ds[1], batch_size=TEST_BATCH_SIZE)[:TEST_BATCHES]


# %%
def test_model(
    img_batch: torch.Tensor,
    context_batch: torch.Tensor,
    model: DoubleAmortModel,
    num_iters: int = 100,
    init_std: float = 0.05,
    fixed_preds: bool = False,
    use_amort: bool = False,
):
    """A hack"""

    # reset model
    model.reset()
    if use_amort:
        model.set_img_batch_amort(img_batch)
        model.forward_mu(context_batch)
    else:
        model.reset_mu(img_batch.size(0), init_std)
    model.set_img_batch(img_batch)

    for n in range(1, model.num_nodes):
        model.preds[n] = model.layers[n - 1].forward(model.mus[n - 1])
        model.errs[n] = model.mus[n] - model.preds[n]

    # preallocate tensors for all layers
    layer_activations = []
    for n in range(model.num_nodes):
        layer_activations.append(
            np.zeros(
                (img_batch.size(0), num_iters, model.nodes[n]),
            )
        )

    itr = 0
    for itr in range(num_iters):
        delta = model.layers[0].backward(model.errs[1])
        model.mus[0] = model.mus[0] + model.mu_dt * (2 * delta)
        for l in range(1, model.num_layers):
            delta = model.layers[l].backward(model.errs[l + 1]) - model.errs[l]
            model.deltas[l] = delta
            model.mus[l] = model.mus[l] + model.mu_dt * (2 * delta)

        for n in range(1, model.num_nodes):
            if not fixed_preds:
                model.preds[n] = model.layers[n - 1].forward(model.mus[n - 1])
            model.errs[n] = model.mus[n] - model.preds[n]

        # save layer activations
        for n in range(model.num_nodes):
            # note we get the prediction (not the state) for the last layer
            if n == model.num_layers:
                layer_activations[n][:, itr, :] = model.preds[n].cpu().numpy()
            else:
                layer_activations[n][:, itr, :] = model.mus[n].cpu().numpy()

    return layer_activations


# %%
# define function to extract layerwise deltas across inference steps
def get_deltas(model, loader, test_iters=100):
    # loop through batches
    all_deltas = []
    for imgs, _, contexts in loader:
        # get activations
        acts = test_model(imgs, contexts, model, num_iters=test_iters)
        # calculate deltas
        deltas = [np.diff(acts[layer], axis=1) for layer in range(model.num_nodes)]
        # append to list on a layer by layer basis
        if len(all_deltas) > 0:
            for layer in range(model.num_nodes):
                all_deltas[layer] = np.concatenate([all_deltas[layer], deltas[layer]])
        else:
            all_deltas = deltas
    return all_deltas


# %%

# initialize a dataframe
df = pd.DataFrame()
for epoch in EPOCHS:
    print(epoch)
    # load model
    model = utils.load_pkl(os.path.join(BASE_DIR, f"model_{epoch}.pkl"))
    # get deltas
    ds = get_deltas(model, test_loader, test_iters=TEST_ITERS)
    # average absolute deltas across images and nodes
    avgs = [np.abs(ds[l]).mean(axis=(0, 2)) for l in range(model.num_nodes)]
    for l, avg in enumerate(avgs):
        # create layer dataframe
        layer_df = pd.DataFrame(avg, columns=["delta"])
        layer_df["layer"] = l
        layer_df["epoch"] = epoch
        layer_df["iteration"] = list(range(1, TEST_ITERS))
        # append to main dataframe
        df = pd.concat([df, layer_df])


# %%
# plot the results
fig, axes = plt.subplots(1, model.num_nodes, figsize=(8, 2), layout="constrained")
# group by epoch and layer
grouped = df.groupby(["layer"])

for (layer), data in grouped:
    ax = axes[layer]
    for epoch in EPOCHS:
        d = df[(df["epoch"] == epoch) & (df["layer"] == layer)]
        ax.plot(d["iteration"], d["delta"], label=f"epoch {epoch}")
        ax.set_title(f"Layer {layer[0]}")
        ax.set_ylabel("Delta")
        ax.set_xlabel("Iteration")
    # ax.set_ylim(0, None)

# add legend
axes[3].legend(loc="center left", bbox_to_anchor=(1, 0.5))
# Save the plot
fig.savefig(
    "../../results/exp_2_norm_tests/delta_normal_twin_0_49_inf.png",
    bbox_inches="tight",
    dpi=DPI,
)

# %%
