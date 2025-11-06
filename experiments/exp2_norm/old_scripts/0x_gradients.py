# %%
""" Check the gradient of each layer across inference"""
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
    gradients = []
    for n in range(model.num_layers):
        gradients.append(
            np.zeros(
                (img_batch.size(0), num_iters),
            )
        )

    itr = 0
    for itr in range(num_iters):
        delta = model.layers[0].backward(model.errs[1])
        model.mus[0] = model.mus[0] + model.mu_dt * (2 * delta)
        gradients[0][:, itr] = delta.abs().mean().cpu().numpy()
        for l in range(1, model.num_layers):
            delta = model.layers[l].backward(model.errs[l + 1]) - model.errs[l]
            model.deltas[l] = delta
            model.mus[l] = model.mus[l] + model.mu_dt * (2 * delta)
            gradients[l][:, itr] = delta.abs().mean().cpu().numpy()

        for n in range(1, model.num_nodes):
            if not fixed_preds:
                model.preds[n] = model.layers[n - 1].forward(model.mus[n - 1])
            model.errs[n] = model.mus[n] - model.preds[n]

    return gradients


# %%
# define function to extract layerwise gradients across inference steps
def get_gradients(model, loader, test_iters=100):
    # loop through batches
    all_gradients = []
    for imgs, _, contexts in loader:
        # get gradients
        grads = test_model(imgs, contexts, model, num_iters=test_iters)
        # append to list on a layer by layer basis
        if len(all_gradients) > 0:
            for layer in range(model.num_layers):
                all_gradients[layer] = np.concatenate(
                    [all_gradients[layer], grads[layer]]
                )
        else:
            all_gradients = grads
    return all_gradients


# %%

# initialize a dataframe
df = pd.DataFrame()
for epoch in EPOCHS:
    print(epoch)
    # load model
    model = utils.load_pkl(os.path.join(BASE_DIR, f"model_{epoch}.pkl"))
    # get gradients
    grads = get_gradients(model, test_loader, test_iters=TEST_ITERS)

    # average absolute gradients across images and nodes
    avgs = [np.abs(grads[l]).mean(axis=(0)) for l in range(model.num_layers)]
    for l, avg in enumerate(avgs):
        # create layer dataframe
        layer_df = pd.DataFrame(avg, columns=["gradient"])
        layer_df["layer"] = l
        layer_df["epoch"] = epoch
        layer_df["iteration"] = list(range(TEST_ITERS))
        # append to main dataframe
        df = pd.concat([df, layer_df])


# %%
# plot the results
fig, axes = plt.subplots(1, model.num_layers, figsize=(8, 2), layout="constrained")
# group by epoch and layer
grouped = df.groupby(["layer"])

for (layer), data in grouped:
    ax = axes[layer]
    for epoch in EPOCHS:
        d = df[(df["epoch"] == epoch) & (df["layer"] == layer)]
        ax.plot(d["iteration"], d["gradient"], label=f"epoch {epoch}")
        ax.set_title(f"Layer {layer[0]} -> {layer[0]+1}")
        ax.set_ylabel("Abs. Gradient")
        ax.set_xlabel("Iteration")
    # ax.set_ylim(0, None)

# add legend
axes[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
# Save the plot
fig.savefig(
    "../../results/exp_2_norm_tests/gradient_normal_twin_0_49_inf.png",
    bbox_inches="tight",
    dpi=DPI,
)

# %%
