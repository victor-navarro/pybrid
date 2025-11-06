# %%
""" Check if more iterations do the trick"""

import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pybrid.datasets import get_dataset, get_dataloader
from pybrid import utils
from pybrid.tests import test_model
from pybrid.postprocessing.label_accuracy import _agg_acts
from pybrid.postprocessing.reconstruction_error import _agg_errs


# %%
TEST_BATCHES = 4
TEST_BATCH_SIZE = 512
TEST_ITERS = 1000
DPI = 600

BASE_DIR = "../../results/exp_2_norm/normal_twin/49/0"

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


# %% define function to test the model (based on get label accuracy)


def get_acc(model, loader, class_labs, test_iters=100):
    full_df = pd.DataFrame()
    # loop through batches
    for imgs, labels, contexts in loader:
        if cfg.model.train_amort:
            # model has amortised inference, so need to test hybrid and amort components
            acts = test_model(
                imgs, contexts, model, layer=0, num_iters=test_iters, use_amort=True
            )
            # score activations using utility function
            hybrid_accs = _agg_acts(acts, labels)
            hybrid_accs["network"] = "hybrid"
            # add to full dataframe
            full_df = pd.concat([full_df, hybrid_accs])

            # test amortization
            acts = test_model(
                imgs,
                contexts,
                model,
                layer=0,
                num_iters=1,
                use_amort=True,
                use_infer=False,
            )

            amort_accs = _agg_acts(acts, labels)
            amort_accs["network"] = "amort"
            full_df = pd.concat([full_df, amort_accs])

        # test pc component
        acts = test_model(imgs, contexts, model, layer=0, num_iters=test_iters)
        pc_accs = _agg_acts(acts, labels)
        pc_accs["network"] = "pc"
        full_df = pd.concat([full_df, pc_accs])

    # get sums
    full_df = full_df.groupby(["class", "iteration", "network"]).sum().reset_index()
    # calculate accuracy
    full_df["accuracy"] = full_df["correct"] / full_df["trials"]
    # relabel class
    full_df["class"] = full_df["class"].apply(lambda x: class_labs[x])
    return full_df


def get_rec(model, loader, class_labs, test_iters=100):
    full_df = pd.DataFrame()
    # loop through batches
    # loop through batches
    for imgs, labels, contexts in loader:
        if cfg.model.train_amort:
            # model has amortised inference, so need to test hybrid and amort components
            acts = test_model(
                imgs, contexts, model, layer=0, num_iters=test_iters, use_amort=True
            )
            acts = torch.tensor(acts).to(model.device)
            hybrid_preds = model.backward(acts)
            # score activations using utility function
            hybrid_rec = _agg_errs(torch.abs(imgs.unsqueeze(1) - hybrid_preds), labels)
            hybrid_rec["network"] = "hybrid"
            # add to full dataframe
            full_df = pd.concat([full_df, hybrid_rec])

            # test amortization
            acts = test_model(
                imgs,
                contexts,
                model,
                layer=0,
                num_iters=1,
                use_amort=True,
                use_infer=False,
            )
            acts = torch.tensor(acts).to(model.device)
            # do a backward pass to get image
            amort_preds = model.backward(acts)
            amort_rec = _agg_errs(torch.abs(imgs.unsqueeze(1) - amort_preds), labels)
            amort_rec["network"] = "amort"
            full_df = pd.concat([full_df, amort_rec])

        # test pc
        acts = test_model(imgs, contexts, model, layer=0, num_iters=test_iters)
        acts = torch.tensor(acts).to(model.device)
        pc_preds = model.backward(acts)
        pc_rec = _agg_errs(torch.abs(imgs.unsqueeze(1) - pc_preds), labels)
        pc_rec["network"] = "pc"
        full_df = pd.concat([full_df, pc_rec])

    # get sums
    full_df = full_df.groupby(["class", "iteration", "network"]).sum().reset_index()
    # calculate average error per image
    full_df["rec_error"] = full_df["total_error"] / full_df["trials"]
    # relabel class
    full_df["class"] = full_df["class"].apply(lambda x: class_labs[x])
    return full_df


# %%
m = utils.load_pkl(os.path.join(BASE_DIR, "model_49.pkl"))
# test on test set
acc_df = get_acc(m, test_loader, class_labels, test_iters=TEST_ITERS)
# get avg accuracy  and add it to the dataframe
avg_acc = acc_df.groupby(["network", "iteration"])["accuracy"].mean().reset_index()
avg_acc["class"] = "AVG"
acc_df = pd.concat([acc_df, avg_acc])
# save to csv
acc_df.to_csv("../../results/exp_2_norm_tests/normal_twin_49_acc_1000.csv", index=False)

# %%
max_iters = acc_df["iteration"].max()
# just plot the data for the AVG class
plot_dat = acc_df[acc_df["class"] == "AVG"].set_index("network")
# plot only the data for the hybrid and pc networks
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

for i, net in enumerate(["hybrid", "pc"]):
    data = plot_dat.loc[net]
    ax.plot(
        data["iteration"],
        data["accuracy"],
        label=net,
    )
ax.set_ylim(0.0, 1.0)
ax.set_xlim(-1, max_iters)
ax.set_xlabel("Inference Step")
ax.set_ylabel("Classification Accuracy")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# Save the plot
fig.savefig(
    "../../results/exp_2_norm_tests/normal_twin_49_acc_1000.png",
    bbox_inches="tight",
    dpi=DPI,
)

# %% RECONSTRUCTION ERROR
m = utils.load_pkl(os.path.join(BASE_DIR, "model_49.pkl"))
# test on test set
rec_df = get_rec(m, test_loader, class_labels, test_iters=TEST_ITERS)
# get avg accuracy  and add it to the dataframe
avg_rec = rec_df.groupby(["network", "iteration"])["rec_error"].mean().reset_index()
avg_rec["class"] = "AVG"
rec_df = pd.concat([rec_df, avg_rec])
# save to csv
rec_df.to_csv("../../results/exp_2_norm_tests/normal_twin_49_rec_1000.csv", index=False)

# %%
max_iters = rec_df["iteration"].max()
# just plot the data for the AVG class
plot_dat = rec_df[rec_df["class"] == "AVG"].set_index("network")
# plot only the data for the hybrid and pc networks
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

for i, net in enumerate(["hybrid", "pc"]):
    data = plot_dat.loc[net]
    ax.plot(
        data["iteration"],
        data["rec_error"],
        label=net,
    )
ax.set_ylim(0.0, 1.0)
ax.set_xlim(-1, max_iters)
ax.set_xlabel("Inference Step")
ax.set_ylabel("Reconstruction Error")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# Save the plot
fig.savefig(
    "../../results/exp_2_norm_tests/normal_twin_49_rec_1000.png",
    bbox_inches="tight",
    dpi=DPI,
)

# %%
