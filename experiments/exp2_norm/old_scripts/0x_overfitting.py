# %%
""" Check overfitting"""

import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pybrid.datasets import get_dataset, get_dataloader
from pybrid import utils
from pybrid.tests import test_model
from pybrid.postprocessing.label_accuracy import _agg_acts
from pybrid.postprocessing.reconstruction_error import _agg_errs


# %%
TEST_BATCHES = 4
TEST_BATCH_SIZE = 512
TEST_ITERS = 100

history = [
    "../../results/exp_1_norm/progenitor/0",
    "../../results/exp_1_norm/normal_twin/0",
    "../../results/exp_2_norm/normal_twin/49/0",
]
# get datasets
# load progenitor config
cfg = utils.load_json_config(os.path.join(history[0], "config.json"))
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


# %%
results = pd.DataFrame(columns=["bi", "epoch", "set", "accuracy"])
for bi, base_dir in enumerate(history):
    models = os.listdir(base_dir)
    models = [m for m in models if m.startswith("model") and m.endswith(".pkl")]
    for model in tqdm(models, desc=f"bi={bi}"):
        m = utils.load_pkl(os.path.join(base_dir, model))
        epoch = int(model.split("_")[1].split(".")[0])
        # test on training set
        train_df = get_acc(m, train_loader, class_labels, test_iters=TEST_ITERS)
        train_df["set"] = "train"
        train_df["bi"] = bi
        train_df["epoch"] = epoch
        # test on test set
        test_df = get_acc(m, test_loader, class_labels, test_iters=TEST_ITERS)
        test_df["set"] = "test"
        test_df["bi"] = bi
        test_df["epoch"] = epoch

        results = pd.concat([results, train_df, test_df])


# %%
results = results.sort_values(by=["bi", "epoch", "set", "class", "iteration"])
results["epoch"][results["bi"] == 1] += 100
results["epoch"][results["bi"] == 2] += 150

# save results
results.to_csv("../../results/exp_2_norm_tests/overfitting_acc.csv", index=False)

# %%
# aggregate across classes
agg_results = (
    results.groupby(["bi", "epoch", "set", "network", "iteration"])["accuracy"]
    .mean()
    .reset_index()
)
# only get the last iteration per network
agg_results = (
    agg_results.groupby(["bi", "epoch", "set", "network"]).last().reset_index()
)
# group by network
agg_results = agg_results.groupby(["network"])
# plot the results
fig, ax = plt.subplots(1, 3, figsize=(8, 3), layout="constrained")
for i, (network, data) in enumerate(agg_results):
    ax[i].axvline(100, color="gray", linestyle="--")
    ax[i].axvline(150, color="gray", linestyle="--")
    for s in ["train", "test"]:
        ax[i].plot(
            data.loc[data["set"] == s]["epoch"],
            data.loc[data["set"] == s]["accuracy"],
            label=s,
        )
    ax[i].set_ylim(0.0, 1.0)
    ax[i].set_title(network[0])
    ax[i].set_xlabel("Epoch")
    ax[i].set_ylabel("Accuracy")
ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.savefig("../../results/exp_2_norm_tests/overfitting_acc.png", dpi=300)


# %%
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
rec_results = pd.DataFrame()
for bi, base_dir in enumerate(history):
    models = os.listdir(base_dir)
    models = [m for m in models if m.startswith("model") and m.endswith(".pkl")]
    for model in tqdm(models, desc=f"bi={bi}"):
        m = utils.load_pkl(os.path.join(base_dir, model))
        epoch = int(model.split("_")[1].split(".")[0])
        # test on training set
        train_df = get_rec(m, train_loader, class_labels, test_iters=TEST_ITERS)
        train_df["set"] = "train"
        train_df["bi"] = bi
        train_df["epoch"] = epoch
        # test on test set
        test_df = get_rec(m, test_loader, class_labels, test_iters=TEST_ITERS)
        test_df["set"] = "test"
        test_df["bi"] = bi
        test_df["epoch"] = epoch

        rec_results = pd.concat([rec_results, train_df, test_df])

# %%
rec_results = rec_results.sort_values(by=["bi", "epoch", "set", "class", "iteration"])
rec_results["epoch"][rec_results["bi"] == 1] += 100
rec_results["epoch"][rec_results["bi"] == 2] += 150

# save rec_results
rec_results.to_csv("../../results/exp_2_norm_tests/overfitting_rec.csv", index=False)

# %%
# aggregate across classes
rec_agg_results = (
    rec_results.groupby(["bi", "epoch", "set", "network", "iteration"])["rec_error"]
    .mean()
    .reset_index()
)
# only get the last iteration per network
rec_agg_results = (
    rec_agg_results.groupby(["bi", "epoch", "set", "network"]).last().reset_index()
)
# group by network
rec_agg_results = rec_agg_results.groupby(["network"])
# plot the results
fig, ax = plt.subplots(1, 3, figsize=(8, 3), layout="constrained")
for i, (network, data) in enumerate(rec_agg_results):
    ax[i].axvline(100, color="gray", linestyle="--")
    ax[i].axvline(150, color="gray", linestyle="--")
    for s in ["train", "test"]:
        ax[i].plot(
            data.loc[data["set"] == s]["epoch"],
            data.loc[data["set"] == s]["rec_error"],
            label=s,
        )
    ax[i].set_ylim(0.0, 1.0)
    ax[i].set_title(network[0])
    ax[i].set_xlabel("Epoch")
    ax[i].set_ylabel("Reconstruction error")
ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.savefig("../../results/exp_2_norm_tests/overfitting_rec.png", dpi=300)

# %%
