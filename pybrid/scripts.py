""" Scripts to run experiments with double amortised inference."""

import logging
from typing import Optional
import pickle as pkl
import os
from pprint import pprint
import torch
from pybrid import utils, datasets, optim
from pybrid.models import DoubleAmortModel
from pybrid.plots import plot_metrics
from pybrid.config import DefaultConfig


def setup(cfg: DefaultConfig):
    """Set ups an experiment with the given configuration"""
    cfg = utils.setup_experiment(cfg)
    test_batch_size = (
        cfg.optim.batch_size
        if cfg.optim.test_batch_size is None
        else cfg.optim.test_batch_size
    )

    train_dataset, test_dataset = datasets.get_dataset(cfg)
    train_loader = datasets.get_dataloader(train_dataset, cfg.optim.batch_size)
    test_loader = datasets.get_dataloader(test_dataset, test_batch_size)

    logging.info(
        "Loaded %s [train %d] [test %d]",
        cfg.data.dataset,
        len(train_loader),
        len(test_loader),
    )
    if cfg.model.model_pkl is not None:
        with open(cfg.model.model_pkl, "rb") as f:
            model = pkl.load(f)
            logging.info("Loaded model from file %s", cfg.model.model_pkl)
    else:
        model = DoubleAmortModel(
            nodes=cfg.model.nodes,
            amort_nodes=cfg.model.amort_nodes,
            n_amort_nets=cfg.model.n_amort_nets,
            mu_dt=cfg.infer.mu_dt,
            act_fn=utils.get_act_fn(cfg.model.act_fn),
            use_bias=cfg.model.use_bias,
            kaiming_init=cfg.model.kaiming_init,
            device=cfg.model.device,
        )

        logging.info("Created model %s", model)

    optimizer = optim.get_optim(
        model.params,
        cfg.optim.name,
        cfg.optim.lr,
        amort_lr=cfg.optim.amort_lr,
        batch_scale=cfg.optim.batch_scale,
        grad_clip=cfg.optim.grad_clip,
        weight_decay=cfg.optim.weight_decay,
    )
    return (
        cfg,
        train_loader,
        test_loader,
        model,
        optimizer,
    )


def main(cfg: DefaultConfig):
    """Train/test a model given a configuration"""
    (
        cfg,
        train_loader,
        test_loader,
        model,
        optimizer,
    ) = setup(cfg)
    infer_imgs, infer_labels, infer_contexts = utils.get_infer_set(test_loader)

    with torch.no_grad():
        metrics = utils.to_attr_dict(
            {
                "batch_idx": [],
                "hybrid_acc": [],
                "pc_acc": [],
                "amort_acc": [],
                "pc_losses": [],
                "pc_errs": [],
                "amort_losses": [],
                "amort_errs": [],
                "num_train_iters": [],
                "num_test_iters": [],
                "num_test_iters_pc": [],
                "init_errs": [],
                "final_errs": [],
            }
        )
        pc_losses, pc_errs, amort_losses, amort_errs, num_train_iters = (
            [],
            [],
            [],
            [],
            [],
        )
        final_errs, init_errs = [], []
        train_batches = len(train_loader)
        for curr_epoch in range(cfg.exp.num_epochs):
            logging.info("Epoch %d/%d", curr_epoch + 1, cfg.exp.num_epochs)
            for batch_id, (img_batch, label_batch, context_batch) in enumerate(
                train_loader
            ):
                num_train_iter, avg_err = model.train_batch(
                    img_batch,
                    label_batch,
                    context_batch,
                    cfg.infer.num_train_iters,
                    init_std=cfg.infer.init_std,
                    fixed_preds=cfg.infer.fixed_preds_train,
                    use_amort=cfg.model.train_amort,
                    thresh=cfg.infer.train_thresh,
                    delta_thresh=cfg.infer.delta_thresh,
                    no_backward=cfg.infer.no_backward,
                    supervised=cfg.model.supervised,
                    freeze_top=cfg.model.freeze_top,
                )

                optimizer.step(
                    curr_epoch=curr_epoch,
                    curr_batch=batch_id,
                    n_batches=len(train_loader),
                    batch_size=img_batch.size(0),
                )

                if cfg.optim.normalize_weights:
                    model.normalize_weights()

            if (((curr_epoch + 1) % cfg.exp.test_every) == 0) or (curr_epoch == 0):
                pc_loss, amort_loss = model.get_losses()
                pc_err, amort_err = model.get_errors()
                pc_losses.append(pc_loss)
                pc_errs.append(pc_err)
                amort_losses.append(amort_loss)
                amort_errs.append(amort_err)
                num_train_iters.append(num_train_iter)
                final_errs.append(avg_err[-1])
                init_errs.append(avg_err[0])

                # save the metrics for the epoch
                if curr_epoch > 0:
                    metrics.batch_idx.append(metrics.batch_idx[-1] + train_batches)
                else:
                    # special case for no previous epoch
                    metrics.batch_idx.append(train_batches)

                metrics.final_errs.append(sum(final_errs) / train_batches)
                metrics.pc_losses.append(sum(pc_losses) / train_batches)
                metrics.pc_errs.append(sum(pc_errs) / train_batches)
                metrics.amort_losses.append(sum(amort_losses) / train_batches)
                metrics.amort_errs.append(sum(amort_errs) / train_batches)
                metrics.num_train_iters.append(sum(num_train_iters) / train_batches)
                metrics.init_errs.append(sum(init_errs) / train_batches)

                # now do the tests on the test set
                logging.info("Test @ epoch %d", curr_epoch)
                test_res = do_test(model, cfg, test_loader)
                metrics.hybrid_acc.append(test_res.hybrid_acc)
                metrics.pc_acc.append(test_res.pc_acc)
                metrics.amort_acc.append(test_res.amort_acc)
                metrics.num_test_iters.append(test_res.num_test_iters)
                metrics.num_test_iters_pc.append(test_res.num_test_iters_pc)

                if cfg.exp.gen_infer_images:
                    logging.info(
                        "Generating image @ %s/%d.png", cfg.exp.img_dir, curr_epoch
                    )
                    label_preds, _, _ = model.test_batch(
                        infer_imgs,
                        infer_contexts,
                        cfg.infer.num_test_iters,
                        fixed_preds=cfg.infer.fixed_preds_test,
                        use_amort=cfg.model.train_amort,
                        thresh=cfg.infer.test_thresh,
                        delta_thresh=cfg.infer.delta_thresh,
                    )
                    # now do a backward pass with the label predictions
                    img_preds = model.backward(label_preds)
                    # post_process
                    img_preds = utils.postprocess_prediction(img_preds)
                    # make mosaic
                    img_preds = utils.make_mosaic(
                        img_preds,
                        nrow=1,
                        ncol=img_preds.shape[0],
                    )
                    img_name = cfg.exp.img_dir + f"/infer_img_{curr_epoch}.png"
                    # save
                    utils.save_img(img_preds, img_name)

                if cfg.exp.gen_label_images:
                    logging.info(
                        "Generating image @ %s/%d.png", cfg.exp.img_dir, curr_epoch
                    )
                    # now do a backward pass with the label predictions
                    img_preds = model.backward(infer_labels)
                    # post_process
                    img_preds = utils.postprocess_prediction(img_preds)
                    # make mosaic
                    img_preds = utils.make_mosaic(
                        img_preds,
                        nrow=1,
                        ncol=img_preds.shape[0],
                    )
                    img_name = cfg.exp.img_dir + f"/label_img_{curr_epoch}.png"
                    # save
                    utils.save_img(img_preds, img_name)

                logging.info("Metrics:")
                logging.info(pprint({k: v[-1] for k, v in metrics.items()}))
                utils.save_json(metrics, cfg.exp.log_dir + "/metrics.json")

                pc_losses, pc_errs, amort_losses, amort_errs, num_train_iters = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                final_errs, init_errs = [], []

                if cfg.exp.save_models:
                    model.reset()  # clean up states
                    logging.info(
                        "Saving model @ %s/model_%d.pkl", cfg.exp.log_dir, curr_epoch
                    )
                    utils.save_pkl(
                        model, os.path.join(cfg.exp.log_dir, f"model_{curr_epoch}.pkl")
                    )

    plot_metrics(metrics, path=cfg.exp.log_dir + "/metrics.png")
    logging.info("Saved metrics plot @ %s/metrics.png", cfg.exp.log_dir)

    # save model
    model.reset()
    utils.save_pkl(model, os.path.join(cfg.exp.log_dir, "final_model.pkl"))
    logging.info("Saved final model @ %s/%s", cfg.exp.log_dir, "final_model.pkl")


# VN: I've thrown this general testing loop to reduce the size of the main function
# Should be a model method in the future
def do_test(model, cfg, loader):
    """Test the model on a dataloader
    Arguments:
        model {HybridModel} -- the model to test
        cfg {dict} -- the experiment configuration
        loader {DataLoader} -- the loader to test on
    """
    hybrid_acc = 0
    pc_acc = 0
    amort_acc = 0
    num_test_iters = []
    num_test_iters_pc = []
    for test_batch_id, (img_batch, label_batch, context_batch) in enumerate(loader):
        if cfg.exp.test_hybrid:
            label_preds, num_test_iter, __path__ = model.test_batch(
                img_batch,
                context_batch,
                cfg.infer.num_test_iters,
                fixed_preds=cfg.infer.fixed_preds_test,
                use_amort=True,
                thresh=cfg.infer.test_thresh,
                delta_thresh=cfg.infer.delta_thresh,
            )
            hybrid_acc = hybrid_acc + datasets.accuracy(label_preds, label_batch)
            num_test_iters.append(num_test_iter)

        if cfg.exp.test_pc:
            label_preds, num_test_iter_pc, _ = model.test_batch(
                img_batch,
                cfg.infer.num_test_iters,
                init_std=cfg.infer.init_std,
                fixed_preds=cfg.infer.fixed_preds_test,
                use_amort=False,
                thresh=cfg.infer.test_thresh,
                delta_thresh=cfg.infer.delta_thresh,
            )
            pc_acc = pc_acc + datasets.accuracy(label_preds, label_batch)
            num_test_iters_pc.append(num_test_iter_pc)

        if cfg.exp.test_amort:
            label_preds = model.forward(img_batch, context_batch)
            amort_acc = amort_acc + datasets.accuracy(label_preds, label_batch)

    # determine effective test batches
    eff_tst_batches = test_batch_id + 1
    test_res = {
        "hybrid_acc": hybrid_acc / eff_tst_batches,
        "pc_acc": pc_acc / eff_tst_batches,
        "amort_acc": amort_acc / eff_tst_batches,
        "num_test_iters": sum(num_test_iters) / eff_tst_batches,
        "num_test_iters_pc": sum(num_test_iters_pc) / eff_tst_batches,
    }
    return utils.AttrDict(test_res)


def run_twins(
    cfg: DefaultConfig,
    progenitor_pkl: str,
    normal_dir: str = "./twins/normal_twin",
    swapped_dir: str = "./twins/swapped_twin",
):
    """Run the twin experiments"""
    # Check path to progenitor model
    assert os.path.exists(
        progenitor_pkl
    ), f"Progenitor model {progenitor_pkl} not found."
    # add to configuration
    cfg.model.model_pkl = progenitor_pkl

    # Assert that the progenitor cfg has superordinate mappings
    assert all(
        [
            cfg.data.train_sord is not None,
            cfg.data.test_sord is not None,
        ]
    ), "Progenitor cfg has no superordinate mapping"

    # Make copies of the progenitor configuration
    normal_cfg = cfg.copy()
    swapped_cfg = cfg.copy()
    # Train the normal twin
    normal_cfg.exp.log_dir = normal_dir
    logging.info(
        "Training swapped twin and saving results in %s", normal_cfg.exp.log_dir
    )
    main(normal_cfg)

    # Train the swapped twin
    swapped_cfg.exp.log_dir = swapped_dir
    logging.info(
        "Training swapped twin and saving results in %s", swapped_cfg.exp.log_dir
    )
    sord = swapped_cfg.data.train_sord.copy()
    # TODO: Get the trap from here
    trap_sord = swapped_cfg.data.trap_sord.copy()
    swapped_cfg.data.train_sord = trap_sord
    swapped_cfg.data.test_sord = trap_sord
    swapped_cfg.data.trap_sord = sord
    main(swapped_cfg)


def resume_training(
    cfg: DefaultConfig, model_pkl: str, output_dir: Optional[str] = None
):
    """Resume training of a model from a given checkpoint"""
    assert os.path.exists(model_pkl), f"Model pkl {model_pkl} not found."
    # add to configuration
    cfg.model.model_pkl = model_pkl
    # set output dir
    if output_dir is not None:
        cfg.exp.log_dir = output_dir
    logging.info("Resuming training from %s", model_pkl)
    logging.info("Saving results in %s", cfg.exp.log_dir)
    # run the main function
    main(cfg)
