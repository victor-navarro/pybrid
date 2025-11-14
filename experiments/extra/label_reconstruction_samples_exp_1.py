"""Makes a mosaic of images from the given pickle files."""

import numpy as np
from PIL import Image, ImageFont
from pybrid import datasets
from pybrid import utils

font = ImageFont.truetype("arial.ttf", 25)


def remake_images():
    """Remake images from the given pickle files."""
    config_file = "results/exp_1_norm/progenitor/0/config.json"
    cfg = utils.load_json_config(config_file)
    utils.seed(cfg.exp.seed)

    # Get the loaders based on configuration
    _, test_dataset = datasets.get_dataset(cfg)
    test_loader = datasets.get_dataloader(test_dataset, cfg.optim.batch_size)
    infer_set = utils.get_infer_set(test_loader)

    preds = {"hybrid": [], "pc": [], "amort": []}

    for epoch in [0, 99]:
        pkl = cfg.exp.log_dir + f"/model_{epoch}.pkl"
        model = utils.load_pkl(pkl)
        # Make inference images
        imgs, _, contexts = infer_set
        img_origs = utils.postprocess_prediction(imgs)

        labs, _, _ = model.test_batch(
            imgs,
            contexts,
            100,
            fixed_preds=cfg.infer.fixed_preds_test,
            use_amort=True,
        )
        # make mosaic with images (interleaved with the original images)
        preds["hybrid"].append(utils.postprocess_prediction(model.backward(labs)))

        # Now do the same but with the pc component alone
        labs, _, _ = model.test_batch(
            imgs,
            contexts,
            100,
            fixed_preds=cfg.infer.fixed_preds_test,
            use_amort=False,
        )
        preds["pc"].append(utils.postprocess_prediction(model.backward(labs)))

        # Now do the same but with the amort component alone
        labs, _, _ = model.test_batch(
            imgs,
            contexts,
            0,
            fixed_preds=cfg.infer.fixed_preds_test,
            use_amort=True,
        )
        preds["amort"].append(utils.postprocess_prediction(model.backward(labs)))

    # convert L to RGB
    for k in preds.keys():
        for i in range(len(preds[k])):
            preds[k][i] = np.repeat(preds[k][i][:, :, :, np.newaxis], 3, axis=3)

    # and the same for the original images
    img_origs = np.repeat(img_origs[:, :, :, np.newaxis], 3, axis=3)

    # Now we create a mosaic for both epochs
    for e, epoch in enumerate([0, 99]):
        all_imgs = np.vstack(
            [img_origs, preds["hybrid"][e], preds["pc"][e], preds["amort"][e]]
        )
        mosaic = utils.make_mosaic(
            all_imgs,
            nrow=4,
            ncol=12,
            padding=0,
        )
        img_path = f"plots/exp_1_norm/progenitor_{epoch}_mosaic_label.png"
        Image.fromarray(mosaic).save(img_path)

    # and now the same for the twins
    for twin in ["normal", "swapped"]:
        preds = {"hybrid": [], "pc": [], "amort": []}
        for epoch in [0, 49]:
            config_file = f"results/exp_1_norm/{twin}_twin/0/config.json"
            cfg = utils.load_json_config(config_file)
            utils.seed(cfg.exp.seed)

            pkl = cfg.exp.log_dir + f"/model_{epoch}.pkl"
            model = utils.load_pkl(pkl)

            # Make inference images
            labs, _, _ = model.test_batch(
                imgs,
                contexts,
                100,
                fixed_preds=cfg.infer.fixed_preds_test,
                use_amort=True,
            )

            # make mosaic with images (interleaved with the original images)
            preds["hybrid"].append(utils.postprocess_prediction(model.backward(labs)))

            # Now do the same but with the pc component alone
            labs, _, _ = model.test_batch(
                imgs,
                contexts,
                100,
                fixed_preds=cfg.infer.fixed_preds_test,
                use_amort=False,
            )

            preds["pc"].append(utils.postprocess_prediction(model.backward(labs)))

            labs, _, _ = model.test_batch(
                imgs,
                contexts,
                0,
                fixed_preds=cfg.infer.fixed_preds_test,
                use_amort=True,
            )
            preds["amort"].append(utils.postprocess_prediction(model.backward(labs)))
            # Now we create a mosaic for both epochs

        # convert L to RGB
        for k in preds.keys():
            for i in range(len(preds[k])):
                preds[k][i] = np.repeat(preds[k][i][:, :, :, np.newaxis], 3, axis=3)
        for e, epoch in enumerate([0, 49]):
            all_imgs = np.vstack(
                [img_origs, preds["hybrid"][e], preds["pc"][e], preds["amort"][e]]
            )
            mosaic = utils.make_mosaic(
                all_imgs,
                nrow=4,
                ncol=12,
                padding=0,
            )
            img_path = f"plots/exp_1_norm/{twin}_{epoch}_mosaic_label.png"
            Image.fromarray(mosaic).save(img_path)


if __name__ == "__main__":
    remake_images()
