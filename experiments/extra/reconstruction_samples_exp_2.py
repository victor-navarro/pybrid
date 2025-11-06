""" Makes a mosaic of images from the given pickle files. """

import numpy as np
from PIL import Image
from pybrid import datasets
from pybrid import utils


EPOCHS = [0, 49]


def remake_images():
    """Remake images from the given pickle files."""
    config_file = "results/exp_2_norm/normal_twin/0/0/config.json"
    cfg = utils.load_json_config(config_file)
    utils.seed(cfg.exp.seed)

    # Get the loaders based on configuration
    _, test_dataset = datasets.get_dataset(cfg)
    test_loader = datasets.get_dataloader(test_dataset, cfg.optim.batch_size)
    infer_set = utils.get_infer_set(test_loader)
    # Make inference images
    imgs, _, contexts = infer_set
    # and now the same for the twins
    for twin in ["normal", "swapped"]:
        config_file = f"results/exp_2_norm/{twin}_twin/0/0/config.json"
        cfg = utils.load_json_config(config_file)
        utils.seed(cfg.exp.seed)
        for epoch in EPOCHS:

            pkl = cfg.exp.log_dir + f"/model_{epoch}.pkl"
            model = utils.load_pkl(pkl)

            # Make inference images
            label_preds, _, _ = model.test_batch(
                imgs,
                contexts,
                100,
                fixed_preds=cfg.infer.fixed_preds_test,
                use_amort=True,
            )

            # make mosaic with images (interleaved with the original images)
            img_preds = utils.postprocess_prediction(model.preds[-1])
            img_origs = utils.postprocess_prediction(imgs)

            # we need to interleave the images a little bit
            all_imgs = np.vstack(
                [img_origs[:6], img_preds[:6], img_origs[6:], img_preds[6:]]
            )

            img_name = f"plots/exp_2_norm/{twin}_{epoch}_hybrid_mosaic.png"
            infer_mosaic = utils.make_mosaic(
                all_imgs,
                nrow=4,
                ncol=6,
            )
            # save
            Image.fromarray(infer_mosaic).save(img_name)

            # Make inference images
            label_preds, _, _ = model.test_batch(
                imgs,
                contexts,
                100,
                fixed_preds=cfg.infer.fixed_preds_test,
                use_amort=False,
            )

            # make mosaic with images (interleaved with the original images)
            img_preds = utils.postprocess_prediction(model.preds[-1])
            img_origs = utils.postprocess_prediction(imgs)

            # we need to interleave the images a little bit
            all_imgs = np.vstack(
                [img_origs[:6], img_preds[:6], img_origs[6:], img_preds[6:]]
            )

            img_name = f"plots/exp_2_norm/{twin}_{epoch}_pc_mosaic.png"
            infer_mosaic = utils.make_mosaic(
                all_imgs,
                nrow=4,
                ncol=6,
            )
            # save
            Image.fromarray(infer_mosaic).save(img_name)


if __name__ == "__main__":
    remake_images()
