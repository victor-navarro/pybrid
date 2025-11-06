""" Creates a gif of cEMNIST images in a folder."""

import os
from pybrid.plots import animate


def animate_images(png_folder, output_folder, **kwargs):
    """Animate images in a folder."""
    animate(png_folder, webm_path=output_folder, **kwargs)
    print(f"Successfully created {output_folder}")


if __name__ == "__main__":
    seeds = list(range(1))
    for seed in seeds:
        # set base folder to layerwise_mnist_full
        base_folder = f"results/exp_1/progenitor/{seed}/reconstruction_imgs/"
        os.makedirs(f"plots/exp_1/progenitor/{seed}", exist_ok=True)
        label_file = f"plots/exp_1/progenitor/{seed}/label.webm"

        # create inference webm
        for t in ["hybrid", "pc", "amort"]:
            infer_file = f"plots/exp_1/progenitor/{seed}/infer_{t}.webm"
            animate_images(
                base_folder,
                infer_file,
                pattern=f"infer_img_{t}",
                font_size=0,
            )
        animate_images(base_folder, label_file, pattern="label_img", font_size=0)

        # now do the twins
        for twin in ["normal_twin", "swapped_twin"]:
            twin_folder = f"results/exp_1/{twin}/{seed}/reconstruction_imgs/"
            os.makedirs(f"plots/exp_1/{twin}/{seed}", exist_ok=True)
            label_file = f"plots/exp_1/{twin}/{seed}/label.webm"

            # create inference webm
            for t in ["hybrid", "pc", "amort"]:
                infer_file = f"plots/exp_1/{twin}/{seed}/infer_{t}.webm"
                animate_images(
                    twin_folder,
                    infer_file,
                    pattern=f"infer_img_{t}",
                    font_size=0,
                )
            animate_images(twin_folder, label_file, pattern="label_img", font_size=0)
