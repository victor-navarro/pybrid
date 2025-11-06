# A fork for the Teufel lab

This is a fork of the hybrid predictive coding model repository by Tschantz et al. group. I've tweaked the `requirements.txt` file a little to make it installable via miniconda in a Windows machine.

### Installation requirements

Note: these installation requirements are what I (Victor) have tested with. There's nothing critical in these libraries that will break functionality with newer versions (famous last words).

- Python 3.8
- PyTorch 1.8.0

### Installation instructions

- Create a conda environment with the correct python version (e.g., `conda create pybrid python=3.8`).
- Optional: If your computer has a CUDA-capable GPU, install pytorch 1.8.0 with with cuda capabilities. See (https://pytorch.org/get-started/previous-versions/)
- Install the rest of the dependencies via pip (`pip install -r requirements.txt`)
- Install the package via pip (`pip install -e .`)

### FAQ 

- **Q**: EMNIST doesn't download. What do I do?
- **A**: Check if the links in utils.run_mnist_dl or utils.run_emnist_dl work. If not, replace the links with wherever NIST has moved the datasets.


