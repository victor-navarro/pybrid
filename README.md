# A hybrid predictive coding account of delusions

This folder contains the Python scripts required to reproduce the experiments reported in our paper.

We developed everything on top of Alec Tschantz's repository implementing the hybrid predictive coding model reported in their paper (Hybrid inference: Inferring fast and slow). You can find the original source code here.

https://github.com/alec-tschantz/pybrid/

You can find their paper here:

https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011280

## Installation

1. Clone our fork of the pybrid repository.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Install the pybrid package by running `pip install .` in the root folder of the cloned repository.

All the experiments were run using Python 3.8.18 and pytorch 1.8.0.
Recommended: Environments built on top of Python 3.11 also work well (06/26).

## Usage

The `experiments` folder contains numbered scripts. Run them in order. You can run scripts on the same level (e.g., `01_*`) in any order.

