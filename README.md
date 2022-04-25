<div align="center">
    <img src="docs/pqr.png" width="450px">
</div>

# 

This package implements the *PQR* framework -- a generative approach to structure-based ligand elaboration. The framework consists of a multi-level contrastive learning protocol that constructs a generative posterior as a product of context factors, representing 1D, 2D and 3D context information. A description of the method can be found [here](https://arxiv.org/abs/2204.10663).

This particular implementation uses stochastic reconstructions during model training, with the 2D and 3D context factors represented by graph-convolutional and hypergraph-convolutional neural networks, respectively.

## Installation

1. Set up a new python3 conda environment
2. Install [pytorch](https://pytorch.org)
3. Install [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
4. Install [rdkit](https://pypi.org/project/rdkit-pypi)
5. Install [benchml](https://pypi.org/project/benchml)
6. Clone and install libpqr

An example installation script looks like this:
```bash
conda create --prefix "./venv" python=3.8
conda activate ./venv
pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.8.1+cu102.html
pip3 install rdkit-pypi
pip3 install benchml
git clone https://github.com/capoe/libpqr.git
cd libpqr
pip install .
```

## Getting started

The workflow is described [here](./workflow). The instructions will guide you through the individual training steps. Note that, to build performant models, you will first need to download the complete datasets.
