<div align="center">
    <img src="docs/pqr.png" width="450px">
</div>

# 

This package implements the *PQR* framework -- a generative approach to structure-based ligand elaboration. The framework consists of a multi-level contrastive learning protocol that constructs a generative posterior as a product of context factors, representing 1D, 2D and 3D context information. A description of the method can be found [here](https://arxiv.org/abs/2204.10663).

This particular implementation uses stochastic reconstructions during model training, with the 2D and 3D context factors represented by graph-convolutional and hypergraph-convolutional neural networks, respectively.

## Installation

### System requirements

The code has been tested on 64-bit Linux only. GPU support is essential for model training, with a recommended GPU RAM of at least 16GB. 

### Installation through conda/pip

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

The pretrained models provided in the ./models directory have been obtained with the following configuration:
```
torch==1.8.2+cu102
torch-cluster==1.6.0
torch-geometric==2.0.4
torch-scatter==2.0.9
torch-sparse==0.6.12
rdkit-pypi==2021.9.5.1
libpqr==0.1.0
```

## Getting started

The workflow is described [here](./workflow). The instructions will guide you through the individual training steps. Note that, to build performant models, you will first need to download and preprocess the complete datasets.


## Citation
A description of the framework is available on arXiv -- please cite this if you find the method and/or code useful:
```
@article{chan_3d_2022,
  doi = {10.48550/ARXIV.2204.10663},
  url = {https://arxiv.org/abs/2204.10663},
  author = {Chan, Lucian and Kumar, Rajendra and Verdonk, Marcel and Poelking, Carl},
  keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), Biomolecules (q-bio.BM)},
  title = {3D pride without 2D prejudice: Bias-controlled multi-level generative models for structure-based ligand design},
  publisher = {arXiv},
  year = {2022}
}
```
