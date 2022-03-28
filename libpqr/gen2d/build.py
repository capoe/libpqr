import os
import sys
import glob
import json
import functools 
import multiprocessing as mp

import torch
import numpy as np

from torch_geometric.loader.dataloader import Collater

from ..aux import Timer
from ..data import DatasetPartition
from .arch import VlbFeaturizer, VlbComponents
from .aux import mol_to_tensors, data_to_torch


def build_featurizer():
    return VlbFeaturizer()


def build_model(device):
    print("Build model ...") 
    feat = build_featurizer()
    module = VlbComponents(
       feat=feat
    ).to(device)
    module.reset_parameters()
    return feat, module


def build_sampler(*args, **kwargs):
    return VlbPreprocessor(*args, **kwargs)


def build_opt(model, lr=10**-3, weight_decay=0.):
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )
    param_count = sum([ par.data.numel() for par in model.parameters() ])
    print("Optimizer: #params=", param_count)
    return optimizer


class VlbPreprocessor:
    def __init__(self, 
            smiles,
            settings
    ):
        self.smiles = smiles
        self.settings = settings
        self.timer = Timer()

    def size(self):
        return len(self.smiles)

    def __len__(self):
        return self.size()

    def batches(self, chunksize, batchsize, shuffle, procs):
        n_samples = len(self)
        print("Sampler: %d samples" % n_samples)
        n_chunks = n_samples // chunksize
        chunksize = int(n_samples / n_chunks)
        n_rest = n_samples % chunksize
        if shuffle:
            order = np.random.permutation(n_samples)
        else:
            order = np.arange(0, n_samples)
        mp_func = functools.partial(
            mol_to_tensors, 
            settings=self.settings, 
            training=True
        )
        i = 0
        for ch in range(n_chunks):
            j = i + chunksize
            if ch < n_rest:
                j += 1
            print("Sampler: Partition %3d:%-3d" % (i,j))
            samples = self.smiles[i:j]
            with self.timer.time("precompute"):
                if procs > 1:
                    with mp.Pool(procs) as pool:
                        datalist = pool.map(mp_func, samples)
                else:
                    datalist = list(map(mp_func, samples))
            datalist = list(filter(lambda d: d is not None, datalist))
            datalist = list(map(lambda d: data_to_torch(d), datalist))
            print("Sampler: Datalist of len", len(datalist))
            part = DatasetPartition(
                datalist, 
                batchsize=batchsize, 
                shuffle_iter=shuffle,
                collate=False
            )
            for batch in part:
                yield batch
            i = j
        print(self.timer.report())

