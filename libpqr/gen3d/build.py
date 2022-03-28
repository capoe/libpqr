import os
import sys
import glob
import json
import functools 
import multiprocessing as mp

import numpy as np
import torch

from torch_geometric.loader.dataloader import Collater

from ..aux import Timer
from ..data import DatasetPartition
from .arch import LexFeaturizer, LexComponents
from .aux import complex_to_tensors, data_to_torch


def build_featurizer():
    return LexFeaturizer()


def build_collater():
    return Collater(["lig"], [])


def build_model(settings, device):
     feat = build_featurizer()
     lex_module = LexComponents(
            feat=feat,
            settings=settings
         ).to(device)
     lex_module.reset_parameters()
     return feat, lex_module


def build_opt(model, lr=10**-4):
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=0.
    )
    print(model)
    print(" #params:", model.num_parameters())
    return optimizer


def build_sampler(*args, **kwargs):
    return LexPreprocessor(*args, **kwargs)


class LexPreprocessor:
    def __init__(self, 
            settings
    ):
        self.settings = settings
        self.skiplist = json.load(open(settings.data_skiplist))
        self.invert_skiplist = settings.invert_skiplist
        complexes = json.load(open(settings.data_complexes))
        self.shuffle = settings.data_shuffle
        self.current_idx = 0
        self.collate = build_collater()
        self.timer = Timer()

        # Apply skip-list
        n_in = len(complexes)
        if self.invert_skiplist:
            complexes = list(filter(lambda c: c["code"] in self.skiplist, complexes))
        else:
            complexes = list(filter(lambda c: c["code"] not in self.skiplist, complexes))
        n_out = len(complexes)
        print("Preprocessor: Have %d/%d complexes" % (n_out, n_in))

        # Shuffle
        if self.shuffle:
            order = np.random.permutation(len(complexes))
            complexes = [ complexes[_] for _ in order ]
        self.samples = complexes

    def size(self):
        return len(self.samples)

    def batches(self, chunksize, batchsize, shuffle, procs):
        n_samples = len(self.samples)
        print("Sampler: %d samples" % n_samples)
        n_chunks = n_samples // chunksize
        chunksize = int(n_samples / n_chunks)
        n_rest = n_samples % chunksize
        if shuffle:
            order = np.random.permutation(n_samples)
        else:
            order = np.arange(0, n_samples)
        mp_func = functools.partial(
            complex_to_tensors, 
            settings=self.settings, 
        )
        i = 0
        for ch in range(n_chunks):
            j = i + chunksize
            if ch < n_rest:
                j += 1
            print("Sampler: Partition %3d:%-3d" % (i,j))
            samples = self.samples[i:j]
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
                collate=False,
                follow_batch=self.collate.follow_batch
            )
            for batch in part:
                yield batch
            i = j


