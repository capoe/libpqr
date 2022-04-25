#! /usr/bin/env python
import os
import sys
import re
import gzip
import json
import multiprocessing as mp
import functools

import rdkit.Chem as chem
import torch
import torch.nn.functional as F
import sklearn.metrics
import numpy as np

import benchml as bml
log = bml.log

from libpqr.aux import (
    Checkpointer, 
    Timer
)

from libpqr.gen1d import (
    load_model_p
)

from libpqr.gen2d import (
    get_default_settings, 
    collate,
    mol_to_tensors,
    data_to_torch,
    build_sampler, 
    build_model, 
    build_opt,
    build_featurizer
)


def train(args):
    settings = get_default_settings()
    settings.baseline = load_model_p(args.vocabulary, args.device)
    if args.input == "":
        if args.recalibrate:
            raise ValueError("Missing 'input' for recalibration")
        feat, model = build_model(args.device)
    else:
        feat = build_featurizer()
        model = torch.load(args.input, map_location=args.device)
    if args.recalibrate:
        model.prepare_recalibrate()
    opt = build_opt(model, lr=10**-3, weight_decay=0.)
    chker = Checkpointer(
        location=os.path.dirname(args.output), 
        format_str="{tag:s}_epoch{epoch:02d}.arch.chk",
        keep=40
    )
    for epoch in range(args.epochs):
        with gzip.open(args.structures, 'rt') as f:
            smiles = json.load(f)
        epochsize = min(args.max_epochsize, len(smiles))
        smiles = [ smiles[_] for _ in np.random.permutation(len(smiles))[0:epochsize]]
        sampler = build_sampler(smiles=smiles, settings=settings)
        model = train_epoch(model, opt, sampler, args, settings)
        chker.save(model, tag="vlb", epoch=epoch)
    return model


def train_epoch(model, opt, sampler, args, settings):
    for batchidx, data in enumerate(sampler.batches(
        chunksize=min(args.max_chunksize, sampler.size()),
        batchsize=min(args.max_batchsize, sampler.size()//20),
        shuffle=True,
        procs=args.procs
    )):
        timer = Timer()
        opt.zero_grad()
        data = data.to(args.device)
        with timer.time("vector"):
            out_vector, y_vector, loss_vector = step_vector(model, opt, data, args)
        with timer.time("linker"):
            out_linker, y_linker, loss_linker = step_linker(model, opt, data, args, settings)
        with timer.time("bonding"):
            out_bonding, y_bonding, loss_bonding = step_bonding(model, opt, data, args)

        loss = loss_vector + loss_linker + loss_bonding
        loss.backward()
        opt.step()

        log << f"Batch {batchidx:5d} v= {loss_vector.item():.4f}" \
            << f"l= {loss_linker.item():.4f}   b= {loss_bonding.item():.4f}" \
            << log.flush
        eval_batch_vector(y_vector, out_vector)
        eval_batch_linker(y_linker, out_linker)
        eval_batch_bonding(y_bonding, out_bonding)
        if batchidx % 10 == 0:
            log << timer.report() << log.flush
        log << log.endl

    return model



def step_vector(model, opt, data, args):
    out = model.vector_model(
        data.x, 
        data.edge_index, 
        data.edge_attr, 
        data.batch
    )
    loss = F.binary_cross_entropy(out, data.atom_vector_label)
    return out, data.atom_vector_label, loss


def eval_batch_vector(yi, out):
    auc = sklearn.metrics.roc_auc_score(
        yi.cpu().numpy().flatten(), 
        out.detach().cpu().numpy().flatten())
    log << f" v_auc= {auc:.4f}" << log.flush


def step_linker(model, opt, data, args, settings):
    # Observations
    xa, xb = model.linker_model(
        data.x, 
        data.edge_index, 
        data.edge_attr, 
        data.batch
    )
    ij = data.pair_link_index
    xai = xa[ij[0]]
    xbi = xb[ij[0]]
    xaj = xa[ij[1]]
    xbj = xb[ij[1]]
    yij = torch.ones((ij.shape[1],1))

    # Baseline
    if args.baseline == "explicit":
        #data_base = settings.baseline.sample(yij.shape[0], "1d")
        data_base = sample_baseline(yij.shape[0], args, settings)
        xa_base, xb_base = model.linker_model(
            data_base.x,
            data_base.edge_index,
            data_base.edge_attr,
            data_base.batch
        )
        j_base = data_base.atom_vector
        xai_base = xai
        xbi_base = xbi
        xaj_base = xa_base[j_base]
        xbj_base = xb_base[j_base]
        yij_base = torch.zeros((ij.shape[1],1))
    elif args.baseline == "permutations":
        j_base = torch.randperm(len(ij[0]))
        xai_base = xai
        xbi_base = xbi
        xaj_base = xaj[j_base]
        xbj_base = xbj[j_base]
        yij_base = torch.zeros((ij.shape[1],1))
    else:
        raise ValueError("--baseline option '%s' not recognized" % (args.baseline))

    # Concat and loss
    xai = torch.cat([ xai, xai_base ], dim=0)
    xbi = torch.cat([ xbi, xbi_base ], dim=0)
    xaj = torch.cat([ xaj, xaj_base ], dim=0)
    xbj = torch.cat([ xbj, xbj_base ], dim=0)
    yij = torch.cat([ yij, yij_base ], dim=0).to(args.device)

    out = model.linker_model.final(xai, xbi, xaj, xbj)
    loss = F.binary_cross_entropy(out, yij)
    return out, yij, loss


def eval_batch_linker(yij, out):
    auc = sklearn.metrics.roc_auc_score(
        yij.cpu().numpy().flatten(), 
        out.detach().cpu().numpy().flatten())
    log << f" l_auc= {auc:.4f}" << log.flush


def mol_to_tensors_baseline(m, settings):
    data = mol_to_tensors(m["mol"], settings=settings, training=False)
    data.atom_vector = torch.tensor([m["vec"]]).long().numpy()
    return data


def sample_baseline(n, args, settings):
    motifs, probs = settings.baseline.sample(n, "1d")
    func = functools.partial(mol_to_tensors_baseline, settings=settings)
    datalist = list(map(func, motifs)) # TODO Multiprocessing?
    datalist = [ data_to_torch(_) for _ in datalist ]
    data = collate(datalist)
    data = data.to(args.device)
    return data


def step_bonding(model, opt, data, args):
    xout = model.bonding_model(
        data.x, 
        data.edge_index, 
        data.edge_attr, 
        data.batch
    )
    # Concat and score
    ij = data.pair_link_index
    xa = xout[ij[0]]
    xb = xout[ij[1]]

    out = model.bonding_model.final(xa, xb)
    tar = data.pair_link_type
    # Loss
    loss = F.binary_cross_entropy(out, tar)
    return out, data.pair_link_type, loss


def eval_batch_bonding(yij, out):
    for ch in range(yij.shape[1]):
        yt = yij[:,ch].cpu().numpy().flatten()
        if np.max(yt) < 0.5:
            auc = 0.
        else:
            yp =  out[:,ch].detach().cpu().numpy().flatten()
            try:
                auc = sklearn.metrics.roc_auc_score(yt, yp)
            except ValueError:
                auc = 0.
        log << f" b_auc[ch{ch:d}]= {auc:.4f} " << log.flush


if __name__ == "__main__":
    log.Connect()
    log.AddArg("structures", str)
    log.AddArg("vocabulary", str)
    log.AddArg("output", str)
    log.AddArg("input", str, default="")
    log.AddArg("epochs", int, default=40)
    log.AddArg("recalibrate", "toggle", default=False)
    log.AddArg("max_epochsize", int, default=250000)
    log.AddArg("max_chunksize", int, default=10000)
    log.AddArg("max_batchsize", int, default=200)
    log.AddArg("baseline", str, default="permutations")
    log.AddArg("seed", int, default=976313)
    log.AddArg("dev", "toggle", default=False)
    log.AddArg("procs", int, default=8)
    args = log.Parse()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed+1)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model = train(args)
    torch.save(model, args.output)


