import os
import time

import torch
import numpy as np
import rdkit.Chem as chem


class Timer(object):
    def __init__(self):
        self.stages = []
        self.times = {}
        self.current_stage = None
        self.t_in = None
        self.t_out = None

    def time(self, stage):
        if not stage in self.times:
            self.times[stage] = []
            self.stages.append(stage)
        else:
            pass
        self.current_stage = stage
        return self

    def __enter__(self):
        self.t_in = time.time()
        return self.current_stage

    def __exit__(self, *args, **kwargs):
        self.t_out = time.time()
        self.times[self.current_stage].append(self.t_out - self.t_in)

    def __call__(self, stage):
        return self.time(stage)

    def report(self, inline=False, delim=None):
        dt_tot = sum([ t for stage in self.stages for t in self.times[stage] ])
        if delim is not None:
            pass
        elif len(self.stages) > 3:
            delim = " " if inline else " \n"
        else:
            delim = " "
        return delim.join(
                list(map(lambda stage: " %20s = %1.4fs" % (
                    "dt(%s)" % stage, 
                    sum(self.times[stage])), 
                    self.stages))
            ) + " || total = %1.4fs" % dt_tot

    def pop(self):
        return sum(self.times[self.stages[-1]])


class Checkpointer:
    def __init__(self, 
            location, 
            format_str="{tag:s}_epoch{epoch:02d}_part{partition:02d}.arch.chk",
            keep=10
    ):
        self.location = location
        self.format_str = format_str
        self.keep = keep
        self.latest = -1
        self.paths = [ None for _ in range(keep) ]

    def save(self, model, tag, epoch, partition=-1):
        os.makedirs(self.location, exist_ok=True)
        save_at = (self.latest + 1) % self.keep
        if self.paths[save_at] is not None:
            os.remove(self.paths[save_at])
        path = os.path.join(self.location, self.format_str.format(
            tag=tag,
            epoch=epoch,
            partition=partition
        ))
        self.paths[save_at] = path
        print("Saving model to '%s' ..." % path)
        torch.save(model, path)
        print("... done")
        self.latest = save_at


def stochastic_argsort(weights):
    if not isinstance(weights, list):
        weights = weights.tolist()
    order = []
    idcs = list(range(len(weights)))
    while len(idcs) > 1:
        w = np.array(weights).cumsum()
        i = np.searchsorted(w, np.random.uniform(0., w[-1]))
        order.append(idcs.pop(i))
        weights.pop(i)
    order.append(idcs.pop())
    return order


def stochastic_argmax(weights):
    w = np.array(weights).cumsum()
    i = np.searchsorted(w, np.random.uniform(0., w[-1]))
    return i


def reorder_along_smiles(mol, **kwargs):
    smi = chem.MolToSmiles(mol, **kwargs)
    order = list(map(int, mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
    return chem.MolFromSmiles(smi), { o: oo for oo, o in enumerate(order) }


def canonical_atom_index(mol, idx):
    ranks = chem.CanonicalRankAtoms(mol, breakTies=False)
    ranks = list(ranks)
    idx = ranks.index(ranks[idx])
    return mol, idx


def normalize_atom_index(mol, idx):
    mol, imap = reorder_along_smiles(mol, isomericSmiles=False)
    idx = imap[idx]
    mol, idx = canonical_atom_index(mol, idx)
    return mol, idx


def tensor_info(x, varname):
    if x.numel() > 0:
        print(f"Info:  {varname:15s}  shape={str(x.shape).replace('torch.Size',''):20s}  min, mean, max= " 
            f"{torch.min(x):+.4f}, {torch.mean(x):+.4f}, {torch.max(x):+.4f}  std= {torch.std(x):+.4f}")
    else:
        print(f"Info:  {varname:15s}  shape={str(x.shape).replace('torch.Size',''):20s}")


