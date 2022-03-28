import functools 

import torch
import numpy as np
import rdkit.Chem as chem

from torch_geometric.loader.dataloader import Collater

from .arch import VlbData, VlbFeaturizer
from ..gen1d import fragment_mol, rebuild_mol
from ..aux import normalize_atom_index


def mol_to_tensors(mol_or_smi, settings, training, verbose=False):
    if isinstance(mol_or_smi, (str,)):
        mol = chem.MolFromSmiles(mol_or_smi)
    else:
        mol = mol_or_smi
    feat = VlbFeaturizer()
    data = VlbData()

    if training:  
        data = mol_to_tensors_fragment(data, mol, settings, verbose=verbose)
        if data is None: return None
        data.atom_vector = torch.zeros(0,).long()
    else:
        data.mol = mol
        data.atom_vector = torch.zeros(0,).long()
        data.atom_vector_label = torch.zeros(0,1).float()
        data.pair_link_index = torch.zeros(2,0).long()
        data.pair_link_type = torch.zeros(0,3).float()

    data = feat(data)
    # >>> smi = chem.MolToSmiles(data.mol)
    # >>> atom_order = data.mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")
    # >>> reindex = torch.tensor(
    # >>>         list(map(int, atom_order))
    # >>>     ).long()
    # >>> order = torch.zeros(len(reindex)).long()
    # >>> order[reindex] = torch.arange(len(reindex))
    return data_to_numpy(data)


def collate(datalist):
    collater = Collater(follow_batch=[], exclude_keys=[])
    return collater(datalist)


def mol_to_tensors_fragment(
        data, 
        mol, 
        settings, 
        verbose=False, 
        fragment_method="stochastic",
        rebuild_method="breadth_and_depth"
):

    # Fragment input mol
    frags, groups, links = fragment_mol(mol, method=fragment_method)
    if len(groups) < 2: return None
    stages = rebuild_mol(frags, groups, links, method=rebuild_method)

    # Collate
    mol_glob = chem.MolFromSmiles("")
    motif_glob = chem.MolFromSmiles("")
    motif_ids = []
    functized_glob = []
    pairs_glob = []
    pairs_bonds_glob = []

    for stage in stages:
        start = stage["start"]
        end = stage["end"]

        offset_mol = mol_glob.GetNumAtoms()
        mol_glob = chem.CombineMols(mol_glob, start)
        start_idx = stage["start_atom"] + offset_mol

        offset_motif = mol_glob.GetNumAtoms()
        mol_glob = chem.CombineMols(mol_glob, end)
        end_idx = stage["end_atom"] + offset_motif

        functized_glob.extend(stage["start_func"])
        functized_glob.extend(stage["end_func"])

        pairs_glob.append([ start_idx, end_idx ])
        pairs_bonds_glob.append(bondtype_to_onehot(stage["bond"]))

        end_norm, end_vec_norm = normalize_atom_index(end, stage["end_atom"])
        motif_ids.append((chem.MolToSmiles(end_norm), end_vec_norm))
    
    # Data attributes
    data.mol = mol_glob
    data.atom_vector_label = torch.tensor(functized_glob).clamp(0,1).float().view(-1,1)
    data.pair_link_index = torch.tensor(pairs_glob).T.long()
    data.pair_link_type = torch.tensor(pairs_bonds_glob).float()
    data.motif_ids = motif_ids
    return data


data_attrs = [
    "x",
    "edge_index",
    "edge_attr",
    "atom_vector",
    "atom_vector_label",
    "pair_link_index",
    "pair_link_type"
]


def data_to_numpy(
        data, 
        keys=data_attrs
):
    for k in keys:
        if hasattr(data, k):
            setattr(data, k, getattr(data, k).numpy())
    return data 


def data_to_torch(
        data, 
        keys=data_attrs
):
    for k in keys:
        if hasattr(data, k):
            setattr(data, k, torch.from_numpy(getattr(data, k)))
    return data 


def bondtype_to_onehot(bondtype):
    b = [0,0,0]
    if bondtype == chem.rdchem.BondType.SINGLE:
        b[0] = 1
    elif bondtype == chem.rdchem.BondType.DOUBLE:
        b[1] = 1
    elif bondtype == chem.rdchem.BondType.TRIPLE:
        b[2] = 1
    else:
        raise ValueError(str(bondtype))
    return b


