import sys
import os
import multiprocessing as mp
import glob
import gzip
import functools

import numpy as np
import rdkit.Chem as chem
import torch

from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
from torch_geometric.loader.dataloader import Collater

from .arch import G3dData, G3dFeaturizer
from ..aux import Timer, normalize_atom_index
from .. import _cxx
from ..gen1d import (
    fragment_mol, 
    rebuild_mol, 
    clear_stars, 
    rebuild_mol_stages_given_order
)


def motif_to_tensors(
        motif_input,
        settings=None
):
    data = G3dData(
        motif=motif_input["mol"], 
        motif_vectors=[motif_input["vec"]],
        motif_ids=[motif_input["id"]],
        motif_index=np.zeros(
            (motif_input["mol"].GetNumAtoms(),),
            dtype=np.int32
        )
    )
    data = featurize_components(data)
    data.x = torch.zeros(0,0).float()
    return data_to_numpy(data)


def complex_to_tensors(
        complex_data, 
        settings,
        report_every=100,
        verbose=False,
):
    T = Timer()
    if complex_data["src"] == "csd": return None

    # Read structure files
    prot, lig, path = read_mols_from_input(complex_data)
    if prot.GetNumAtoms() < 1 or lig.GetNumAtoms() < 1: return None

    # Generate reconstruction sequence
    data = generate_recon_sequence(lig, prot, complex_data, settings)
    if data is None: return None

    # Generate environments and (hyper-)edges
    data_env, reindex_mask = generate_environments(
        data=data,
        depth_2d=settings.depth_2d,
        crop=settings.crop_3d,
        cut_hyper_intra=settings.cut_hyper_intra,
        cut_hyper_inter=settings.cut_hyper_inter,
        cut_hyper_edge=settings.cut_hyper_edge,
        include_environment=settings.include_environment,
        perturb_pos=settings.perturb_pos,
        perturb_steps=settings.perturb_steps,
        perturb_ampl=settings.perturb_ampl,
        error_tag=path
    )
    if data_env is None: return None

    # Featurize molecular graphs
    data_env = featurize_components(data, data_env, reindex_mask)
    data_env.path = path
    data_env.src = complex_data["src"]
    data_env.num_nodes = data_env.x.shape[0]
    data_env = data_to_numpy(data_env)
    return data_env


def featurize_components(data, data_env=None, reindex_mask=None):
    if data_env is None:
        data_env = G3dData()
    feat = G3dFeaturizer()
    if hasattr(data, "mol"):
        data_env.mol = data.mol
        data_env = feat(
            data_env, 
            selection_mask=reindex_mask
        )
    if hasattr(data, "motif"):
        data_motif = G3dData(mol=data.motif)
        data_motif = feat(data_motif)
        data_env.motif = data.motif
        data_env.motif_index = torch.from_numpy(data.motif_index).long()
        data_env.motif_x = data_motif.x
        data_env.motif_edge_index = data_motif.edge_index
        data_env.motif_edge_attr = data_motif.edge_attr
        data_env.motif_vectors = torch.tensor(data.motif_vectors).long()
        data_env.motif_ids = data.motif_ids
    if hasattr(data, "lig"):
        data_out_lig = G3dData(mol=data.lig)
        data_out_lig = feat(data_out_lig)
        data_env.lig = torch.zeros((data_out_lig.x.shape[0],)).long()
        data_env.lig_x = data_out_lig.x
        data_env.lig_edge_index = data_out_lig.edge_index
        data_env.lig_edge_attr = data_out_lig.edge_attr
        data_env.lig_center_index=torch.from_numpy(data.lig_centers).long()
        data_env.lig_flags=torch.from_numpy(data.lig_flags)
    return data_env


def generate_recon_sequence(lig, prot, complex_data, settings, verbose=False):
    if lig.GetNumAtoms() < 1: return None

    cut, vector = complex_data.get("cut", None), complex_data.get("vector", None)
    if cut is not None or vector is not None:
        stages, rest = stages_from_user_input(lig, complex_data, cut=cut, vector=vector)
    else:
        try:
            stages, rest = fragment_and_rebuild(lig)
        except:
            print("UNKERR", chem.MolToSmiles(lig))
            return None
    if len(stages) < 1: return None

    mol_glob = chem.MolFromSmiles("")
    mol_lig = chem.MolFromSmiles("")
    mol_motif = chem.MolFromSmiles("")
    flags_glob = []
    centers_glob = []
    centers_lig = []
    flags_lig = []
    motif_vectors = []
    motif_ids = []
    motif_index = []

    # Append stages
    for stageidx, stage in enumerate(stages):
        if settings.stage_final_only:
            if stageidx < len(stages) - 1:
                continue
        if settings.stage_chains_only:
            mol_motif_stage = stage.get("end")
            if "1" in chem.MolToSmiles(mol_motif_stage):
                continue
        if settings.stage_motif_size_cutoff > 0:
            mol_motif_stage = stage.get("end")
            if mol_motif_stage.GetNumAtoms() > settings.stage_motif_size_cutoff:
                continue
        if settings.stage_contacts_only:
            mol_motif_stage = stage.get("end")
            pos = mol_motif_stage.GetConformer(0).GetPositions()
            pos_prot = prot.GetConformer(0).GetPositions()
            u2 = np.sum(pos**2, axis=1)
            v2 = np.sum(pos_prot**2, axis=1)
            dmat = np.add.outer(u2, v2) - 2*pos.dot(pos_prot.T)
            sign = -1 if settings.stage_contacts_invert else +1
            if sign*np.min(dmat) > sign*settings.stage_contacts_cutoff**2:
                continue

        mol = stage.get("start")
        c = stage.get("start_atom")
        off = mol_glob.GetNumAtoms()
        flags = np.zeros((mol.GetNumAtoms(),), dtype=np.int32)
        flags.fill(2*stageidx + 2)
        flags[c] = 2*stageidx + 1
        mol_glob = chem.CombineMols(mol_glob, mol)
        flags_glob.append(flags)
        centers_glob.append(c + off)

        off_lig = mol_lig.GetNumAtoms()
        mol_lig = chem.CombineMols(mol_lig, mol)
        centers_lig.append(c + off_lig)
        flags_lig.append(flags)

        off_motif = mol_motif.GetNumAtoms() 
        motif_counter = len(motif_ids)
        mol_motif_stage = stage.get("end")
        mol_motif = chem.CombineMols(mol_motif, mol_motif_stage)
        motif_vectors.append(off_motif + stage.get("end_atom"))
        motif_index = motif_index + [ motif_counter ]*mol_motif_stage.GetNumAtoms()

        try:
            mol_motif_norm, motif_vec_norm = normalize_atom_index(mol_motif_stage, stage.get("end_atom"))
            motif_ids.append((chem.MolToSmiles(mol_motif_norm), motif_vec_norm))
        except:
            print("UNKERR <motif_norm>", chem.MolToSmiles(lig))
            return None

    if mol_glob.GetNumAtoms() < 1:
        return None

    # Append inert ligands and prot
    if settings.include_environment:
        for mol in (rest + [ prot ]):
            mol_glob = chem.CombineMols(mol_glob, mol)
            flags = np.zeros((mol.GetNumAtoms(),), dtype=np.int32)
            flags_glob.append(flags)

    centers_glob = np.array(centers_glob, dtype=np.int32)
    flags_glob = np.concatenate(flags_glob)

    centers_lig = np.array(centers_lig, dtype=np.int32)
    flags_lig = np.concatenate(flags_lig)

    # >>> import benchml as bml
    # >>> symbols = []
    # >>> pos = []
    # >>> for s in stages:
    # >>>     for atom in s["start"].GetAtoms():
    # >>>         symbols.append(atom.GetSymbol())
    # >>>     pos.append(s["start"].GetConformer().GetPositions())
    # >>> for l in (rest + [ prot ]):
    # >>>     for atom in l.GetAtoms():
    # >>>         symbols.append(atom.GetSymbol())
    # >>>     pos.append(l.GetConformer().GetPositions())
    # >>> pos = np.concatenate(pos, axis=0)
    # >>> config = bml.readwrite.ExtendedXyz(symbols=symbols, pos=pos)
    # >>> config.info["path"] = path
    # >>> bml.write('tmp.xyz', [ config ])

    # >>> import benchml as bml
    # >>> alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    # >>> pos = mol_glob.GetConformer(0).GetPositions()
    # >>> symbols = [ alph[_] for _ in flags_glob ]
    # >>> config = bml.readwrite.ExtendedXyz(symbols=symbols, pos=pos)
    # >>> config.info["path"] = path
    # >>> bml.write('tmp_flags.xyz', [ config ])

    return G3dData(
        mol=mol_glob,
        flags=flags_glob,
        centers=centers_glob,
        lig=mol_lig,
        lig_centers=centers_lig,
        lig_flags=flags_lig,
        motif=mol_motif,
        motif_index=np.array(motif_index, dtype=np.int32),
        motif_vectors=np.array(motif_vectors, dtype=np.int32),
        motif_ids=motif_ids
    )


def fragment_and_rebuild(
        mol, 
        fragment_method="stochastic",
        rebuild_method="breadth_and_depth",
):
    mols = list(chem.GetMolFrags(mol, asMols=True))
    mol = mols.pop(np.random.randint(0, len(mols)))

    frags, groups, links = fragment_mol(
        mol, method=fragment_method
    )
    if len(groups) < 2: return [], mols
    stages = rebuild_mol(
        frags, groups, links, method=rebuild_method
    )
    return stages, mols


def stages_from_user_input(lig, complex_data, cut=None, vector=None):
    if cut is not None:
        i, j = cut
        bond = lig.GetBondBetweenAtoms(i, j)
        links = [ [i, j, bond.GetBondType() ] ]
        frags = chem.FragmentOnBonds(lig, [bond.GetIdx()])
        frags = clear_stars(frags)
        groups = chem.GetMolFrags(frags)
        stages = rebuild_mol_stages_given_order(
            frags, groups, links, [0]
        )
        stages = rebuild_mol(
            frags, groups, links, stages=stages
        )
    elif vector is not None:
        start = lig
        if "motif_id" in complex_data:
            motif_smi, motif_vector = complex_data["motif_id"].split("_")
            end = chem.MolFromSmiles(motif_smi)
            end_atom = int(motif_vector)
        else:
            end = chem.MolFromSmiles("*")
            end_atom = 0
        stages = [{
            "start": start,
            "start_atom": vector,
            "end": end,
            "end_atom": end_atom
        }]
    else:
        raise ValueError("Either 'cut' of 'vector' != None required")
    return stages, []


def generate_environments(
        data, 
        depth_2d,
        cut_hyper_intra,
        cut_hyper_inter,
        cut_hyper_edge,
        include_environment=True,
        perturb_pos=True,
        perturb_steps=5,
        perturb_ampl=0.5,
        crop=True, 
        verbose=False,
        catch_errors=True,
        training=True,
        buffer_node_per_centre_hedge=1000, # <- TODO Too large!
        buffer_node_per_centre_func=20,
        error_tag="?"
):
    mol = data.mol
    pos = data.mol.GetConformer(0).GetPositions()
    flags = data.flags
    centers = data.centers

    if crop:
        selection, reindex_mask, pos  = crop_structure(
            mol=mol, 
            pos=pos, 
            flags=flags, 
            cut=cut_hyper_inter,
            depth_2d=depth_2d,
            perturb_pos=perturb_pos,
            perturb_steps=perturb_steps,
            perturb_ampl=perturb_ampl
        )
        pos = pos[selection]
        flags = flags[selection]
        centers = reindex_mask[centers]

        if mol.GetNumAtoms() < 1 or len(selection) < 1: 
            return None, reindex_mask
    else:
        reindex_mask = None
  
    # >>> import benchml as bml
    # >>> alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    # >>> symbols = [ alph[_] for _ in flags ]
    # >>> config = bml.readwrite.ExtendedXyz(symbols=symbols, pos=pos)
    # >>> config.info["path"] = data.path
    # >>> bml.write('tmp.xyz', [ config ])
    # >>> input('...')

    # Construct hyperedges
    (
        env_center_index, 
        env_node_index, 
        env_hyperedge_index, 
        env_hyperedge_attr
    ) = construct_hyperedges(
        centers=centers, 
        flags=flags, 
        pos=pos, 
        cut_min=0.5,
        cut_max_intra=cut_hyper_intra, 
        cut_max_inter=cut_hyper_inter,
        cut_max_edge=cut_hyper_edge,
        fl=0 if include_environment else 1,
        fu=0 if include_environment else -1,
        dim_hyperedge=5,
        buffer_node_per_centre=buffer_node_per_centre_hedge,
        buffer_edge_per_centre=20*buffer_node_per_centre_hedge,
        catch_errors=catch_errors,
        error_tag="hedge-"+error_tag
    )
    if env_center_index is None:
        return None, reindex_mask

    data_out = G3dData()
    data_out.flags = torch.from_numpy(flags)
    data_out.center_index = centers.long()
    data_out.env_center_index = torch.from_numpy(env_center_index).long()
    data_out.env_node_index = torch.from_numpy(env_node_index).long()
    data_out.env_hyperedge_index = torch.from_numpy(env_hyperedge_index).long()
    data_out.env_hyperedge_attr = torch.from_numpy(env_hyperedge_attr).float()

    return data_out, reindex_mask


def construct_hyperedges(
        centers, 
        flags, 
        pos, 
        cut_min,
        cut_max_intra, 
        cut_max_inter,
        cut_max_edge,
        fl, 
        fu,
        dim_hyperedge,
        buffer_node_per_centre,
        buffer_edge_per_centre,
        cut_intra=None,
        verbose=False,
        catch_errors=False,
        error_tag=""
):
    assert cut_max_inter >= cut_max_intra
    assert cut_max_edge >= cut_max_intra

    buffersize_node = buffer_node_per_centre*len(centers)
    buffersize_edge = buffer_edge_per_centre*len(centers)

    center_index = np.zeros((len(centers),), dtype=np.int32)
    node_index = np.zeros((buffersize_node,), dtype=np.int32)
    hyperedge_index = np.zeros((3, buffersize_edge), dtype=np.int32)
    hyperedge_attr = np.zeros((buffersize_edge, dim_hyperedge), dtype=np.float64)

    # Growth vectors have flag off+1, atoms of same molecule off+2
    # Func points have flag off+3, atoms of same molecule off+4
    # Background atoms have flag 0
    limits = _cxx.hyperedge_search(
        centers,
        flags,
        pos,
        center_index,
        node_index,
        hyperedge_index,
        hyperedge_attr,
        cut_min,
        cut_max_intra,
        cut_max_inter,
        cut_max_edge,
        dim_hyperedge,
        buffersize_node,
        buffersize_edge,
        fl, # flag_bound_lower
        fu  # flag_bound_upper  (lower > upper := exclude 0)
    )

    (n_nodes, n_hyperedges) = limits
    if n_nodes < 0 or n_hyperedges < 0:
        mssg = ""
        if n_nodes < 0:
            mssg += "  ERROR Buffer overflow (node): '%s'" % error_tag
        if n_hyperedges < 0:
            mssg += "  ERROR Buffer overflow (edge): '%s'" % error_tag
        if not catch_errors:
            raise IndexError(mssg)
        else:
            print(mssg)
        return (None, None, None, None)

    if verbose: print("  Nodes, hyperedges:", n_nodes, n_hyperedges, centers)

    return (
        center_index,
        node_index[0:n_nodes], 
        hyperedge_index[:,0:n_hyperedges], 
        hyperedge_attr[0:n_hyperedges]
    )


def crop_structure(
        mol, 
        pos, 
        flags,
        cut,
        depth_2d,
        perturb_pos,
        perturb_steps,
        perturb_ampl
):
    mask = np.zeros_like(flags)
    active = np.where(flags > 0)[0].astype(np.int32)

    # Apply cutoff
    _cxx.mark_region(
        active,    
        pos,
        mask,
        0.0,
        cut
    )

    # Add decoy groups
    decoys = np.where(flags < -1.5)[0]
    inactive = np.where(flags == -1)[0]
    mask[decoys] = 1
    mask[inactive] = 0

    # Get sparse adj matrix
    bonds = list(mol.GetBonds())
    edges = []
    for atom in mol.GetAtoms():
        edges.append([ atom.GetIdx(), atom.GetIdx() ])
    for bond in mol.GetBonds():
        (i,j) = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        edges.append([i, j])
        edges.append([j, i])
    edges = torch.tensor(edges).T

    # Extend selection along bonds
    mask = torch.from_numpy(mask)
    for _ in range(depth_2d+1):
        mask_in = mask[edges[0]]
        mask = scatter_add(mask_in, edges[1], dim_size=mask.shape[0])
        mask = torch.heaviside(mask, torch.tensor([0], dtype=torch.int))

    # >>> def write(pos_list):
    # >>>     with open('tmp.xyz', 'w') as f:
    # >>>         for pos in pos_list:
    # >>>             f.write('%d\n\n' % len(pos))
    # >>>             for i in range(len(pos)):
    # >>>                 f.write('C %1.4f %1.4f %1.4f\n' % tuple(pos[i]))
    # >>> timer = Timer()
    # >>> pos_list = [ pos ]
    # >>> for s in [5,4,3,2,1,0]:
    # >>>     x = perlin_noise(pos, edges, steps=s)
    # >>>     pos_x = pos + x.numpy()
    # >>>     pos_list.append(pos_x)
    # >>> write(pos_list)

    if perturb_pos:
        x = perlin_noise(
            pos=pos, 
            edges=edges, 
            steps=perturb_steps, 
            x_min=-1., 
            x_max=+1, 
            ampl=perturb_ampl
        )
        pos = pos + x.numpy()

    s = torch.where(mask > 0)[0]
    mask.fill_(-1)
    mask[s] = torch.arange(0, len(s), dtype=torch.int)
    return s, mask, pos


def perlin_noise(pos, edges, steps=5, x_min=-1., x_max=+1., ampl=0.5):
    x = torch.from_numpy(np.random.normal(0., 1., size=(3,pos.shape[0]))).float()
    t = torch.stack([ edges[1] ]*3)
    for _ in range(steps):
        x_in = x[:,edges[0]]
        x = scatter_mean(x_in, t, dim_size=x.shape[1], dim=-1) #x_in.shape[0])
    x = x.float()
    x = ampl*x/torch.std(x)
    x = torch.clamp(x, x_min, x_max)
    return x.T


data_attrs = [
    "x",
    "edge_index",
    "edge_attr",
    "flags",
    "center_index",
    "env_node_index",
    "env_center_index",
    "env_hyperedge_index",
    "env_hyperedge_attr",
    "lig",
    "lig_x",
    "lig_edge_index",
    "lig_edge_attr",
    "lig_flags",
    "lig_center_index",
    "motif_index",
    "motif_x",
    "motif_edge_index",
    "motif_edge_attr",
    "motif_vectors"
]


def data_to_numpy(
        data, 
        keys=data_attrs
):
    for k in keys:
        if hasattr(data, k):
            val = getattr(data, k)
            if val is not None:
                setattr(data, k, val.numpy())
    return data 


def data_to_torch(
        data, 
        keys=data_attrs
):
    for k in keys:
        if hasattr(data, k):
            val = getattr(data, k)
            if val is not None: 
                setattr(data, k, torch.from_numpy(val))
    return data 


def read_mols_from_input(complex_data):
    pdbfile_prot = complex_data["env_pdb"]
    if pdbfile_prot != None:
        prot = read_mol_from_pdb(pdbfile_prot)
    else:
        sdfile_prot = complex_data["env_sdf"]
        prot = read_mol_from_sdf(sdfile_prot)

    if prot.GetNumConformers() > 1:
        for i in reversed(range(1, prot.GetNumConformers())):
            prot.RemoveConformer(i)
        
    sdfile_lig = complex_data["mol_sdf"]
    lig = read_mol_from_sdf(sdfile_lig)
    path = sdfile_lig
    return prot, lig, path


def read_mol_from_sdf(sdfile, remove_hs=True):
    with chem.SDMolSupplier(sdfile, removeHs=remove_hs) as reader:
        for mol in reader:
            return mol


def read_mol_from_pdb(pdbfile):
    mol = chem.rdmolfiles.MolFromPDBFile(pdbfile)
    return mol


def get_smi_with_index_map(mol, *args, **kwargs):
    smi = chem.MolToSmiles(mol, *args, **kwargs)
    order = list(map(int, mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
    return smi, { i: ii for ii, i in enumerate(order) }


def slice_mol(
        mol,
        sel,
        add_positions=False,
        conf_idx=0
):
    index_map = { f: ff for ff, f in enumerate(sel) }
    atoms = []
    bonds = []
    for ff, f in enumerate(sel):
        atom = mol.GetAtomWithIdx(int(f))
        atoms.append([ atom.GetSymbol(), atom.GetFormalCharge(), atom.GetNumExplicitHs() ])
        for nb in atom.GetNeighbors():
            g = nb.GetIdx()
            gg = index_map[nb.GetIdx()]
            if gg > ff: continue
            bond = mol.GetBondBetweenAtoms(int(f), g)
            bonds.append((ff, gg, bond.GetBondType()))
    fmol = chem.RWMol()
    for idx, atom in enumerate(atoms):
        a = chem.Atom(atom[0])
        a.SetFormalCharge(atom[1])
        a.SetNumExplicitHs(atom[2])
        fmol.AddAtom(a)
    for bond in bonds:
        fmol.AddBond(*bond)
    fmol = fmol.GetMol()
    flag = chem.SanitizeMol(fmol, catchErrors=True)
    if flag != 0:
        assert False

    # Add conformer?
    if add_positions:
        pos = mol.GetConformer(conf_idx).GetPositions()
        pos = pos[sel]
        conf = chem.Conformer()
        for ff, f in enumerate(sel):
            conf.SetAtomPosition(ff, pos[ff].tolist())
        fmol.AddConformer(conf)
    return fmol


