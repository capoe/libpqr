import numpy as np
import rdkit.Chem as chem

from ..aux import stochastic_argmax, stochastic_argsort
from .aux import *
from .fragment import map_links_on_groups


def rebuild_mol_stages(mol, groups, links, method):
    # Functionalization indicators
    functized = np.zeros((mol.GetNumAtoms(),), dtype=np.int32)
    for link in links:
        functized[link[0]] += 1
        functized[link[1]] += 1
    # Order and rebuild
    order = rebuild_mol_order(mol, groups, links, method=method)
    records_out = []
    records_out.append([mol, links[order[0]], np.copy(functized)])
    mol = chem.RWMol(mol)
    for rank, idx in enumerate(order):
        link = links[idx]
        mol.AddBond(link[0], link[1], link[2])
        # Decrement func indicators
        functized[link[0]] -= 1
        functized[link[1]] -= 1
        # Desaturate
        a0 = mol.GetAtomWithIdx(link[0])
        a1 = mol.GetAtomWithIdx(link[1])
        bo = int(float(link[2])+0.1)
        dh0 = max(0, a0.GetNumExplicitHs() - bo)
        dh1 = max(0, a1.GetNumExplicitHs() - bo)
        a0.SetNumExplicitHs(dh0)
        a1.SetNumExplicitHs(dh1)
        records_out.append([ 
            mol.GetMol(), 
            links[order[rank+1]] if rank < len(links) -1 else None,
            np.copy(functized)
        ])
    return records_out


def rebuild_mol_order(mol, groups, links, method):
    idx_to_group = { gg: g for g in range(len(groups)) for gg in groups[g] }
    if method == "distributed":
        link_weights = [ len(groups[idx_to_group[l[0]]]) \
            + len(groups[idx_to_group[l[1]]]) for l in links ]
        order = stochastic_argsort(link_weights)
    elif method == "sequential":
        adj = np.zeros((len(groups), len(groups)), dtype=int)
        adj.fill(-1)
        for link_idx, link in enumerate(links):
            i = idx_to_group[link[0]]
            j = idx_to_group[link[1]]
            adj[i,j] = adj[j,i] = link_idx
        weights = [ len(g) for g in groups ]
        s = stochastic_argmax(weights)
        order = step(adj, [s], [], [])
    elif method == "breadth_and_depth":
        adj = np.zeros((len(groups), len(groups)), dtype=int)
        adj.fill(-1)
        for link_idx, link in enumerate(links):
            i = idx_to_group[link[0]]
            j = idx_to_group[link[1]]
            adj[i,j] = adj[j,i] = link_idx
        weights = [ len(g) for g in groups ]
        s = stochastic_argmax(weights)
        order = step_breadth_and_depth(adj, [s], [])
    else:
        raise ValueError(method)
    return order


def rebuild_mol_stages_given_order(mol, groups, links, order):
    functized = np.zeros((mol.GetNumAtoms(),), dtype=np.int32)
    for link in links:
        functized[link[0]] += 1
        functized[link[1]] += 1
    records_out = []
    records_out.append([mol, links[order[0]], np.copy(functized)])
    if len(order) < 2:
        return records_out
    mol = chem.RWMol(mol)
    for rank, idx in enumerate(order):
        link = links[idx]
        mol.AddBond(link[0], link[1], link[2])
        # Decrement func indicators
        functized[link[0]] -= 1
        functized[link[1]] -= 1
        # Desaturate
        a0 = mol.GetAtomWithIdx(link[0])
        a1 = mol.GetAtomWithIdx(link[1])
        bo = int(float(link[2])+0.1)
        dh0 = max(0, a0.GetNumExplicitHs() - bo)
        dh1 = max(0, a1.GetNumExplicitHs() - bo)
        a0.SetNumExplicitHs(dh0)
        a1.SetNumExplicitHs(dh1)
        records_out.append([ 
            mol.GetMol(), 
            links[order[rank+1]] if rank < len(links) -1 else None,
            np.copy(functized)
        ])
    return records_out


def rebuild_mol(frags, groups, links, method=None, stages=None):
    group_links, idx_to_group, idx_to_group_atom = \
        map_links_on_groups(frags, groups, links)
    if stages is None:
        assert method is not None
        stages = rebuild_mol_stages(
            frags, groups, links, method=method) 

    stages_out = []
    joined_groups = set()
    joined_atoms = set()
    for stage_idx, stage in enumerate(stages):
        (mol, next_link, functized) = stage
        if next_link is None: continue

        # Map atom idcs onto fragments
        frags_idcs = chem.GetMolFrags(mol)
        frags_mols = chem.GetMolFrags(mol, asMols=True)
        idx_to_frag = {}
        idx_to_frag_idx = {}
        for f, idcs in enumerate(frags_idcs):
            idx_to_frag_idx[f] = {}
            for ii, i in enumerate(idcs):
                idx_to_frag[i] = f
                idx_to_frag_idx[f][i] = ii

        # Find start and end fragments
        atom_idx_link_start = next_link[0]
        atom_idx_link_end = next_link[1]
        group_start = idx_to_group[next_link[0]]
        group_end = idx_to_group[next_link[1]]
        if stage_idx > 0 and group_start not in joined_groups:
            group_start, group_end = (group_end, group_start)
            atom_idx_link_start, atom_idx_link_end = (atom_idx_link_end, atom_idx_link_start)

        joined_groups.add(group_start)
        joined_groups.add(group_end)
        joined_atoms = joined_atoms.union(groups[group_start])

        # Find start and end atom idcs
        idcs_start = groups[group_start]
        idcs_end = groups[group_end]
        frag_idx_start = idx_to_frag[idcs_start[0]]
        frag_idx_end = idx_to_frag[idcs_end[0]]
        frag_start = frags_mols[frag_idx_start]
        frag_end = frags_mols[frag_idx_end]
        atom_idx_link_start = idx_to_frag_idx[frag_idx_start][atom_idx_link_start]
        atom_idx_link_end = idx_to_frag_idx[frag_idx_end][atom_idx_link_end]

        # Functionalizations
        functized_start = []
        functized_end = []
        for g in frags_idcs[frag_idx_start]:
            functized_start.append(functized[g])
        for g in frags_idcs[frag_idx_end]:
            functized_end.append(functized[g])

        stages_out.append({
            "start": frag_start, 
            "end": frag_end, 
            "start_group": list(joined_atoms),
            "end_group": list(groups[group_end]),
            "start_func": functized_start,
            "end_func": functized_end,
            "start_atom": atom_idx_link_start, 
            "end_atom": atom_idx_link_end, 
            "bond": next_link[2] 
        })

    return stages_out


