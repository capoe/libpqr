import numpy as np
import rdkit.Chem as chem

from ..aux import stochastic_argmax, stochastic_argsort
from .aux import *


HALO = set([ "F", "Cl", "Br", "I" ])
NOS = set([ "N", "O", "S" ])
PRIORITY = set([ "N", "O", "S", "B", "P", "Si" ])


def fragment_mol(mol, method="stochastic", **kwargs):
    if method == "simple":
        return fragment_mol_simple(mol, **kwargs)
    elif method == "prior":
        return fragment_mol_prior(mol, **kwargs)
    elif method == "stochastic":
        return fragment_mol_stochastic(mol, **kwargs)
    else:
        raise ValueError(method)


def fragment_mol_simple(mol, **kwargs):
    frag_bonds, links = find_fragmentation_bonds(mol)
    if len(frag_bonds) > 0:
        frags = chem.FragmentOnBonds(mol, frag_bonds)
        frags = clear_stars(frags)
    else:
        frags = mol
    if frags is None:
        return None, None, None
    groups = chem.GetMolFrags(frags)
    return frags, groups, links


def fragment_mol_stochastic(mol, max_depth=2, **kwargs):
    mol_split, cuts, links = split_rings_vs_chains(mol)
    mol_split, cuts_add, links_add = fragment_mol_stochastic_recursive(
        mol_split, max_depth=max_depth
    )
    links.extend(links_add)
    groups = chem.GetMolFrags(mol_split)
    return mol_split, groups, links


def fragment_mol_stochastic_recursive(mol, max_depth):
    todo = set()
    done = set()
    for atom in mol.GetAtoms():
        add = True
        if not atom.IsInRing():
            for nb in atom.GetNeighbors():
                if nb.IsInRing():
                    add = False
                    break
        else:
            add = False
        if add: todo.add(atom.GetIdx())

    # Atomic weights for seed sampling
    weights = []
    for atom in mol.GetAtoms():
        w = atom.GetDegree() + atom.GetExplicitValence()
        weights.append(w)
    weights = np.array(weights)

    # Generate paths
    paths = []
    while len(todo):
        start_idcs = list(todo)
        start_weights = weights[start_idcs]
        order = stochastic_argsort(start_weights.tolist())
        start_idx = start_idcs[order[0]]
        depth = np.random.randint(0, max_depth+1)
        isotropic = np.random.randint(0, 2, size=(depth,))
        path = walk(mol, start_idx, depth, isotropic, exclusions=done)
        path = set(path)
        done = done.union(path)
        todo = todo.difference(path)
        paths.append(path) 

    # Find bonds on which to fragment
    cuts = set()
    for path in paths:
        for r in path:
            a = mol.GetAtomWithIdx(r)
            for nb in a.GetNeighbors():
                if not nb.GetIdx() in path:
                    bond = mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx())
                    cuts.add(bond.GetIdx())

    # Fragment mol
    cuts = list(cuts)
    frag_bonds = [ mol.GetBondWithIdx(c) for c in cuts ]
    links = [ (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) for bond in frag_bonds ]
    if len(cuts):
        frags = chem.FragmentOnBonds(mol, cuts)
        frags = clear_stars(frags)
    else:
        frags = mol
    return frags, cuts, links


def fragment_mol_prior(mol):
    # Rings vs chains
    mol_split, cuts, links = split_rings_vs_chains(mol)

    # Priority atom bias
    bias = [ 0 for atom in mol.GetAtoms() ]
    props = [ { "degree": atom.GetDegree() } for atom in mol.GetAtoms() ]
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == "C":
            has_o = False
            has_n = False
            for nb in atom.GetNeighbors():
                if nb.GetSymbol() == "O": has_o = True
                if nb.GetSymbol() == "N": has_n = True
            props[idx]["cno"] = has_o and has_n
        else:
            props[idx]["cno"] = False
        if atom.GetSymbol() in PRIORITY: bias[atom.GetIdx()] += 1
        if atom.GetSymbol() in NOS: 
            if atom.GetDegree() < 2:
                bias[atom.GetIdx()] += 5

    # Backtrack on chains
    mol_split, cuts, links = fragment_mol_recursive(mol_split, props, bias, cuts, links)
    if mol_split is None:
        return None, None, None

    groups = chem.GetMolFrags(mol_split)
    return mol_split, groups, links


def fragment_mol_recursive(mol, props, bias, cuts=None, links=None):  
    if cuts is None:
        cuts = []
        links = []
    frags = chem.GetMolFrags(mol)
    frag_bonds = []
    for frag in frags:
        atoms = [ mol.GetAtomWithIdx(f) for f in frag ]
        if np.any(list(map(lambda a: a.IsInRing(), atoms))): continue
        if len(frag) <= 2: continue
        chain_bonds = list(set([ b.GetIdx() \
            for f in frag \
                for b in mol.GetAtomWithIdx(f).GetBonds() ]))

        # Assign priorities
        priors = []
        cc_bonds = []
        for bondidx in chain_bonds:
            bond = mol.GetBondWithIdx(bondidx)
            a = bond.GetBeginAtom()
            b = bond.GetEndAtom()
            va = props[a.GetIdx()]["degree"]
            vb = props[b.GetIdx()]["degree"]
            ha = a.GetSymbol() in PRIORITY
            hb = b.GetSymbol() in PRIORITY
            xa = a.GetSymbol() in HALO
            xb = b.GetSymbol() in HALO
            prior = 0
            if xa or xb:
                prior += 10
            if ha or hb:
                prior += 5
            if ha and hb and bond.GetBondTypeAsDouble() > 1.1:
                prior += 10
            if (ha or hb) and (props[a.GetIdx()]["cno"] or props[b.GetIdx()]["cno"]):
                prior += 10
            prior += np.abs(va - vb)
            prior += bias[a.GetIdx()]
            prior += bias[b.GetIdx()]
            priors.append(prior)
            if a.GetSymbol() == b.GetSymbol() == "C":
                cc_bonds.append(bondidx)

        # Fragment
        priors = np.array(priors)
        if len(frag) > 4:
            pmin = min(priors)
            cuts_this = np.where(priors == pmin)[0]
        elif len(frag) > 2:
            pmin = 0
            cuts_this = np.where(priors == pmin)[0]
        else:
            cuts_this = []
        for c in cuts_this:
            frag_bonds.append(mol.GetBondWithIdx(chain_bonds[c]))
   
    if len(frag_bonds):
        cuts_this = [ bond.GetIdx() for bond in frag_bonds ]
        links_this = [ (
            bond.GetBeginAtomIdx(), 
            bond.GetEndAtomIdx(), 
            bond.GetBondType()) for bond in frag_bonds 
        ]
        mol = chem.FragmentOnBonds(mol, cuts_this)
        mol = clear_stars(mol)
        cuts = cuts + cuts_this
        links = links + links_this
        return fragment_mol_recursive(mol, props, bias, cuts=cuts, links=links)
    else:
        return mol, cuts, links


def map_links_on_groups(frags, groups, links):
    idx_to_group = { gg: g for g in range(len(groups)) for gg in groups[g] }
    idx_to_group_atom = { g: {} for g in range(len(groups)) }
    for gidx, group in enumerate(groups):
        for gg, g in enumerate(group):
            idx_to_group_atom[gidx][g] = gg
    links_out = []
    for link in links:
        g0 = idx_to_group[link[0]]
        g1 = idx_to_group[link[1]]
        a0 = idx_to_group_atom[g0][link[0]]
        a1 = idx_to_group_atom[g1][link[1]]
        links_out.append([
            (g0, a0), (g1, a1), link[2]
        ])
    return links_out, idx_to_group, idx_to_group_atom


def find_fragmentation_bonds(mol, cut_double_o=True):
    degree = [ atom.GetDegree() for atom in mol.GetAtoms() ]
    absorbed = [ False for atom in mol.GetAtoms() ]
    frag_bonds = []
    keep_bonds = []
    for bond in mol.GetBonds():
        i, j = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        in_ring = bond.IsInRing()
        if in_ring:
            continue
        degree_l = bond.GetBeginAtom().GetDegree()
        degree_r = bond.GetEndAtom().GetDegree()
        bond_order = bond.GetBondTypeAsDouble()
        cut = False
        absorb = False

        # Ring-ring or ring-linear bond?
        ring_l = bond.GetBeginAtom().IsInRing()
        ring_r = bond.GetEndAtom().IsInRing()
        if ring_l and ring_r:
            cut = True
        elif ring_l:
            symbol_r = bond.GetEndAtom().GetSymbol()
            arom_l = bond.GetBeginAtom().GetIsAromatic()
            if arom_l and degree_r == 1 and bond_order > 1.1 and symbol_r == "O":
                cut = cut_double_o
            else:
                cut = True
        elif ring_r:
            symbol_l = bond.GetBeginAtom().GetSymbol()
            arom_r = bond.GetEndAtom().GetIsAromatic()
            if arom_r and degree_l == 1 and bond_order > 1.1 and symbol_l == "O":
                cut = cut_double_o
            else:
                cut = True
        # Simple linear chain?
        elif degree_l < 3 and degree_r < 3 and bond_order < 1.1:
            cut = True
        # Double or triple bond?
        elif degree_l < 3 and degree_r < 3 and bond_order > 1.1:
            absorbed[i] = True
            absorbed[j] = True
            absorb = True
        if cut:
            degree[i] -= 1
            degree[j] -= 1
            frag_bonds.append(bond)
        elif not absorb:
            keep_bonds.append(bond)

    # Cut bonds adjacent to double/triple bonds
    keep_bonds_out = []
    for bond in keep_bonds:
        i, j = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        ai, aj = (absorbed[i], absorbed[j])
        cut = False
        if ai or aj:
            cut = True
        if cut:
            degree[i] -= 1
            degree[j] -= 1
            frag_bonds.append(bond)
        else:
            keep_bonds_out.append(bond)

    # Cut bonds linking high-degree atoms
    for bond in keep_bonds_out:
        i, j = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        cut = False
        if degree[i] > 1 and degree[j] > 1:
            cut = True
        if cut:
            degree[i] -= 1
            degree[j] -= 1
            frag_bonds.append(bond)

    bond_idcs = [ bond.GetIdx() for bond in frag_bonds ]
    links = [ (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) for bond in frag_bonds ]
    return bond_idcs, links


