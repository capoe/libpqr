import numpy as np
import rdkit.Chem as chem

from ..aux import stochastic_argmax, stochastic_argsort


def walk(mol, start_idx, depth, isotropic, exclusions, path=None, step=0):
    if path is None:
        path = []
        path.append(start_idx)
    if step >= depth:
        return path
    add = []
    for nb in mol.GetAtomWithIdx(start_idx).GetNeighbors():
        if nb.GetIdx() in exclusions: continue
        add.append(nb.GetIdx())
    if len(add) < 1:
        return path
    start_idx = add[np.random.randint(0, len(add))]
    if isotropic[step]:
        path.extend(add)
    else:
        path.extend([ start_idx ])
    for r in path: exclusions.add(r)
    return walk(mol, start_idx, depth, isotropic, exclusions, path=path, step=step+1)


def split_rings_vs_chains(mol, cut_double_o=False):
    frag_bonds = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        bond_order = bond.GetBondTypeAsDouble()

        if a.IsInRing() and b.IsInRing():
            frag_bonds.append(bond) 
            continue

        # Exempt ring =O bonds
        if (a.IsInRing() and not b.IsInRing()) or (not a.IsInRing() and b.IsInRing()):
            if bond_order > 1.1 and (a.GetSymbol() == "O" or b.GetSymbol() == "O"):
                if not cut_double_o:
                    if a.GetIsAromatic() or b.GetIsAromatic(): 
                        continue
                    if len(set([ a.GetSymbol(), b.GetSymbol() ]).intersection({ "S", "P" })):
                        continue
            frag_bonds.append(bond)

    cuts = [ bond.GetIdx() for bond in frag_bonds ]
    links = [ (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) for bond in frag_bonds ]
    if len(cuts):
        frags = chem.FragmentOnBonds(mol, cuts)
        frags = clear_stars(frags)
    else:
        frags = mol
    return frags, cuts, links


def get_chains(mol, exclusions=None):
    if exclusions is None: exclusions = set()
    atoms = [ atom for atom in mol.GetAtoms() if (not atom.IsInRing() and not atom.GetIdx() in exclusions) ]

    # Adjacency matrix with non-ring bonds only
    adj = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))
    np.fill_diagonal(adj, 1)
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.IsInRing() or b.IsInRing(): continue
        adj[a.GetIdx(),b.GetIdx()] = 1
        adj[b.GetIdx(),a.GetIdx()] = 1

    # Backtrack chains
    unassigned = set([ a.GetIdx() for a in atoms ])
    assigned = set()
    chains = []
    while len(assigned) < len(atoms):
        v = np.zeros((mol.GetNumAtoms(),))
        i = list(unassigned)[0]
        v[i] = 1
        while True:
            v_out = np.heaviside(adj.dot(v), 0.0)
            if np.max(v_out - v) < 0.5:
                break
            v = v_out
        chain = np.where(v > 0.5)[0]
        chains.append(chain.tolist())
        for c in chain:
            unassigned.remove(c)
            assigned.add(c)

    atom_to_chain = {}
    for cidx, chain in enumerate(chains):
        for c in chain:
            atom_to_chain[c] = cidx
    return chains, atom_to_chain


def step(adj, start, todo, path):
    options = np.where(adj[start] > -0.5)[0]
    if len(options) < 1:
        if len(todo) < 1:
            return path
        else:
            path = step(adj, todo.pop(0), todo, path)
    else:
        np.random.shuffle(options)
        path.append(adj[start, options[0]])
        adj[start, options[0]] = -1
        adj[options[0], start] = -1
        todo.append(options[0])
        path = step(adj, start=start, todo=todo, path=path)
    return path


def step_breadth_and_depth(adj, seeds, path):
    seed_idx = np.random.randint(0, len(seeds))
    start = seeds[seed_idx]
    options = np.where(adj[start] > -0.5)[0]
    if len(options) < 1:
        seeds.pop(seed_idx)
        if len(seeds) < 1:
            return path
        else:
            path = step_breadth_and_depth(adj, seeds=seeds, path=path)
    else:
        np.random.shuffle(options)
        path.append(adj[start, options[0]])
        adj[start, options[0]] = -1
        adj[options[0], start] = -1
        seeds.append(options[0])
        path = step_breadth_and_depth(adj, seeds=seeds, path=path)
    return path


def clear_stars(frags, update_hs=True, sanitize=True):
    mol = chem.RWMol(frags)
    rm_stars = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            rm_stars.append(atom.GetIdx())
            if update_hs:
                nbs = list(atom.GetNeighbors())
                assert len(nbs) == 1
                nb = nbs.pop(0)
                bond = list(atom.GetBonds()).pop(0)
                nb.SetNumExplicitHs(nb.GetNumExplicitHs() + \
                    int(bond.GetBondTypeAsDouble()+0.1))
    rm_stars = rm_stars[::-1]
    for i in rm_stars:
        mol.RemoveAtom(i)
    mol = mol.GetMol()
    flag = chem.SanitizeMol(mol, catchErrors=True)
    if flag != 0:
        return None
    return mol


