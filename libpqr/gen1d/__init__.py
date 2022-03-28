import rdkit.Chem as chem

from ..aux import normalize_atom_index
from .aux import clear_stars
from .fragment import fragment_mol
from .rebuild import rebuild_mol, rebuild_mol_stages_given_order


def motifs_from_smiles(
        smi, 
        method="stochastic",
        ):
    idx, smi = smi
    if idx % 1000 == 0:
        print("%8d" % idx, smi)
    mol = chem.MolFromSmiles(smi)

    # Fragment
    frags, groups, links = fragment_mol(mol, method=method)
    if len(links) < 1: return []
    stages = rebuild_mol(frags, groups, links, "breadth_and_depth")

    # Collect
    motifs = []
    for stage in stages:
        motif = stage["end"]
        vector = stage["end_atom"]
        motif, vector = normalize_atom_index(motif, vector)
        motifs.append(chem.MolToSmiles(motif)+"_"+str(vector))

    return motifs


