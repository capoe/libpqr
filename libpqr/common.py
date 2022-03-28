import torch
import numpy as np
import rdkit.Chem as Chem


class Featurizer(object):
    def __init__(self, 
            symbols="C,N,O,S,H,F,Cl,Br,I,B,P,Si".split(","),
            max_degree=6,
            connect_full=True,
            edge_full_cut_high=8,
            edge_full_clamp_high=7,
            chiral=False,
            add_hs=False,
            ignore_missing=False
    ):
        self.symbols = symbols
        self.max_degree = max_degree
        self.hybridizations = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED
        ]
        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]
        self.chiral = chiral
        self.connect_full = connect_full
        self.edge_full_cut_high = edge_full_cut_high
        self.edge_full_clamp_high = edge_full_clamp_high
        self.add_hs = add_hs
        self.ignore_missing = ignore_missing

    def dimNode(self):
        return 27 + (self.max_degree+1) + (3 if self.chiral else 0)

    def dimEdge(self):
        return 6 + (4 if self.chiral else 0)
        
    def dimEdgeFull(self):
        return self.edge_full_clamp

    def featurizeNodes(self, mol, data, selection_mask=None):
        xs = []
        if selection_mask is None:
            for atom in mol.GetAtoms():
                x = featurize_atom(self, atom)
                xs.append(x)
        else:
            for atom in mol.GetAtoms():
                if selection_mask[atom.GetIdx()] < 0: continue
                x = featurize_atom(self, atom)
                xs.append(x)
        data.x = torch.stack(xs, dim=0)
        return data

    def featurizeEdges(self, mol, data, selection_mask=None):
        # Edges
        edge_index = []
        edge_attr = []
        if selection_mask is None:
            for bond in mol.GetBonds():
                edge_index_bond, edge_attr_bond = featurize_bond(self, bond)
                edge_index.extend(edge_index_bond)
                edge_attr.extend(edge_attr_bond)
        else:
            for bond in mol.GetBonds():
                (i,j) = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                if selection_mask[i] < 0 or selection_mask[j] < 0: continue
                edge_index_bond, edge_attr_bond = featurize_bond(self, bond)
                edge_index.extend(edge_index_bond)
                edge_attr.extend(edge_attr_bond)
        if len(edge_attr) == 0:
            data.edge_index = \
                torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = \
                torch.zeros((0, self.dimEdge()), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_index).t().contiguous()
            data.edge_attr = torch.stack(edge_attr, dim=0)
            if selection_mask is not None:
                data.edge_index = torch.stack([
                    selection_mask[data.edge_index[0]],
                    selection_mask[data.edge_index[1]]
                ]).long()
        return data

    def featurizeEdgesFull(self, mol, data, selection_mask=None):
        if self.connect_full:
            if selection_mask is not None: 
                raise NotImplementedError("connect_full with selection != none")
            edge_index_full, edge_dist_full, edge_attr_full = \
                generate_pairwise_edges(
                    mol=mol, 
                    cut_high=self.edge_full_cut_high, 
                    clamp_high=self.edge_full_clamp_high
            )
            if len(edge_index_full) == 0:
                data.edge_index_full = \
                    torch.zeros((2, 0), dtype=torch.long)
                data.edge_dist_full = \
                    torch.zeros((0, 1), dtype=torch.int)
                data.edge_attr_full = \
                    torch.zeros((0, self.edge_full_cut_high-1))
            else:
                data.edge_index_full = \
                    torch.tensor(edge_index_full, dtype=torch.long)
                data.edge_dist_full = edge_dist_full
                data.edge_attr_full = edge_attr_full
        return data

    def __call__(self, data, selection_mask=None):
        if hasattr(data, "mol"): mol = data.mol
        else: mol = Chem.MolFromSmiles(data.smiles)
        if hasattr(self, "add_hs") and self.add_hs: mol = Chem.AddHs(mol)
        self.featurizeNodes(mol, data, selection_mask)
        self.featurizeEdges(mol, data, selection_mask)
        self.featurizeEdgesFull(mol, data, selection_mask)
        return data


def featurize_atom(feat, atom):
    symbol = [0.] * len(feat.symbols)
    try:
        symbol[feat.symbols.index(atom.GetSymbol())] = 1.
    except ValueError as err:
        if not feat.ignore_missing:
            raise ValueError("Unknown element: '%s'" % atom.GetSymbol())
    degree = [0.] * (feat.max_degree+1)
    if atom.GetDegree() <= feat.max_degree:
        degree[atom.GetDegree()] = 1.
    formal_charge = atom.GetFormalCharge()
    radical_electrons = atom.GetNumRadicalElectrons()
    hybridization = [0.] * len(feat.hybridizations)
    hybridization[feat.hybridizations.index(
        atom.GetHybridization())] = 1.
    aromaticity = 1. if atom.GetIsAromatic() else 0.
    hydrogens = [0.] * 5
    if atom.GetTotalNumHs() < 5:
        hydrogens[atom.GetTotalNumHs()] = 1.
    x = symbol + degree + [formal_charge] + \
        [radical_electrons] + hybridization + \
        [aromaticity] + hydrogens
    if feat.chiral:
        chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
        chirality_type = [0.] * 2
        if atom.HasProp('_CIPCode'):
            chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
        x = x + [ chirality ]+ chirality_type
    x = torch.tensor(x)
    return x


def featurize_bond(feat, bond):
    edge_index_bond = [
        [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
        [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    ]
    bond_type = bond.GetBondType()
    single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
    double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
    triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
    aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
    conjugation = 1. if bond.GetIsConjugated() else 0.
    ring = 1. if bond.IsInRing() else 0.
    x = [single, double, triple, aromatic, conjugation, ring]
    if feat.chiral:
        stereo = [0.] * 4
        stereo[feat.stereos.index(bond.GetStereo())] = 1.
        x = x + stereo
    x = torch.tensor(x)
    edge_attr_bond = [ x, x ]
    return edge_index_bond, edge_attr_bond


def generate_pairwise_edges(
        mol, 
        cut_high=5,
        clamp_high=4,
):
    dmat = (Chem.GetDistanceMatrix(mol) + 0.5).astype('int')
    edges = []
    edge_attr = []
    mask1 = np.heaviside(dmat - 0.5, 0.) # skip self-edges
    mask2 = np.heaviside(cut_high - dmat, 0.)
    mask = mask1*mask2
    ij = np.where(mask > 0.5)
    dij = torch.from_numpy(dmat[(ij[0], ij[1])])
    edge_attr = torch.zeros((len(ij[0]), clamp_high), dtype=torch.float)
    edge_attr[(torch.arange(len(ij[0])), torch.clamp(dij, 0, clamp_high)-1)] = 1.
    return [ ij[0].tolist(), ij[1].tolist() ], dij, edge_attr


