import json
import gzip

import torch
import numpy as np
import rdkit.Chem as chem

from torch_geometric.loader.dataloader import Collater

from .arch import VlbData, VlbFeaturizer


class Baseline:
    def __init__(self, model, motifs, device):
        if isinstance(motifs, (str,)):
            with gzip.open(motifs, 'rt') as f:
                motifs = json.load(f)
        if isinstance(model, (str,)):
            model = torch.load(model, map_location=device).eval()
        self.model = model
        self.device = device
        self.motifs = []
        self.id_to_motif = {}
        self.weights = torch.tensor([])
        self.weights_cum = torch.tensor([])
        self.vectors = torch.tensor([])
        self.xa_linker, self.xb_linker = (None, None)

        # Initialize
        self.load(motifs)
        if self.model is not None:
            self.embed()

    def to(self, device):
        pass

    @torch.no_grad()
    def embed(self):
        print("Embed motiflib ...")
        feat = VlbFeaturizer()
        datalist = []
        for m in self.motifs:
            data = VlbData(mol=m["mol"])
            data = feat(data)
            data.atom_vector = torch.tensor([ m["vec"] ]).long()
            datalist.append(data)
        collate = Collater([], [])
        data = collate(datalist).to(self.device)
        self.xa_linker, self.xb_linker = self.model.linker_model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )
        self.xa_linker = self.xa_linker[data.atom_vector]
        self.xb_linker = self.xb_linker[data.atom_vector]

    @torch.no_grad()
    def project(self, xa, xb):
        linker_mat = self.model.linker_model.final_dense(
            xa, xb, self.xa_linker, self.xb_linker
        )
        return linker_mat

    def __str__(self):
        return f"Baseline: Motiflib with {len(self.motifs)} motifs"

    def load(self, motifs):
        self.motifs = []
        self.id_to_motif = {}
        vectors = []
        freqs = []
        for m in motifs:
            mol = chem.MolFromSmiles(m["smi"])
            vectors.append(m["vec"])
            freqs.append(m["freq"])
            motif_data = {
                "mol": mol,
                "idx": len(self.motifs),
                "id": (m["smi"], m["vec"]),
                "smi": m["smi"],
                "type": "ring" if "1" in m["smi"] else "chain",
                "size": mol.GetNumAtoms(),
                "freq": m["freq"],
                "vec": m["vec"]
            }
            self.motifs.append(motif_data)
            self.id_to_motif[motif_data["id"]] = motif_data
        self.weights = torch.tensor(freqs).to(self.device)
        self.weights = 1.*self.weights / torch.sum(self.weights)
        self.weights_cum = self.weights.cumsum(dim=-1)
        self.vectors = torch.tensor(vectors)
       
    def sample(self, n, type):
        if type == "0d":
            sel = np.random.randint(0, len(self.motifs), size=(n,))
            probs = 1.*torch.ones((len(sel),)) / len(self.motifs)
        elif type == "1d":
            u = np.random.uniform(0, 1, size=(n,)) 
            u = torch.from_numpy(u).to(self.device)
            sel = torch.searchsorted(self.weights_cum, u)
            probs = torch.tensor([ self.weights[s] for s in sel ])
        else:
            raise ValueError(type)
        return [ self.motifs[_] for _ in sel ], probs

    def lookup(self, ids):
        motifs_data = [ self.id_to_motif[s] if s in self.id_to_motif else None for s in ids ]
        matched = [ i for i in range(len(motifs_data)) \
            if motifs_data[i] != None ]
        motifs_data = list(filter(lambda i: i is not None, motifs_data))
        return motifs_data, matched


def select_weighted_2d(scores, n):
    scores = scores.cumsum(dim=-1)
    scores = (scores.T/scores[:,-1]).T
    u = torch.from_numpy(
            np.random.uniform(size=(scores.shape[0],n))
        ).to(scores.device)
    i = torch.searchsorted(scores, u)
    return i.view(-1)


@torch.no_grad()
def sample_baseline(
        baseline, 
        x,
        edge_index,
        edge_attr,
        batch,
        center_index, # data_complex.lig_center_index
        type
):
    if type in {"0d", "1d"}:
        n = center_index.shape[0]
        sampled, probs = baseline.sample(n, type)
    elif type in {"2d", "2d_pretrain"}:
        # Embed ligand growth vectors
        xa_lig, xb_lig = baseline.model.linker_model(
            x,
            edge_index,
            edge_attr,
            batch
        )
        xa_lig = xa_lig[center_index]
        xb_lig = xb_lig[center_index]

        # Project and sample
        q = baseline.project(xa_lig, xb_lig)
        if type == "2d_pretrain":
            pass
        else:
            q = torch.clamp(q, 1e-5, 0.9999)
            q = q/(1-q)

        pq = q*baseline.weights
        s = select_weighted_2d(pq, 1).cpu().tolist()
        sampled = [ baseline.motifs[_] for _ in s ]
        probs = pq[(torch.arange(0, len(sampled)), s)]
    else:
        raise ValueError(type)
    return sampled, probs


@torch.no_grad()
def score_baseline(
        baseline,
        data_complex
):
    # Match motifs against motif library
    ids = [ tuple(i) for m in data_complex.motif_ids for i in m ]
    motiflib_motifs = baseline.lookup(ids)
    valid = [ i for i in range(len(motiflib_motifs)) \
        if motiflib_motifs[i] != None ]
    motiflib_motifs = list(filter(lambda i: i is not None, motiflib_motifs))
    sel = [ m["idx"] for m in motiflib_motifs ]

    # 2D probs
    xa_lig, xb_lig = baseline.model.linker_model(
        data_complex.lig_x,
        data_complex.lig_edge_index,
        data_complex.lig_edge_attr,
        data_complex.lig_batch
    )
    xa_lig = xa_lig[data_complex.lig_center_index]
    xb_lig = xb_lig[data_complex.lig_center_index]
    q = baseline.project(xa_lig, xb_lig)
    pq = q*baseline.weights

    # Slice and return
    probs_0 = 1.*torch.ones(len(sel),) / len(baseline.motifs)
    probs_1 = baseline.weights[sel]
    probs_2 = pq[(valid,sel)]
    return probs_0, probs_1, probs_2, motiflib_motifs, valid


@torch.no_grad()
def embed_baseline(
        baseline,
        data_complex
):
    xa_lig, xb_lig = baseline.model.linker_model(
        data_complex.lig_x,
        data_complex.lig_edge_index,
        data_complex.lig_edge_attr,
        data_complex.lig_batch
    )
    xa_lig = xa_lig[data_complex.lig_center_index]
    xb_lig = xb_lig[data_complex.lig_center_index]



def load_baseline(model_arch, motifs_json, device):
    baseline = Baseline(
        model=model_arch,
        motifs=motifs_json,
        device=device
    )
    print("Loaded:", baseline)
    return baseline


