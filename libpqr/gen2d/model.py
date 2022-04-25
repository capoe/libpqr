import json
import gzip

import torch
import numpy as np
import rdkit.Chem as chem

from torch_geometric.loader.dataloader import Collater

from .arch import G2dData, G2dFeaturizer
from ..gen1d import G1d


class G2d:
    def __init__(self, g2d, motifs, device):
        if isinstance(g2d, (str,)):
            g2d = torch.load(g2d, map_location=device).eval()
        self.g2d = g2d
        self.g1d = G1d(motifs, device)
        self.device = device
        self.xa_linker, self.xb_linker = (None, None)
        if self.g2d is not None:
            self.embed()

    def to(self, device):
        pass

    def eval(self):
        self.g2d.eval()
        self.g1d.eval()
        return self

    def __str__(self):
        return f"G2d: Motiflib with {len(self.g1d.motifs)} motifs"

    @torch.no_grad()
    def embed(self):
        print("Embed G2d ...")
        feat = G2dFeaturizer()
        datalist = []
        for m in self.g1d.motifs:
            data = G2dData(mol=m["mol"])
            data = feat(data)
            data.atom_vector = torch.tensor([ m["vec"] ]).long()
            datalist.append(data)
        collate = Collater([], [])
        data = collate(datalist).to(self.device)
        self.xa_linker, self.xb_linker = self.g2d.linker_model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )
        self.xa_linker = self.xa_linker[data.atom_vector]
        self.xb_linker = self.xb_linker[data.atom_vector]

    @torch.no_grad()
    def project(self, xa, xb):
        linker_mat = self.g2d.linker_model.final_dense(
            xa, xb, self.xa_linker, self.xb_linker
        )
        return linker_mat

    @torch.no_grad()
    def sample(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            center_index, # data_complex.lig_center_index
            type
    ):
        if type == "0d":
            n = center_index.shape[0]
            sel = np.random.randint(0, len(self.g1d.motifs), size=(n,))
            sampled = [ self.motifs[_] for _ in sel ]
            probs = 1.*torch.ones((len(sel),)) / len(self.g1d.motifs)
        elif type == "1d":
            n = center_index.shape[0]
            u = np.random.uniform(0, 1, size=(n,)) 
            u = torch.from_numpy(u).to(self.device)
            sel = torch.searchsorted(self.weights_cum, u)
            sampled = [ self.motifs[_] for _ in sel ]
            probs = torch.tensor([ self.weights[s] for s in sel ])
        elif type in {"2d", "2d_pretrain"}:
            # Embed ligand growth vectors
            xa_lig, xb_lig = self.g2d.linker_model(
                x,
                edge_index,
                edge_attr,
                batch
            )
            xa_lig = xa_lig[center_index]
            xb_lig = xb_lig[center_index]

            # Project and sample
            q = self.project(xa_lig, xb_lig)
            if type == "2d_pretrain":
                pass
            else:
                q = torch.clamp(q, 1e-5, 0.9999)
                q = q/(1-q)

            pq = q*self.g1d.weights
            s = select_weighted_2d(pq, 1).cpu().tolist()
            sampled = [ self.g1d.motifs[_] for _ in s ]
            probs = pq[(torch.arange(0, len(sampled)), s)]
        else:
            raise ValueError(type)
        return sampled, probs

    def lookup(self, *args, **kwargs):
        return self.g1d.lookup(*args, **kwargs)


def select_weighted_2d(scores, n):
    scores = scores.cumsum(dim=-1)
    scores = (scores.T/scores[:,-1]).T
    u = torch.from_numpy(
            np.random.uniform(size=(scores.shape[0],n))
        ).to(scores.device)
    i = torch.searchsorted(scores, u)
    return i.view(-1)


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
    xa_lig, xb_lig = baseline.g2d.linker_model(
        data_complex.lig_x,
        data_complex.lig_edge_index,
        data_complex.lig_edge_attr,
        data_complex.lig_batch
    )
    xa_lig = xa_lig[data_complex.lig_center_index]
    xb_lig = xb_lig[data_complex.lig_center_index]
    q = baseline.project(xa_lig, xb_lig)
    pq = q*baseline.g1d.weights

    # Slice and return
    probs_0 = 1.*torch.ones(len(sel),) / len(baseline.g1d.motifs)
    probs_1 = baseline.g1d.weights[sel]
    probs_2 = pq[(valid,sel)]
    return probs_0, probs_1, probs_2, motiflib_motifs, valid


@torch.no_grad()
def embed_baseline(
        baseline,
        data_complex
):
    xa_lig, xb_lig = baseline.g2d.linker_model(
        data_complex.lig_x,
        data_complex.lig_edge_index,
        data_complex.lig_edge_attr,
        data_complex.lig_batch
    )
    xa_lig = xa_lig[data_complex.lig_center_index]
    xb_lig = xb_lig[data_complex.lig_center_index]



def load_model_pq(g2d_arch, motifs_json, device):
    baseline = G2d(
        g2d=g2d_arch,
        motifs=motifs_json,
        device=device
    )
    print("Loaded:", baseline)
    return baseline


