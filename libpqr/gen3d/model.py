import json
import gzip

import torch
import numpy as np
import rdkit.Chem as chem

from torch_geometric.loader.dataloader import Collater

from .aux import motif_to_tensors, data_to_torch
from .build import build_collater
from .arch import lex_dot, lex_dot_dense

from ..gen1d import G1d
from ..gen2d import G2d


class G3d:
    def __init__(self, g3d, g2d, device):
        if isinstance(g3d, (str,)):
            g3d = torch.load(g3d, map_location=device).eval()
        self.g3d = g3d
        self.g2d = g2d
        self.device = device
        self.x_motif = None
        self.embed()

    def to(self, device):
        pass

    def eval(self):
        self.g3d.eval()
        self.g3d.settings.perturb_pos = False # TODO Move to G3dComponents
        self.g2d.eval()
        return self

    @torch.no_grad()
    def embed(self):
        print("Embed G3d ...")
        motifs = self.g2d.g1d.motifs
        data = []
        for m in motifs:
            motif_data = motif_to_tensors(m)
            data.append(data_to_torch(motif_data))
        collate = build_collater()
        data = collate(data).to(self.device)
        self.x_motif = self.g3d.forward_motif(data, self.device)
        print('... done: shape(x_motif)', self.x_motif.shape)
        return data

    @torch.no_grad()
    def project(self, x):
        return lex_dot_dense(x, self.x_motif)

    def __str__(self):
        return f"LexBaseline: Motiflib with {len(self.g2d.g1d.motifs)} motifs"


def load_model_pqr(g3d_arch, g2d_arch, motifs, device):
    g2d = G2d(g2d_arch, motifs, device)
    g3d = G3d(g3d_arch, g2d, device)
    return g3d


