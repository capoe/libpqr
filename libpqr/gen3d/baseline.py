import json
import gzip

import torch
import numpy as np
import rdkit.Chem as chem

from torch_geometric.loader.dataloader import Collater

from .aux import motif_to_tensors, data_to_torch
from .build import build_collater
from .arch import lex_dot, lex_dot_dense


class LexBaseline:
    def __init__(self, model, baseline_2d, device):
        self.baseline_2d = baseline_2d
        if isinstance(model, (str,)):
            model = torch.load(model, map_location=device).eval()
        self.model = model
        self.device = device
        self.x_motif = None
        self.embed()

    def to(self, device):
        pass

    @torch.no_grad()
    def embed(self):
        print("Embed LexBaseline")
        motifs = self.baseline_2d.motifs
        data = []
        for m in motifs:
            motif_data = motif_to_tensors(m)
            data.append(data_to_torch(motif_data))
        collate = build_collater()
        data = collate(data).to(self.device)
        self.x_motif = self.model.forward_motif(data, self.device)
        print('... done: shape(x_motif)', self.x_motif.shape)
        return data

    @torch.no_grad()
    def project(self, x):
        return lex_dot_dense(x, self.x_motif)

    def __str__(self):
        return f"LexBaseline: Motiflib with {len(self.baseline_2d.motifs)} motifs"


