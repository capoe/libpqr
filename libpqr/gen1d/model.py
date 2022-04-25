import json
import gzip

import torch
import numpy as np
import rdkit.Chem as chem


class G1d:
    def __init__(self, motifs, device):
        if isinstance(motifs, (str,)):
            with gzip.open(motifs, 'rt') as f:
                motifs = json.load(f)
        self.device = device
        self.motifs = []
        self.id_to_motif = {}
        self.weights = torch.tensor([])
        self.weights_cum = torch.tensor([])
        self.vectors = torch.tensor([])

        # Initialize
        self.load(motifs)

    def to(self, device):
        self.weights = self.weights.to(device)
        self.weights_cum = self.weights_cum.to(device)
        self.vectors = self.vectors.to(device)

    def eval(self):
        return self

    def __str__(self):
        return f"G1d: Motiflib with {len(self.motifs)} motifs"

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


def load_model_p(motifs_json, device):
    baseline = G1d(
        motifs=motifs_json,
        device=device
    )
    print("Loaded:", baseline)
    return baseline


