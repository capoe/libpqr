#! /usr/bin/env python
import json
import os
import gzip
import multiprocessing as mp

from collections import Counter

import numpy as np
import rdkit.Chem as chem
import benchml as bml
import libpqr.gen1d as gen1d

log = bml.log


def run(smiles, replicates, procs):
    print("Fragment ...")
    smiles = smiles * replicates
    smiles = [(idx, smi) for idx, smi in enumerate(smiles)]
    if procs < 2:
        motifs = list(map(gen1d.motifs_from_smiles, smiles))
    else:
        with mp.Pool(procs) as pool:
            motifs = pool.map(gen1d.motifs_from_smiles, smiles)

    print("Merge ...")
    motifs_global = Counter([])
    for mots in motifs:
        motifs_global.update(mots)
    motifs_global = [[smi, freq] for smi, freq in motifs_global.items()]
    motifs_global = list(sorted(motifs_global, key=lambda m: -m[1]))
    motifs_global = [
        {"smi": m[0].split("_")[0], "vec": int(m[0].split("_")[1]), "freq": m[1]}
        for m in motifs_global
    ]

    return motifs_global


if __name__ == "__main__":
    log.Connect()
    log.AddArg("json_gz", str)
    log.AddArg("output", str)
    log.AddArg("procs", int, default=8)
    log.AddArg("replicates", int, default=10)
    args = log.Parse()

    with gzip.open(args.json_gz, "rt") as f:
        smiles = json.load(f)
    motifs = run(smiles=smiles, replicates=args.replicates, procs=args.procs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with gzip.open(args.output, "wt") as f:
        json.dump(motifs, f, indent=1)
