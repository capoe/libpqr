#! /usr/bin/env python
import os
import json

import numpy as np
import torch
import torch.nn.functional as F
import rdkit.Chem as chem
import sklearn.metrics
import benchml as bml
log = bml.log

from libpqr.aux import Timer
from libpqr.gen2d import load_baseline
from libpqr.gen3d import (
    build_collater,
    complex_to_tensors,
    data_to_torch,
    LexBaseline,
)


def run(args):
    model = torch.load(args.model, map_location=args.device)
    model.settings.perturb_pos = False # TODO as .eval()
    for k, v in model.settings.items():
        log << log.mb << "Settings: %-20s = %s" % (k, str(v)) << log.endl
    baseline_2d = load_baseline(
        model_arch=args.baseline, 
        motifs_json=args.vocabulary,
        device=args.device
    )
    baseline_3d = LexBaseline(model, baseline_2d, args.device)
    cmplx_data = json.load(open(args.complexes))
    for cmplx in cmplx_data:
        log << log.my << "Complex:" \
            << os.path.basename(cmplx["env_pdb"]) \
            << os.path.basename(cmplx["mol_sdf"]) \
            << log.endl
        output = single_point_sample(
            baseline_2d,
            baseline_3d,
            settings=model.settings,
            args=args,
            cmplx=cmplx
        )
    return


@torch.no_grad()
def single_point_sample(
        baseline_2d,
        baseline_3d,
        settings,
        args,
        cmplx,
        pq_highpass=100
):
    data = complex_to_tensors(cmplx, settings=settings)
    data = data_to_torch(data)
    collate = build_collater()
    data_complex = collate([data])

    T = Timer()
    with T("to"):
        data_complex = data_complex.to(args.device)

    with T("ba"):
        xa_lig, xb_lig = baseline_2d.model.linker_model(
            data_complex.lig_x,
            data_complex.lig_edge_index,
            data_complex.lig_edge_attr,
            data_complex.lig_batch
        )
        xa_lig = xa_lig[data_complex.lig_center_index]
        xb_lig = xb_lig[data_complex.lig_center_index]
        z = 1.*torch.ones_like(baseline_2d.weights) / baseline_2d.weights.shape[0]
        p = baseline_2d.weights
        q = baseline_2d.project(xa_lig, xb_lig)
        q = torch.clamp(q, 0., 0.9999)
        q_agg = q/(1-q)

    with T("fw"):
        x_complex = baseline_3d.model.forward_complex(data_complex, args.device)
        r = baseline_3d.project(x_complex)
        r = torch.clamp(r, 0., 0.9999)
        r_agg = r/(1-r)

    # Likelihood factors
    p_agg = baseline_2d.weights.cpu().numpy()
    q_agg = q_agg.cpu().numpy().flatten()
    r_agg = r_agg.cpu().numpy().flatten()
    pq_agg = q_agg * p_agg
    pqr_agg = q_agg * r_agg * p_agg
    pq_agg = pq_agg / np.sum(pq_agg)
    pqr_agg = pqr_agg / np.sum(pqr_agg)
    r_eff = pqr_agg/pq_agg

    # Apply pq plateau filter
    order_pq = np.argsort(pq_agg)[::-1]
    top_n = order_pq[0:pq_highpass]
    r_top_n = r_eff[top_n]
    motifs_top_n = []
    for i,r in zip(top_n, r_top_n):
        id = baseline_2d.motifs[i]["smi"]+"_"+str(baseline_2d.motifs[i]["vec"])
        motifs_top_n.append([ id, pq_agg[i], pqr_agg[i], r ])
    motifs_top_n = list(sorted(motifs_top_n, key=lambda m: -m[3]))
    print("Ranking (with pq high-pass filter)")
    for midx, m in enumerate(motifs_top_n):
        if m[3] < 1.: break
        print("   %-25s pq= %1.4f   pqr= %1.4f   r= %9.4f" % tuple(m))

    if args.verbose:
        log << log.mb << T.report(delim="\n") << log.endl
   
    return np.concatenate(
        [
            p_agg.reshape(-1,1), 
            q_agg.reshape(-1,1), 
            r_agg.reshape(-1,1), 
            r_eff.reshape(-1,1)
        ], 
        axis=1
    )


if __name__ == "__main__":
    log.Connect()
    log.AddArg("complexes", str)
    log.AddArg("baseline", str)
    log.AddArg("vocabulary", str)
    log.AddArg("model", str, default="")
    log.AddArg("verbose", "toggle", default=False)
    args = log.Parse()
    args.device = torch.device("cpu")
    run(args)


