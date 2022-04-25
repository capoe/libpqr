#! /usr/bin/env python
import os

import numpy as np
import torch
import torch.nn.functional as F
import rdkit.Chem as chem
import sklearn.metrics
import benchml as bml
log = bml.log

from libpqr.aux import (
    Timer, 
    Checkpointer
)

from libpqr.gen2d import (
    load_model_pq
)

from libpqr.gen3d import (
    get_default_settings,
    build_sampler, 
    build_model, 
    build_opt,
    build_collater,
    motif_to_tensors,
    data_to_torch,
    lex_dot
)


def get_configure_settings(args):
    settings = get_default_settings()
    settings.data_complexes = args.complexes
    settings.data_skiplist = args.complexes_skiplist
    settings.data_shuffle = True
    settings.invert_skiplist = False
    settings.crop_3d = True
    settings.perturb_pos = True
    settings.perturb_steps = 5
    settings.perturb_ampl = 0.5
    settings.cut_hyper_width = 0.5
    settings.cut_hyper_intra = 3.0
    settings.cut_hyper_inter = 7.5
    settings.cut_hyper_edge = 7.5
    settings.radial_weight_pivot = 6.5
    settings.radial_weight_decay = 0.1
    settings.radial_weight_const = 1.0
    settings.depth_2d = 4
    settings.baseline_type = "2d"
    return settings


def train(args):
    # Settings
    np.random.seed(args.seed)
    torch.manual_seed(args.seed+1)
    torch.set_num_threads(1)
    settings = get_configure_settings(args)
    if args.pretrain:
        log << "Setting baseline type: 2D_pretrain" << log.endl
        settings.baseline_type = "2d_pretrain"
    for k, v in settings.items():
        print("Settings: %-20s = %s" % (k, str(v)))

    # Load models
    feat, model = build_model(settings, args.device)
    if args.input != "":
        print("Loading model from '%s'" % args.input)
        model = torch.load(args.input, map_location=args.device)
    opt = build_opt(model)
    chk = Checkpointer(
        location=os.path.dirname(args.output),
        format_str="{tag:s}_epoch{epoch:03d}.arch.chk",
        keep=30
    )

    # Baseline
    baseline = load_model_pq(
        g2d_arch=args.baseline, 
        motifs_json=args.vocabulary,
        device=args.device
    )

    # Train
    chk.save(
        { "model": model, "opt": opt, "epoch": -1 },
        tag="lex", 
        epoch=-1
    )
    for epoch in range(args.epochs):
        log << "Start epoch" << epoch << log.endl
        if epoch > 1:
            log << "Setting baseline type: 2D" << log.endl
            settings.baseline_type = "2d"
        sampler = build_sampler(settings=settings)
        model = train_epoch(
            model,
            baseline,
            sampler,
            opt,
            settings=settings,
            args=args
        )
        log << "Sampler: Profiling" << sampler.timer.report(inline=True) << log.endl
        chk.save(
            { "model": model, "opt": opt, "epoch": epoch },
            tag="lex", 
            epoch=epoch
        )
    torch.save(model, args.output)
    return


def train_epoch(
        model,
        baseline,
        sampler,
        opt,
        settings,
        args
):
    opt.zero_grad()
    acc_metrics = []

    for batchidx, data_complex in enumerate(sampler.batches(
            chunksize=min(args.chunksize, sampler.size()) if not args.dev else 10, 
            batchsize=min(args.batchsize, sampler.size()//20) if not args.dev else 4,
            shuffle=True,
            procs=args.procs
    )):
        T = Timer()
        if data_complex.center_index.shape[0] > 150: 
            log << "SKIP" << data_complex.center_index.shape[0] << data_complex.path << log.endl
            continue
        log << f" Batch {batchidx:6d} {data_complex.center_index.shape[0]:3d} " << log.flush

        with T("to"):
            data_complex = data_complex.to(args.device)

        with T("ba"):
            data_baseline = sample_baseline_batch(
                baseline, 
                data_complex,
                type=settings.baseline_type
            )

        with T("fw"):
            x_complex = model.forward_complex(data_complex, args.device)
            x_motif_data = model.forward_motif(data_complex, args.device)
            x_motif_base = model.forward_motif(data_baseline, args.device)

        with T("lo"):
            cab_data = lex_dot(x_complex, x_motif_data)
            cab_base = lex_dot(x_complex, x_motif_base)
            cab = torch.cat([ cab_data, cab_base ]).view(-1)
            yab = torch.cat([ 
                    torch.ones_like(cab_data), 
                    torch.zeros_like(cab_base) 
                ]).view(-1)
            
            loss = F.binary_cross_entropy(cab, yab)

        with T("ev"):
            log << f"loss= {loss:.4f} " << log.flush
            auc, acc, info = measure_batch(cab, yab)
            acc_metrics.append({ "auc": auc, "acc": acc })
            log << info << log.flush
            if batchidx % 10 == 0:
                log << T.report(inline=True) << log.flush

        with T("bw"):
            loss = loss / args.acc_grads
            loss.backward()

        if (batchidx+1) % args.acc_grads == 0:
            opt.step()
            opt.zero_grad()
            auc = np.mean(np.array([ m["auc"] for m in acc_metrics ]))
            acc = np.mean(np.array([ m["acc"] for m in acc_metrics ]))
            acc_metrics = []
            log << f"<auc>= {auc:1.4f} <acc>= {acc:1.4f} [step]" << log.flush
        log << log.endl

        if batchidx % args.chk_every == 0:
            torch.save(model, os.path.join(os.path.dirname(args.output), "latest.arch"))

    opt.step()
    return model


def sample_baseline_batch(
        baseline, 
        data_complex,
        type
):
    motifs, probs = baseline.sample(
        x=data_complex.lig_x,
        edge_index=data_complex.lig_edge_index,
        edge_attr=data_complex.lig_edge_attr,
        batch=data_complex.lig_batch,
        center_index=data_complex.lig_center_index,
        type=type
    )
    data_baseline = []
    for m in motifs:
        motif_data = motif_to_tensors(m)
        data_baseline.append(data_to_torch(motif_data))
    collate = build_collater()
    data_baseline = collate(data_baseline).to(args.device)
    return data_baseline


def measure_batch(out, y):
    y = y.detach().cpu().numpy().flatten()
    out = out.detach().cpu().numpy().flatten()
    auc_tot = sklearn.metrics.roc_auc_score(y, out)
    out_bin = np.heaviside(out - 0.5, 0.)
    acc = (y*out_bin).sum() + ((1-y)*(1-out_bin)).sum()
    acc_tot = acc / len(out_bin)
    info = f" auc= {auc_tot:.4f}  acc= {acc_tot:.4f}" 
    return auc_tot, acc_tot, info 


if __name__ == "__main__":
    log.Connect()
    log.AddArg("complexes", str)
    log.AddArg("complexes_skiplist", str)
    log.AddArg("baseline", str)
    log.AddArg("vocabulary", str)
    log.AddArg("output", str)
    log.AddArg("input", str, default="")
    log.AddArg("pretrain", "toggle", default=False)
    log.AddArg("chunksize", int, default=1000)
    log.AddArg("batchsize", int, default=4)
    log.AddArg("acc_grads", int, default=10)
    log.AddArg("chk_every", int, default=1000)
    log.AddArg("epochs", int, default=40)
    log.AddArg("seed", int, default=1)
    log.AddArg("dev", "toggle", default=False)
    log.AddArg("procs", int, default=12)
    args = log.Parse()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    train(args)


