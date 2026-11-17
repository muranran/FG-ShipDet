"""Fit distance heads and score independent queries using a fixed reference graph."""

import argparse
import json
import math
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

import fgship_core as core


def run(data, output, fold, seed, device, ratio, poolseed, controls):
    core.DEVICE = device
    core.K = 15
    core.ALPHA = 0.9
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    split = data / f"fold{fold}"
    src = output / f"b{seed}" / f"fold{fold}"
    fd = src / "features"
    dest = src / ("method" if ratio == 1 else f"pool{ratio}_seed{poolseed}")
    dest.mkdir(exist_ok=True)
    if (dest / "complete.json").exists():
        raise FileExistsError("Scores already frozen; use another output directory.")
    meta = json.loads((fd / "complete.json").read_text())
    assert core.sha(src / "final.pt") == meta["model_sha256"]
    for n, h in meta["sha256"].items():
        assert core.sha(fd / n) == h
    ff = {
        s: dict(np.load(fd / (s + ".npz")))
        for s in ["train", "pool", "calibration", "test"]
    }
    keep = np.ones(len(ff["pool"]["paths"]), bool)
    if ratio < 1:
        groups = sorted(
            set(ff["pool"]["groups"]),
            key=lambda v: hashlib.sha256(
                (str(poolseed) + "|" + v).encode()
            ).hexdigest(),
        )
        selected = groups[: max(1, math.ceil(len(groups) * ratio))]
        keep &= np.isin(ff["pool"]["groups"], selected)
        ff["pool"] = core.take(ff["pool"], keep)
    fit = core.join(ff["train"], ff["pool"])
    for key in ["paths", "groups", "pixel_hash", "resized_hash"]:
        assert set(fit[key]).isdisjoint(ff["test"][key]), key
        assert set(fit[key]).isdisjoint(ff["calibration"][key]), key
    calix = np.load(split / "calibration_indices.npz")
    gi = calix["normalization"]
    di = calix["decision"]
    assert set(ff["calibration"]["groups"][gi]).isdisjoint(
        ff["calibration"]["groups"][di]
    )
    raw = {}
    known = {}
    for s in ff:
        L, _ = core.distances(ff["train"], ff[s])
        U, _ = core.distances(ff["pool"], ff[s])
        raw[s] = L["legacy"] - U["legacy"]
        known[s] = L["legacy"]
    heads = {}
    for headseed in [24042, 24043, 24044]:
        core.HS = headseed
        hd = dest / f"head{headseed}"
        hd.mkdir(exist_ok=True)
        heads[headseed] = core.train_head(ff, raw, gi, 49, hd)
    P = core.graph(fit, fit)
    Pc = core.graph(fit, ff["calibration"])
    Pt = core.graph(fit, ff["test"])
    nl = len(ff["train"]["paths"])
    H, conv = core.diffuse(P, nl, np.ones((nl, 1)))
    H0 = np.concatenate([np.ones((nl, 1)), np.zeros((len(fit["paths"]) - nl, 1))])
    support = {}
    for name, h in [("Multistep", H), ("OneHop", H0)]:
        support[name] = (1 - 0.9 * (Pc @ h).ravel(), 1 - 0.9 * (Pt @ h).ravel())
    support["SimpleDistance"] = (known["calibration"], known["test"])
    if controls:
        support["Energy"] = tuple(
            -logsumexp(ff[s]["logits"], axis=1) for s in ["calibration", "test"]
        )
        for rs in [81001, 81002, 81003]:
            Pr, _ = core.rewired(P, fit, fit, rs)
            Pcr, _ = core.rewired(Pc, ff["calibration"], fit, rs + 100)
            Ptr, _ = core.rewired(Pt, ff["test"], fit, rs + 200)
            hh, _ = core.diffuse(Pr, nl, np.ones((nl, 1)))
            support[f"Rewire{rs}"] = (
                1 - 0.9 * (Pcr @ hh).ravel(),
                1 - 0.9 * (Ptr @ hh).ravel(),
            )
        for kk, aa in [(5, 0.9), (30, 0.9), (15, 0.5), (15, 0.7)]:
            core.K = kk
            core.ALPHA = aa
            pp, pc, pt = (
                (P, Pc, Pt)
                if kk == 15
                else (
                    core.graph(fit, fit),
                    core.graph(fit, ff["calibration"]),
                    core.graph(fit, ff["test"]),
                )
            )
            hh, _ = core.diffuse(pp, nl, np.ones((nl, 1)))
            support[f"K{kk}A{aa}"] = (
                1 - aa * (pc @ hh).ravel(),
                1 - aa * (pt @ hh).ravel(),
            )
        core.K = 15
        core.ALPHA = 0.9
    scores = {}
    normalization = {}
    for hs, (cc, tt) in heads.items():
        scores[f"Regression_s{hs}"] = tt
        scores[f"cal_Regression_s{hs}"] = cc
        c, t, n = core.norm(cc, tt, gi)
        for name, (sc, st) in support.items():
            a, b, sn = core.norm(sc, st, gi)
            key = name + f"_s{hs}"
            scores[key] = (t + b) / 2
            scores["cal_" + key] = (c + a) / 2
            normalization[key] = dict(relation=n, support=sn, weights=[0.5, 0.5])
    # No-head ablations reuse the same reference distances and graphs.
    for label, base in [("RawRelative", raw), ("KnownDistance", known)]:
        cc, tt = base["calibration"], base["test"]
        scores[label] = tt
        scores["cal_" + label] = cc
        c, t, n = core.norm(cc, tt, gi)
        for method in ["OneHop", "Multistep"]:
            a, b, sn = core.norm(*support[method], gi)
            key = label + "_" + method
            scores[key] = (t + b) / 2
            scores["cal_" + key] = (c + a) / 2
    for method in ["OneHop", "Multistep"]:
        scores["Graph_" + method] = support[method][1]
        scores["cal_Graph_" + method] = support[method][0]
    assert all(np.isfinite(v).all() for v in scores.values())
    thresholds = {
        n: {
            str(b): float(np.quantile(scores["cal_" + n][di], 1 - b, method="higher"))
            for b in [0.01, 0.05, 0.1]
        }
        for n in scores
        if not n.startswith("cal_")
    }
    np.savez_compressed(dest / "scores.npz", **scores)
    core.dump(dest / "thresholds.json", thresholds)
    core.dump(dest / "normalization.json", normalization)
    core.dump(
        dest / "complete.json",
        dict(
            status="scores_frozen",
            fold=fold,
            backbone_seed=seed,
            pool_ratio=ratio,
            pool_seed=poolseed,
            L=nl,
            U=len(ff["pool"]["paths"]),
            convergence=conv,
            test_labels_read=False,
            sha256={n: core.sha(dest / n) for n in ["scores.npz", "thresholds.json"]},
            features_sha256=meta["sha256"],
        ),
    )
    print("Frozen:", dest, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/fgsrcs"))
    p.add_argument("--output", type=Path, default=Path("runs"))
    p.add_argument("--fold", type=int, required=True, choices=range(4))
    p.add_argument("--seed", type=int, default=27042)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pool-ratio", type=float, choices=[1.0, 0.5, 0.25], default=1.0)
    p.add_argument("--pool-seed", type=int, choices=[23001, 23002], default=23001)
    p.add_argument("--controls", action="store_true")
    a = p.parse_args()
    run(
        a.data,
        a.output,
        a.fold,
        a.seed,
        a.device,
        a.pool_ratio,
        a.pool_seed,
        a.controls,
    )
