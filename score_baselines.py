"""Eighteen fixed post-hoc baselines fitted on known FGSRCS training data."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax, logsumexp
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from fgship_core import dump, rows, sha
import fgship_openmax as om

METHODS = [
    "MSP",
    "Energy_T1",
    "MaxLogit",
    "Entropy",
    "PrototypeCos",
    "MahalanobisEmpirical_LastLayer",
    "MahalanobisLW_LastLayer",
    "KNN_1",
    "KNN_10",
    "KNN_50",
    "KNN_1000",
    "Residual_D1000",
    "ViM_D1000",
    "ReAct90_Energy",
    "ReAct99_Energy",
    "OpenMax_Eucos20",
    "OpenMax_Euclid20",
    "OpenMax_Euclid10",
]


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def mahal(x, means, prec):
    xp = x @ prec
    mp = means @ prec
    return np.maximum(
        (xp * x).sum(1)[:, None] + (mp * means).sum(1)[None] - 2 * xp @ means.T, 0
    ).min(1)


def score(data, output, fold, seed):
    D = data / f"fold{fold}"
    O = output / f"b{seed}"
    di = np.load(D / "calibration_indices.npz")["decision"]
    dest = O / f"fold{fold}/posthoc"
    dest.mkdir(exist_ok=True)
    assert not (dest / "complete.json").exists(), "Baseline scores already frozen"
    src = dest.parent / "features"
    meta = json.loads((src / "complete.json").read_text())
    assert sha(dest.parent / "final.pt") == meta["model_sha256"]
    for name, digest in meta["sha256"].items():
        assert sha(src / name) == digest
    raw = {
        k: dict(np.load(src / (k + ".npz"))) for k in ["train", "calibration", "test"]
    }
    ci = json.loads((D / "class_to_idx.json").read_text())
    tr = rows(D / "train.jsonl")
    cal = rows(D / "calibration.jsonl")
    test = rows(D / "test_paths.jsonl")
    for split, rr in [("train", tr), ("calibration", cal), ("test", test)]:
        assert list(raw[split]["paths"]) == [r["path"] for r in rr]
    y = np.array([ci[r["label"]] for r in tr])
    x = {k: r["features"].astype("float64") for k, r in raw.items()}
    state = torch.load(
        dest.parent / "final.pt", map_location="cpu", weights_only=False
    )["model"]
    w = state["fc.weight"].numpy().astype("float64")
    b = state["fc.bias"].numpy().astype("float64")
    z = {k: xx @ w.T + b for k, xx in x.items()}
    err = max(float(np.max(np.abs(z[k] - raw[k]["logits"]))) for k in x)
    assert err < 1e-4, err
    K = len(ci)
    means = np.stack([x["train"][y == k].mean(0) for k in range(K)])
    protos = normalize(
        np.stack([normalize(x["train"])[y == k].mean(0) for k in range(K)])
    )
    centered = x["train"] - means[y]
    ec = EmpiricalCovariance(assume_centered=True).fit(centered)
    lw = LedoitWolf(assume_centered=True).fit(centered)
    u = -np.linalg.pinv(w) @ b
    xc = x["train"] - u
    ev, vec = eigh(xc.T @ xc / len(xc))
    ns = np.ascontiguousarray(vec[:, :1048])
    alpha = float(z["train"].max(1).mean() / np.linalg.norm(xc @ ns, axis=1).mean())
    assert np.isfinite(alpha) and alpha > 0
    unit = normalize(x["train"]) + 1e-10
    clips = {q: float(np.quantile(x["train"], q)) for q in [0.9, 0.99]}
    scores = {}
    preds = {}
    for split in ["calibration", "test"]:
        xx = x[split]
        zz = z[split]
        p = softmax(zz, axis=1)
        sv = {
            "MSP": 1 - p.max(1),
            "Energy_T1": -logsumexp(zz, axis=1),
            "MaxLogit": -zz.max(1),
            "Entropy": -(p * np.log(p.clip(1e-300))).sum(1),
            "PrototypeCos": 1 - (normalize(xx) @ protos.T).max(1),
            "MahalanobisEmpirical_LastLayer": mahal(xx, means, ec.precision_),
            "MahalanobisLW_LastLayer": mahal(xx, means, lw.precision_),
        }
        kn = {k: [] for k in [1, 10, 50, 1000]}
        for start in range(0, len(xx), 128):
            q = normalize(xx[start : start + 128]) + 1e-10
            dd = np.maximum(
                (q * q).sum(1)[:, None] + (unit * unit).sum(1)[None] - 2 * q @ unit.T, 0
            )
            part = np.partition(dd, [0, 9, 49, 999], axis=1)
            for k in kn:
                kn[k].extend(part[:, k - 1])
        for k, v in kn.items():
            sv["KNN_" + str(k)] = np.array(v)
        residual = np.linalg.norm((xx - u) @ ns, axis=1)
        sv["Residual_D1000"] = residual
        sv["ViM_D1000"] = alpha * residual - logsumexp(zz, axis=1)
        for q in [0.9, 0.99]:
            logits = np.minimum(xx, clips[q]) @ w.T + b
            name = "ReAct" + str(round(q * 100)) + "_Energy"
            sv[name] = -logsumexp(logits, axis=1)
            preds[split + "_" + name] = logits.argmax(1)
        for name, values in sv.items():
            assert np.isfinite(values).all()
            scores[split + "_" + name] = values
            preds.setdefault(split + "_" + name, zz.argmax(1))
        for i in [0, len(xx) // 2, len(xx) - 1]:
            scalar = min(
                float((xx[i] - m) @ ec.precision_ @ (xx[i] - m)) for m in means
            )
            assert abs(scalar - sv["MahalanobisEmpirical_LastLayer"][i]) < 1e-5
            direct = np.sort(
                cdist(normalize(xx[i : i + 1]) + 1e-10, unit, "sqeuclidean")[0]
            )
            for k in kn:
                assert abs(direct[k - 1] - sv["KNN_" + str(k)][i]) < 1e-10
    for name in METHODS[-3:]:
        kind = "eucos" if "Eucos" in name else "euclidean"
        tail = 10 if name.endswith("10") else 20
        dd = dest / name
        dd.mkdir(exist_ok=True)
        mu, mrs, saved = om.fit(z["train"], y, tail, kind, dd)
        for split in ["calibration", "test"]:
            pr = om.score(z[split], mu, mrs, kind, alpha=10)
            scores[split + "_" + name] = pr[:, -1]
            preds[split + "_" + name] = z[split].argmax(1)
            preds["native_" + split + "_" + name] = pr.argmax(1)

    assert set(k.replace("test_", "") for k in scores if k.startswith("test_")) == set(
        METHODS
    )
    np.savez_compressed(
        dest / "scores.npz", **scores, **{"pred_" + k: v for k, v in preds.items()}
    )
    thresholds = {
        m: {
            str(q): float(
                np.quantile(scores["calibration_" + m][di], 1 - q, method="higher")
            )
            for q in [0.01, 0.05, 0.1]
        }
        for m in METHODS
    }
    dump(dest / "thresholds.json", thresholds)
    np.savez_compressed(
        dest / "fit.npz",
        means=means,
        protos=protos,
        empirical_precision=ec.precision_,
        lw_precision=lw.precision_,
        u=u,
        NS=ns,
        alpha=alpha,
        fc_weight=w,
        fc_bias=b,
        train_unit=unit,
    )
    audit = dict(
        seed=seed,
        known_only_train_support=True,
        fc_max_error=err,
        mahal_scalar_verified=True,
        knn_direct_verified=True,
        clip=clips,
        lw_shrinkage=float(lw.shrinkage_),
        vim_alpha=alpha,
    )
    dump(dest / "audit.json", audit)
    print("SCORED", seed, flush=True)
    dump(
        dest / "complete.json",
        dict(
            status="scores_frozen",
            test_labels_read=False,
            sha256={n: sha(dest / n) for n in ["scores.npz", "thresholds.json"]},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/fgsrcs"))
    p.add_argument("--output", type=Path, default=Path("runs"))
    p.add_argument("--fold", type=int, required=True, choices=range(4))
    p.add_argument("--seed", type=int, default=27042)
    a = p.parse_args()
    score(a.data, a.output, a.fold, a.seed)
