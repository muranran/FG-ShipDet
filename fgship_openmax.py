"""Single-crop OpenMax scoring, fitted to correct known training logits.
Uses libmr for Weibull tail fitting; no third-party source is bundled.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from fgship_core import dump
import libmr


def dist(x, m, kind):
    d = cdist(x, m, "euclidean") / 200.0
    if kind == "eucos":
        d += cdist(x, m, "cosine")
    return d


def fit(z, y, tail, kind, dest):
    c = z.shape[1]
    m = []
    mrs = []
    audit = []
    saved = {}
    for cl in range(c):
        ix = (y == cl) & (z.argmax(1) == cl)
        v = z[ix]
        assert len(v) >= 3, (cl, len(v))
        mu = v.mean(0)
        d = dist(v, mu[None], kind)[:, 0]
        t = np.sort(d)[-min(tail, len(d)) :]
        mr = libmr.MR()
        mr.fit_high(t.astype(np.float64), len(t))
        assert np.isfinite([mr.w_score(float(q)) for q in t]).all()
        m.append(mu)
        mrs.append(mr)
        saved[f"tail{cl}"] = t
        audit.append(
            dict(
                class_index=cl,
                train_total=int((y == cl).sum()),
                correct_fit_samples=len(v),
                tail_used=len(t),
                parameters=list(mr.get_params()),
            )
        )
    saved["means"] = np.stack(m)
    np.savez_compressed(dest / "fit_support.npz", **saved)
    dump(dest / "fit_audit.json", audit)
    return np.stack(m), mrs, saved


def score(z, means, mrs, kind, alpha=10):
    d = dist(z, means, kind)
    w = np.stack(
        [[mr.w_score(float(a)) for a in d[:, cl]] for cl, mr in enumerate(mrs)], 1
    )
    ranks = np.argsort(z, axis=1)[:, ::-1][:, :alpha]
    rw = np.zeros_like(z)
    for rank in range(alpha):
        rw[np.arange(len(z)), ranks[:, rank]] = (alpha - rank) / alpha
    removed = z * w * rw
    modified = z - removed
    unknown = removed.sum(1)
    p = softmax(np.column_stack([modified, unknown]), axis=1)
    assert np.allclose(modified.sum(1) + unknown, z.sum(1), atol=1e-8)
    assert np.allclose(p.sum(1), 1)
    assert np.isfinite(p).all()
    return p
