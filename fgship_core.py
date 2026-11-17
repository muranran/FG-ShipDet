"""Distance regression and fixed-reference neighborhood scoring on FGSRCS."""

import hashlib, json, math, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F

DEVICE = "cuda"
K = 15
ALPHA = 0.9
HS = 24042


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    t.replace(p)


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def design(f):
    return np.concatenate(
        [
            f["global_features"],
            f["local_features"].reshape(len(f["paths"]), -1) / np.sqrt(3),
        ],
        axis=1,
    ).astype(np.float32)


def groupmin(x, g, ng):
    y = torch.full((len(x), ng), float("inf"), device=x.device)
    return y.scatter_reduce_(
        1, g[None].expand(len(x), -1), x, reduce="amin", include_self=True
    )


@torch.inference_mode()
def distances(ref, q):
    g = torch.tensor(ref["global_features"], device=DEVICE)
    l = torch.tensor(ref["local_features"], device=DEVICE)
    ug, gi = np.unique(ref["groups"], return_inverse=True)
    gi = torch.tensor(gi, device=DEVICE)
    out = {k: [] for k in ["legacy", "clean", "coupled", "group3"]}
    bad_nearest = 0
    total = len(q["paths"])
    for i in range(0, total, 128):
        end = min(i + 128, total)
        a = torch.tensor(q["global_features"][i:end], device=DEVICE)
        b = torch.tensor(q["local_features"][i:end], device=DEVICE)
        dg = (1 - a @ g.T).clamp_min(0)
        dl = torch.minimum(
            1 - torch.einsum("qpc,npc->qn", b, l) / 3,
            1 - torch.einsum("qpc,npc->qn", b.flip(1), l) / 3,
        ).clamp_min(0)
        mask = torch.tensor(
            q["groups"][i:end, None] == ref["groups"][None, :], device=DEVICE
        )
        dg.masked_fill_(mask, float("inf"))
        dl.masked_fill_(mask, float("inf"))
        out["legacy"].append(((dg.min(1).values + dl.min(1).values) / 2).cpu().numpy())
        duplicate = torch.tensor(
            q["pixel_hash"][i:end, None] == ref["pixel_hash"][None, :], device=DEVICE
        )
        near = dg.argmin(1)
        bad_nearest += int(duplicate[torch.arange(len(a), device=DEVICE), near].sum())
        dg.masked_fill_(duplicate, float("inf"))
        dl.masked_fill_(duplicate, float("inf"))
        out["clean"].append(((dg.min(1).values + dl.min(1).values) / 2).cpu().numpy())
        joint = (dg + dl) / 2
        out["coupled"].append(joint.min(1).values.cpu().numpy())
        gm = groupmin(joint, gi, len(ug))
        top = gm.topk(3, largest=False).values
        assert torch.isfinite(top).all()
        out["group3"].append(top.mean(1).cpu().numpy())
    result = {k: np.concatenate(v) for k, v in out.items()}
    assert all(np.isfinite(x).all() for x in result.values())
    return result, {
        "query_n": total,
        "reference_n": len(g),
        "reference_image_groups": len(ug),
        "legacy_global_nearest_cross_group_exact_duplicate_queries": bad_nearest,
    }


@torch.inference_mode()
def predict(net, x):
    return torch.cat([net(v).flatten() for v in x.split(512)]).cpu().numpy()


@torch.inference_mode()
def graph(ref, q):
    g = torch.tensor(ref["global_features"], device=DEVICE)
    l = torch.tensor(ref["local_features"], device=DEVICE)
    inds = []
    weights = []
    for start in range(0, len(q["paths"]), 128):
        stop = start + 128
        a = torch.tensor(q["global_features"][start:stop], device=DEVICE)
        b = torch.tensor(q["local_features"][start:stop], device=DEVICE)
        dg = (1 - a @ g.T).clamp_min(0)
        dl = torch.minimum(
            1 - torch.einsum("qpc,npc->qn", b, l) / 3,
            1 - torch.einsum("qpc,npc->qn", b.flip(1), l) / 3,
        ).clamp_min(0)
        d = (dg + dl) / 2
        mask = (q["groups"][start:stop, None] == ref["groups"][None, :]) | (
            q["pixel_hash"][start:stop, None] == ref["pixel_hash"][None, :]
        )
        d.masked_fill_(torch.tensor(mask, device=DEVICE), float("inf"))
        dist, idx = d.topk(K, largest=False)
        assert torch.isfinite(dist).all()
        scale = dist.median(1).values.clamp_min(1e-6)
        w = torch.softmax(-dist / scale[:, None], dim=1)
        inds.append(idx.cpu().numpy())
        weights.append(w.cpu().numpy())
    inds = np.concatenate(inds)
    weights = np.concatenate(weights).astype(np.float64)
    weights /= weights.sum(1, keepdims=True)
    P = sp.csr_matrix(
        (weights.ravel(), (np.repeat(np.arange(len(inds)), K), inds.ravel())),
        shape=(len(inds), len(ref["paths"])),
    )
    assert np.allclose(np.asarray(P.sum(1)).ravel(), 1)
    return P


def diffuse(P, nl, Y):
    W = P[nl:, nl:]
    known = P[nl:, :nl] @ Y
    H = np.zeros((P.shape[0] - nl, Y.shape[1]), np.float64)
    for iteration in range(200):
        nxt = ALPHA * (known + W @ H)
        error = float(np.max(np.abs(nxt - H)))
        H = nxt
        if error < 1e-9:
            break
    assert error < 1e-8 and np.isfinite(H).all()
    assert H.min() > -1e-8 and np.max(H.sum(1)) <= 1 + 1e-7
    return np.concatenate([Y, H]), {
        "iterations": iteration + 1,
        "fixed_point_error": error,
    }


def take(f, ix):
    return {k: v[ix] for k, v in f.items()}


def join(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def norm(c, t, ix):
    mu = float(c[ix].mean())
    sd = float(c[ix].std())
    assert sd > 1e-6
    return (c - mu) / sd, (t - mu) / sd, dict(mean=mu, std=sd)


def train_head(ff, raw, gi, steps, out):
    nl = len(ff["train"]["paths"])
    nu = len(ff["pool"]["paths"])
    fit = torch.tensor(design(join(ff["train"], ff["pool"])), device=DEVICE)
    rng = np.random.default_rng(HS)
    ls = rng.integers(nl, size=(30, steps, 64))
    needed = 30 * steps * 64
    us = np.concatenate([rng.permutation(nu) for _ in range(math.ceil(needed / nu))])[
        :needed
    ].reshape(30, steps, 64)
    assert np.bincount(us.ravel(), minlength=nu).min() > 0
    schedule = np.concatenate([ls, nl + us], 2)
    mu = float(raw["calibration"][gi].mean())
    sd = float(raw["calibration"][gi].std())
    assert sd > 1e-6
    y = torch.tensor(
        (np.concatenate([raw["train"], raw["pool"]]) - mu) / sd,
        dtype=torch.float32,
        device=DEVICE,
    )
    torch.manual_seed(HS)
    net = torch.nn.Sequential(
        torch.nn.Linear(5120, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)
    ).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30)
    hist = []
    for ep in range(30):
        losses = []
        for ix in schedule[ep]:
            loss = F.mse_loss(net(fit[ix]).flatten(), y[ix])
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sch.step()
        hist.append(dict(epoch=ep + 1, loss=float(np.mean(losses))))
        dump(out / "history.json", hist)
        dump(
            out / "progress.json",
            dict(stage="training_PC_head", epoch=ep + 1, time=time.time()),
        )
    net.eval()
    c = predict(net, torch.tensor(design(ff["calibration"]), device=DEVICE))
    t = predict(net, torch.tensor(design(ff["test"]), device=DEVICE))
    torch.save(
        dict(
            head=net.state_dict(),
            target_mean=mu,
            target_std=sd,
            head_seed=HS,
            epochs=30,
            steps_per_epoch=steps,
        ),
        out / "head.pt",
    )
    dump(
        out / "schedule.json",
        dict(
            sha256=hashlib.sha256(schedule.tobytes()).hexdigest(),
            minimum_unlabeled_exposures=int(
                np.bincount(us.ravel(), minlength=nu).min()
            ),
            L=nl,
            U=nu,
            steps_per_epoch=steps,
        ),
    )
    return c, t


def rewired(P, q, ref, seed):
    rng = np.random.default_rng(seed)
    A = P.copy()
    before = A.indices.copy()
    count = np.diff(A.indptr)
    assert np.all(count == K)
    ids = A.indices.reshape(-1, K)
    sets = [set(x) for x in ids]
    nr = len(ids)
    accepted = 0
    attempted = 0
    goal = 2 * len(before)
    limit = 10 * len(before)
    for chunk in range(0, limit, 10000):
        pairs = rng.integers(len(before), size=(min(10000, limit - chunk), 2))
        for ai, bi in pairs:
            attempted += 1
            r, a = divmod(int(ai), K)
            s, b = divmod(int(bi), K)
            u = int(ids[r, a])
            v = int(ids[s, b])
            if r == s or u == v or v in sets[r] or u in sets[s]:
                continue
            if (
                q["groups"][r] == ref["groups"][v]
                or q["groups"][s] == ref["groups"][u]
                or q["pixel_hash"][r] == ref["pixel_hash"][v]
                or q["pixel_hash"][s] == ref["pixel_hash"][u]
            ):
                continue
            sets[r].remove(u)
            sets[r].add(v)
            sets[s].remove(v)
            sets[s].add(u)
            ids[r, a] = v
            ids[s, b] = u
            accepted += 1
            if accepted >= goal:
                break
        if accepted >= goal:
            break
    assert accepted == goal
    assert np.array_equal(
        np.bincount(before, minlength=P.shape[1]),
        np.bincount(A.indices, minlength=P.shape[1]),
    )
    assert np.array_equal(A.data, P.data)
    assert np.allclose(np.asarray(A.sum(1)).ravel(), 1)
    qi = np.repeat(np.arange(nr), K)
    assert np.all(q["groups"][qi] != ref["groups"][A.indices])
    assert np.all(q["pixel_hash"][qi] != ref["pixel_hash"][A.indices])
    assert all(len(x) == K for x in sets)
    return A, {
        "seed": seed,
        "accepted_swaps": accepted,
        "attempted_swaps": attempted,
        "edge_slot_changed_fraction": float((before != A.indices).mean()),
        "row_degree_and_row_weights_preserved": True,
        "column_unweighted_degree_preserved": True,
        "weighted_column_strength_not_preserved": True,
        "same_source_and_pixel_exclusions_preserved": True,
    }
