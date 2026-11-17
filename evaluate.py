"""Evaluate frozen scores after every requested backbone and class split is complete."""

import argparse
import json
from pathlib import Path

import numpy as np
from fgship_core import dump, rows, sha


def evaluate(data, output, seeds, setting):
    for seed in seeds:
        for fold in range(4):
            dest = output / f"b{seed}" / f"fold{fold}" / setting
            done = json.loads((dest / "complete.json").read_text())
            assert done["status"] == "scores_frozen"
            for name, h in done["sha256"].items():
                assert sha(dest / name) == h
    # Test labels are first opened below, after the full requested scoring barrier.
    records = []
    for seed in seeds:
        for fold in range(4):
            split = data / f"fold{fold}"
            base = output / f"b{seed}" / f"fold{fold}"
            dest = base / setting
            rr = rows(split / "test_labels.jsonl")
            ci = json.loads((split / "class_to_idx.json").read_text())
            tf = np.load(base / "features/test.npz")
            assert np.array_equal(tf["paths"], [r["path"] for r in rr])
            labels = np.array([r["label"] for r in rr])
            y = np.array([ci.get(l, -1) for l in labels])
            km = y >= 0
            scores = np.load(dest / "scores.npz")
            th = json.loads((dest / "thresholds.json").read_text())
            di = np.load(split / "calibration_indices.npz")["decision"]
            for name, budgets in th.items():
                pred = (
                    scores["pred_test_" + name]
                    if setting == "posthoc"
                    else tf["logits"].argmax(1)
                )
                cc = (
                    scores["calibration_" + name]
                    if setting == "posthoc"
                    else scores["cal_" + name]
                )
                tt = scores["test_" + name] if setting == "posthoc" else scores[name]
                arm, head = (
                    (name.rsplit("_s", 1)[0], int(name.rsplit("_s", 1)[1]))
                    if "_s" in name
                    else (name, None)
                )
                for budget, tau in budgets.items():
                    assert tau == float(
                        np.quantile(cc[di], 1 - float(budget), method="higher")
                    )
                    rejected = tt > tau
                    pc = {
                        cl: float(rejected[labels == cl].mean())
                        for cl in sorted(set(labels[~km]))
                    }
                    records.append(
                        dict(
                            backbone_seed=seed,
                            fold=fold,
                            head_seed=head,
                            arm=arm,
                            budget=budget,
                            unknown_per_class=pc,
                            unknown_macro=float(np.mean(list(pc.values()))),
                            known_frr=float(rejected[km].mean()),
                            known_correct_accepted_macro=float(
                                np.mean(
                                    [
                                        ((pred == y) & ~rejected)[y == k].mean()
                                        for k in range(len(ci))
                                    ]
                                )
                            ),
                        )
                    )
    aggregate = {}
    for arm in sorted({r["arm"] for r in records}):
        aggregate[arm] = {}
        for budget in ["0.01", "0.05", "0.1"]:
            by_seed = {}
            for seed in seeds:
                rr = [
                    r
                    for r in records
                    if r["arm"] == arm
                    and r["budget"] == budget
                    and r["backbone_seed"] == seed
                ]
                classes = {}
                for row in rr:
                    for cl, v in row["unknown_per_class"].items():
                        classes.setdefault(cl, []).append(v)
                assert len(classes) == 17
                ur = float(np.mean([np.mean(v) for v in classes.values()]))
                cca = float(np.mean([r["known_correct_accepted_macro"] for r in rr]))
                by_seed[str(seed)] = dict(
                    unknown_macro=ur,
                    known_frr=float(np.mean([r["known_frr"] for r in rr])),
                    known_correct_accepted_macro=cca,
                    H=2 * ur * cca / (ur + cca) if ur + cca else 0.0,
                )
            keys = ["unknown_macro", "known_frr", "known_correct_accepted_macro", "H"]
            aggregate[arm][budget] = dict(
                by_backbone=by_seed,
                mean={
                    k: float(np.mean([v[k] for v in by_seed.values()])) for k in keys
                },
                sd={
                    k: float(np.std([v[k] for v in by_seed.values()], ddof=1))
                    if len(seeds) > 1
                    else None
                    for k in keys
                },
            )
    dump(output / (setting + "_per_run_results.json"), records)
    dump(
        output / (setting + "_results.json"),
        dict(aggregate=aggregate, seeds=seeds, setting=setting),
    )
    print("Evaluated", len(records), "metric rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/fgsrcs"))
    p.add_argument("--output", type=Path, default=Path("runs"))
    p.add_argument("--seeds", type=int, nargs="+", default=[27042, 27043, 27044])
    p.add_argument(
        "--setting", default="method", help="method, posthoc, or pool0.5_seed23001 etc."
    )
    a = p.parse_args()
    evaluate(a.data, a.output, a.seeds, a.setting)
