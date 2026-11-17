"""Prepare FGSRCS object crops and four independent-query class splits."""

import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from fgship_core import dump, sha


def write_rows(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def crop_scene(task):
    source, output, split, annotation = task
    image_path = source / split / "images" / (annotation.stem + ".png")
    records = []
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        for line, text in enumerate(annotation.read_text().splitlines(), 1):
            if not text.strip():
                continue
            fields = text.split()
            assert len(fields) == 10, annotation
            xy = list(map(float, fields[:8]))
            assert all(math.isfinite(x) for x in xy)
            box = (
                max(0, math.floor(min(xy[::2]))),
                max(0, math.floor(min(xy[1::2]))),
                min(width, math.ceil(max(xy[::2]))),
                min(height, math.ceil(max(xy[1::2]))),
            )
            assert box[2] > box[0] and box[3] > box[1]
            name = f"{split}_{annotation.stem}_{line:03d}.png"
            crop = image.crop(box)
            crop.save(output / "crops" / name)
            pixels = np.asarray(crop)
            resized = np.asarray(crop.resize((224, 224), Image.Resampling.BILINEAR))
            records.append(
                dict(
                    path=name,
                    sample_id=name[:-4],
                    label=fields[8],
                    image_group=f"{split}/{annotation.stem}",
                    difficulty=int(fields[9]),
                    crop_box=box,
                    crop_sha256=sha(output / "crops" / name),
                    pixel_hash=hashlib.sha256(
                        str(pixels.shape).encode() + pixels.tobytes()
                    ).hexdigest(),
                    resized_hash=hashlib.sha256(resized.tobytes()).hexdigest(),
                )
            )
    return records


def prepare(source, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Use a new output directory; prepared data are not overwritten."
        )
    (output / "crops").mkdir(parents=True)
    tasks = [
        (source, output, s, p)
        for s in ["train", "val"]
        for p in sorted((source / s / "annfiles_lv2").glob("*.txt"))
    ]
    assert len(tasks) == 4500, "Expected official FGSRCS train + val (4500 images)."
    records = []
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for rows in pool.map(crop_scene, tasks):
            records.extend(rows)
    labels = sorted({r["label"] for r in records})
    assert len(records) == 11187 and len(labels) == 17
    order = np.random.default_rng(27142).permutation(labels).tolist()
    unknowns = [order[:4], order[4:8], order[8:12], order[12:]]
    calgroups = {
        r["image_group"]
        for r in records
        if r["image_group"].startswith("train/")
        and int(
            hashlib.sha256(
                ("fgsrcs-cal-31842/" + r["image_group"]).encode()
            ).hexdigest()[:8],
            16,
        )
        % 5
        == 0
    }
    for fold, unknown in enumerate(unknowns):
        dest = output / f"fold{fold}"
        dest.mkdir()
        ci = {l: i for i, l in enumerate(l for l in labels if l not in unknown)}
        train = [
            r
            for r in records
            if r["image_group"].startswith("train/")
            and r["image_group"] not in calgroups
            and r["label"] in ci
        ]
        cal = [r for r in records if r["image_group"] in calgroups and r["label"] in ci]
        test = [r for r in records if r["image_group"].startswith("val/")]
        lp = {r["path"] for r in train}
        # Unknown-class labels are removed before writing the fitting manifests.
        pool = [
            r
            for r in records
            if r["image_group"] not in calgroups
            and r["path"] not in lp
            and r["image_group"].startswith("train/")
        ]
        for key in ["path", "image_group", "pixel_hash", "resized_hash"]:
            excluded = {r[key] for r in test}
            pool = [r for r in pool if r[key] not in excluded]
            assert not {r[key] for r in train} & excluded, (fold, key)
            assert not {r[key] for r in train + pool} & {r[key] for r in cal}, (
                fold,
                key,
            )
            assert not {r[key] for r in cal} & excluded, (fold, key)
        unlabeled = lambda rr: [
            {k: v for k, v in r.items() if k != "label"} for r in rr
        ]
        for name, rr in [
            ("train", train),
            ("calibration", cal),
            ("pool", unlabeled(pool)),
            ("test_paths", unlabeled(test)),
            ("test_labels", test),
        ]:
            write_rows(dest / (name + ".jsonl"), rr)
        decision = []
        for label in sorted(ci):
            ids = [i for i, r in enumerate(cal) if r["label"] == label]
            ids.sort(
                key=lambda i: hashlib.sha256(
                    ("decision31843/" + cal[i]["sample_id"]).encode()
                ).hexdigest()
            )
            decision.extend(ids[: max(1, len(ids) // 2)])
        decision = np.array(sorted(decision))
        dg = {cal[i]["image_group"] for i in decision}
        normalization = np.array(
            [i for i, r in enumerate(cal) if r["image_group"] not in dg]
        )
        assert len(normalization) > 30
        assert set(r["label"] for r in train) == set(ci) == set(r["label"] for r in cal)
        np.savez(
            dest / "calibration_indices.npz",
            decision=decision,
            normalization=normalization,
        )
        dump(dest / "class_to_idx.json", ci)
        dump(
            dest / "protocol.json",
            dict(
                fold=fold,
                unknown_classes=unknown,
                known_classes=list(ci),
                counts=dict(
                    train=len(train),
                    pool=len(pool),
                    calibration=len(cal),
                    test=len(test),
                ),
                head_steps_per_epoch=49,
                test_in_reference=False,
                manifests={p.name: sha(p) for p in sorted(dest.glob("*.jsonl"))},
            ),
        )
    dump(
        output / "complete.json",
        dict(
            dataset="FGSRCS",
            objects=len(records),
            classes=labels,
            class_partition_seed=27142,
            splits=4,
            test_partition="official val",
            protocol_sha256={
                f"fold{i}": sha(output / f"fold{i}/protocol.json") for i in range(4)
            },
        ),
    )
    print("Prepared FGSRCS:", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", default=Path("data/fgsrcs"), type=Path)
    a = p.parse_args()
    prepare(a.source, a.output)
