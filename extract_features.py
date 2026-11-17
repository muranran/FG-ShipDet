"""Extract frozen FP32 features; never fit on pool, calibration, or test images."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet50

import fgship_features as features
from fgship_core import dump, rows, sha


def extract(data, output, fold, seed, device):
    split = data / f"fold{fold}"
    dest = output / f"b{seed}" / f"fold{fold}"
    fd = dest / "features"
    fd.mkdir(exist_ok=True)
    if (fd / "complete.json").exists():
        raise FileExistsError("Feature extraction already complete.")
    ci = json.loads((split / "class_to_idx.json").read_text())
    checkpoint = torch.load(dest / "final.pt", map_location="cpu", weights_only=False)
    assert checkpoint["config"]["unlabeled_images_used"] is False
    model = resnet50(weights=None, num_classes=len(ci))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval().to(memory_format=torch.channels_last)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    features.SOURCE = data / "crops"
    features.DEVICE = device
    manifests = json.loads((split / "protocol.json").read_text())["manifests"]
    all_features = {}
    for name in ["train", "pool", "calibration", "test"]:
        filename = ("test_paths" if name == "test" else name) + ".jsonl"
        assert sha(split / filename) == manifests[filename]
        rr = rows(split / filename)
        ff = features.extract(model, rr)
        ff["groups"] = np.array([r["image_group"] for r in rr])
        ff["resized_hash"] = np.array([r["resized_hash"] for r in rr])
        assert np.array_equal(ff["pixel_hash"], [r["pixel_hash"] for r in rr])
        np.savez_compressed(fd / (name + ".npz"), **ff)
        all_features[name] = ff
        print(name, len(rr), flush=True)
    for key in ["paths", "groups", "pixel_hash", "resized_hash"]:
        fit = set(all_features["train"][key]) | set(all_features["pool"][key])
        assert fit.isdisjoint(all_features["calibration"][key])
        assert fit.isdisjoint(all_features["test"][key])
    dump(
        fd / "complete.json",
        dict(
            model_sha256=sha(dest / "final.pt"),
            sha256={p.name: sha(p) for p in sorted(fd.glob("*.npz"))},
            test_labels_read=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/fgsrcs"))
    p.add_argument("--output", type=Path, default=Path("runs"))
    p.add_argument("--fold", type=int, required=True, choices=range(4))
    p.add_argument("--seed", type=int, default=27042)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    extract(a.data, a.output, a.fold, a.seed, a.device)
