"""Train a known-label ResNet50 for one FGSRCS class split."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights

from fgship_core import dump, rows, sha
from fgship_features import cache_images, prepare


def train(data, output, fold, seed, weights):
    dest = output / f"b{seed}" / f"fold{fold}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "final.pt").exists():
        raise FileExistsError("Final checkpoint exists; use another output directory.")
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    split = data / f"fold{fold}"
    protocol = json.loads((split / "protocol.json").read_text())
    assert sha(split / "train.jsonl") == protocol["manifests"]["train.jsonl"]
    tr = rows(split / "train.jsonl")
    ci = json.loads((split / "class_to_idx.json").read_text())
    model = resnet50(weights=None if weights else ResNet50_Weights.IMAGENET1K_V2)
    if weights:
        model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True)
        )
    model.fc = torch.nn.Linear(2048, len(ci))
    model = model.cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
    config = dict(
        seed=seed,
        fold=fold,
        epochs=30,
        batch_size=64,
        lr=5e-5,
        weight_decay=0.01,
        backbone="ResNet50 ImageNet1K V2",
        train_manifest_sha256=sha(split / "train.jsonl"),
        unlabeled_images_used=False,
        precision="bfloat16 autocast",
        checkpoint_selection="fixed final epoch",
        torch=str(torch.__version__),
    )
    images = cache_images(tr, data / "crops")
    targets = torch.tensor([ci[r["label"]] for r in tr])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, targets),
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    dump(dest / "config.json", config)
    history = []
    for epoch in range(30):
        model.train()
        loss_sum = 0.0
        n = 0
        start = time.time()
        for im, y in loader:
            x = prepare(im, "cuda", True)
            y = y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            n += len(y)
        scheduler.step()
        history.append(
            dict(epoch=epoch + 1, loss=loss_sum / n, seconds=time.time() - start)
        )
        dump(dest / "history.json", history)
        print(history[-1], flush=True)
    torch.save(
        dict(
            model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            config=config,
        ),
        dest / "final.pt",
    )
    dump(
        dest / "training_complete.json",
        dict(model_sha256=sha(dest / "final.pt"), config=config),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/fgsrcs"))
    p.add_argument("--output", type=Path, default=Path("runs"))
    p.add_argument("--fold", type=int, required=True, choices=range(4))
    p.add_argument("--seed", type=int, default=27042)
    p.add_argument(
        "--weights",
        type=Path,
        help="Optional local torchvision ImageNet1K V2 state_dict",
    )
    a = p.parse_args()
    train(a.data, a.output, a.fold, a.seed, a.weights)
