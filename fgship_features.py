"""Shared ResNet50 global and three-strip features for FGSRCS."""

import concurrent.futures, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

SOURCE = Path("data/crops")
DEVICE = "cuda"


def load(r):
    with Image.open(SOURCE / r["path"]) as im:
        orig = np.array(im.convert("RGB"))
    pix = hashlib.sha256(str(orig.shape).encode() + orig.tobytes()).hexdigest()
    a = np.array(Image.fromarray(orig).resize((224, 224), Image.Resampling.BILINEAR))
    yy, xx = np.where(a.max(2) > 10)
    mask = np.zeros((3, 224, 224), np.float32)
    if len(xx) < 3:
        mask[:] = 1 / 3
    else:
        xy = np.stack([xx, yy], 1)
        v = np.linalg.eigh(np.cov(xy.T))[1][:, -1]
        if v[np.argmax(abs(v))] < 0:
            v = -v
        t = xy @ v
        idx = np.minimum(((t - t.min()) / max(np.ptp(t), 1) * 3).astype(int), 2)
        mask[idx, yy, xx] = 1
        for k in range(3):
            if not mask[k].any():
                mask[k, yy, xx] = 1
    return a.transpose(2, 0, 1), mask.reshape(3, 14, 16, 14, 16).mean((2, 4)), pix


@torch.inference_mode()
def extract(model, rr):
    zs = []
    gs = []
    ls = []
    raws = []
    hashes = []
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for i in range(0, len(rr), 64):
            items = list(pool.map(load, rr[i : i + 64]))
            a, m, h = zip(*items)
            hashes.extend(h)
            x = torch.tensor(np.stack(a), device=DEVICE, dtype=torch.float32) / 255
            x = (
                x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            ) / x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            x = x.contiguous(memory_format=torch.channels_last)
            mm = torch.tensor(np.stack(m), device=DEVICE)
            v = model.maxpool(model.relu(model.bn1(model.conv1(x))))
            v = model.layer3(model.layer2(model.layer1(v)))
            ll = F.normalize(
                torch.einsum("bchw,bphw->bpc", v, mm)
                / mm.sum((2, 3)).clamp_min(1e-8)[:, :, None],
                dim=2,
            )
            gg = model.avgpool(model.layer4(v)).flatten(1)
            zz = model.fc(gg)
            raws.append(gg.cpu().numpy())
            zs.append(zz.cpu().numpy())
            gs.append(F.normalize(gg, dim=1).cpu().numpy())
            ls.append(ll.cpu().numpy())
    return dict(
        features=np.concatenate(raws),
        logits=np.concatenate(zs),
        global_features=np.concatenate(gs),
        local_features=np.concatenate(ls),
        pixel_hash=np.array(hashes),
        paths=np.array([r["path"] for r in rr]),
    )


def cache_images(rows, source):
    data = np.empty((len(rows), 3, 224, 224), dtype=np.uint8)

    def read(r):
        with Image.open(source / r["path"]) as im:
            return np.asarray(
                im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            ).transpose(2, 0, 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, im in enumerate(pool.map(read, rows)):
            data[i] = im
    return torch.from_numpy(data)


def prepare(x, device, augment=False):
    x = x.to(device=device, dtype=torch.float32, non_blocking=True).div_(255)
    if augment:
        # Full-image flips preserve the entire ship, unlike aggressive random crops.
        mask = torch.rand(len(x), device=device) < 0.5
        x = torch.where(mask[:, None, None, None], x.flip(-1), x)
        mask = torch.rand(len(x), device=device) < 0.5
        x = torch.where(mask[:, None, None, None], x.flip(-2), x)
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)
