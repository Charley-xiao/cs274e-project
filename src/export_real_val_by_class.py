from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import EuroSAT


EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./data")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_class", type=int, default=1000000)
    return ap.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = EuroSAT(root=args.root, download=True)

    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)

    n_val = int(len(indices) * args.val_ratio)
    val_idx = indices[:n_val]

    counters = {i: 0 for i in range(10)}

    for idx in val_idx:
        img, y = ds[idx]   # PIL image, label
        if counters[y] >= args.max_per_class:
            continue

        class_dir = Path(args.outdir) / f"class_{y}"
        ensure_dir(class_dir)

        out_path = class_dir / f"{counters[y]:05d}.png"
        img.save(out_path)
        counters[y] += 1

    print("[done] real validation images exported by class.")
    for k, v in counters.items():
        print(k, EUROSAT_CLASSES[k], v)


if __name__ == "__main__":
    main()