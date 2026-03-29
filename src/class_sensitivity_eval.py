from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch_fidelity import calculate_metrics


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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", type=str, required=True)
    ap.add_argument("--fake_root", type=str, required=True)
    ap.add_argument("--traj_root", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["euler", "midpoint", "heun", "rk4", "ab2", "rk23-adaptive"],
    )
    ap.add_argument("--classes", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--fid", action="store_true", default=True)
    ap.add_argument("--kid", action="store_true", default=False)
    return ap.parse_args()


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_geom_for_method(traj_root: str, method: str) -> pd.DataFrame:
    arr = np.load(Path(traj_root) / f"{method}.npz", allow_pickle=True)
    labels = arr["labels"]
    path_length = arr["path_length"]
    curvature = arr["integrated_curvature"]

    return pd.DataFrame(
        {
            "label": labels.astype(int),
            "path_length": path_length.astype(float),
            "integrated_curvature": curvature.astype(float),
        }
    )


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    rows = []

    for method in args.methods:
        print(f"\n=== evaluating {method} ===")
        geom_df = load_geom_for_method(args.traj_root, method)

        for cls in args.classes:
            real_dir = Path(args.real_root) / f"class_{cls}"
            fake_dir = Path(args.fake_root) / method / f"class_{cls}"

            if not real_dir.exists():
                raise FileNotFoundError(f"Missing real dir: {real_dir}")
            if not fake_dir.exists():
                raise FileNotFoundError(f"Missing fake dir: {fake_dir}")

            metrics = calculate_metrics(
                input1=str(real_dir),
                input2=str(fake_dir),
                cuda=True,
                fid=args.fid,
                kid=args.kid,
                isc=False,
                verbose=False,
                samples_find_deep=True,
                dataloader_num_workers=0,
            )

            sub = geom_df[geom_df["label"] == cls]

            row = {
                "sampler": method,
                "class_id": cls,
                "class_name": EUROSAT_CLASSES[cls],
                "n_real": len(list(real_dir.glob("*.png"))),
                "n_fake": len(list(fake_dir.glob("*.png"))),
                "fid": float(metrics["frechet_inception_distance"]) if "frechet_inception_distance" in metrics else np.nan,
                "kid_mean": float(metrics["kernel_inception_distance_mean"]) if "kernel_inception_distance_mean" in metrics else np.nan,
                "path_length_mean": float(sub["path_length"].mean()),
                "path_length_std": float(sub["path_length"].std(ddof=1)),
                "curvature_mean": float(sub["integrated_curvature"].mean()),
                "curvature_std": float(sub["integrated_curvature"].std(ddof=1)),
            }
            rows.append(row)
            print(method, cls, row["fid"])

    df = pd.DataFrame(rows)
    df.to_csv(Path(args.outdir) / "class_metrics.csv", index=False)

    # sensitivity summaries
    sens = (
        df.groupby(["class_id", "class_name"], as_index=False)
        .agg(
            fid_range=("fid", lambda x: float(np.max(x) - np.min(x))),
            path_length_range=("path_length_mean", lambda x: float(np.max(x) - np.min(x))),
            curvature_range=("curvature_mean", lambda x: float(np.max(x) - np.min(x))),
        )
        .sort_values("fid_range", ascending=False)
    )
    sens.to_csv(Path(args.outdir) / "class_sensitivity.csv", index=False)

    with open(Path(args.outdir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "most_fid_sensitive_class": sens.iloc[0]["class_name"],
                "max_fid_range": float(sens.iloc[0]["fid_range"]),
            },
            f,
            indent=2,
        )

    print("[done] saved:")
    print(Path(args.outdir) / "class_metrics.csv")
    print(Path(args.outdir) / "class_sensitivity.csv")


if __name__ == "__main__":
    main()