from __future__ import annotations
import argparse, json
from pathlib import Path
from torch_fidelity import calculate_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=str, required=True, help="Real images dir.")
    ap.add_argument("--fake", type=str, required=True, help="Generated images dir.")
    ap.add_argument("--outfile", type=str, default=None)
    args = ap.parse_args()

    assert Path(args.real).is_dir(), f"Real dir not found: {args.real}"
    assert Path(args.fake).is_dir(), f"Fake dir not found: {args.fake}"

    metrics = calculate_metrics(
        input1=args.fake, input2=args.real,
        fid=True, isc=True, verbose=True, cuda=True,
        samples_find_deep=True,
    )
    print(metrics)

    if args.outfile:
        with open(args.outfile, "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
