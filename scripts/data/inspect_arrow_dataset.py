import argparse
import json
import time
from pathlib import Path

import numpy as np
from gluonts.dataset.common import FileDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect GluonTS Arrow datasets.")
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--freq", default="h")
    parser.add_argument("--min_length", type=int, default=288)
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs/logs")
    return parser.parse_args()


def inspect_path(path: Path, args) -> dict:
    dataset = FileDataset(path=path, freq=args.freq)
    count = 0
    too_short = 0
    nan_count = 0
    inf_count = 0
    lengths = []
    examples = []
    for entry in dataset:
        target = np.asarray(entry["target"], dtype=np.float32)
        count += 1
        lengths.append(len(target))
        nan_count += int(np.isnan(target).sum())
        inf_count += int(np.isinf(target).sum())
        if len(target) < args.min_length:
            too_short += 1
        if len(examples) < args.max_examples:
            examples.append(
                {
                    "index": count - 1,
                    "length": int(len(target)),
                    "mean": float(np.nanmean(target)),
                    "std": float(np.nanstd(target)),
                    "min": float(np.nanmin(target)),
                    "max": float(np.nanmax(target)),
                }
            )
    return {
        "path": str(path),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_series_checked": count,
        "min_length_observed": int(min(lengths)) if lengths else None,
        "max_length_observed": int(max(lengths)) if lengths else None,
        "mean_length": float(np.mean(lengths)) if lengths else None,
        "required_min_length": args.min_length,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "too_short_count": too_short,
        "examples": examples,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for raw_path in args.paths:
        path = Path(raw_path)
        report = inspect_path(path, args)
        reports.append(report)
        out = output_dir / f"inspect_{path.stem}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    combined = output_dir / "inspect_arrow_summary.json"
    combined.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
