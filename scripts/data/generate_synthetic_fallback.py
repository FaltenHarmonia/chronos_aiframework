import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from gluonts.dataset.arrow import ArrowWriter


SCHEMA_VERSION = "chronos-synthetic-fallback-v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic fallback dataset for Chronos reproduction.")
    parser.add_argument("--output_path", default="data/processed/synthetic_fallback_50k.arrow")
    parser.add_argument("--num_series", type=int, default=50_000)
    parser.add_argument("--min_length", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_level", default="fallback")
    parser.add_argument("--compression", default="lz4")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_series(index: int, args) -> np.ndarray:
    rng = np.random.default_rng(args.seed + index * 104729)
    length = int(rng.integers(args.min_length, args.max_length + 1))
    t = np.arange(length, dtype=np.float32)
    level = rng.uniform(-3.0, 3.0)
    trend = rng.uniform(-0.008, 0.008) * t
    seasonal_1 = rng.uniform(0.2, 2.5) * np.sin(2.0 * math.pi * t / rng.choice([12, 24, 48]) + rng.uniform(0, 6.28))
    seasonal_2 = rng.uniform(0.0, 1.5) * np.sin(2.0 * math.pi * t / rng.choice([96, 168, 336]) + rng.uniform(0, 6.28))
    random_walk = np.cumsum(rng.normal(0.0, rng.uniform(0.005, 0.04), size=length)).astype(np.float32)
    noise = rng.normal(0.0, rng.uniform(0.03, 0.2), size=length)
    spikes = (rng.random(length) < rng.uniform(0.01, 0.06)).astype(np.float32) * rng.uniform(-4.0, 4.0, size=length)
    level_shift = np.zeros(length, dtype=np.float32)
    if rng.random() < 0.35:
        level_shift[int(length * rng.uniform(0.3, 0.7)) :] = rng.uniform(-2.0, 2.0)
    y = level + trend + seasonal_1 + seasonal_2 + random_walk + noise + spikes + level_shift
    return np.asarray(y, dtype=np.float32)


def atomic_write_json(path: Path, payload: dict):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_suffix(".summary.json")
    progress_path = output_path.with_suffix(".progress.json")

    if output_path.exists() and not args.overwrite:
        print(f"{output_path} already exists. Use --overwrite to regenerate.")
        return

    records = []
    lengths = []
    for idx in range(args.num_series):
        if idx % 1000 == 0:
            atomic_write_json(
                progress_path,
                {
                    "status": "generating",
                    "dataset_name": "synthetic_fallback",
                    "data_level": args.data_level,
                    "written": idx,
                    "target": args.num_series,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
        target = make_series(idx, args)
        lengths.append(len(target))
        records.append({"start": np.datetime64("2000-01-01 00:00", "s"), "target": target})

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    ArrowWriter(compression=args.compression).write_to_file(records, path=tmp_path)
    os.replace(tmp_path, output_path)

    summary = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "dataset_name": "synthetic_fallback",
        "data_level": args.data_level,
        "num_series": args.num_series,
        "min_length_requested": args.min_length,
        "max_length_requested": args.max_length,
        "min_length_observed": int(min(lengths)),
        "max_length_observed": int(max(lengths)),
        "mean_length": float(np.mean(lengths)),
        "seed": args.seed,
        "output_path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write_json(summary_path, summary)
    atomic_write_json(progress_path, {"status": "complete", "summary": str(summary_path)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
