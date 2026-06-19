import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from gluonts.dataset.arrow import ArrowWriter


SCHEMA_VERSION = "chronos-repro-data-v1"


@dataclass
class SeriesRecord:
    item_id: str
    freq: str
    target: list[float]
    source: str
    split_points: dict
    generator: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare robust Chronos reproduction datasets.")
    parser.add_argument("--dataset", choices=["synthetic"], default="synthetic")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--num_series", type=int, default=2000)
    parser.add_argument("--min_length", type=int, default=384)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--prediction_length", type=int, default=32)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--freq", default="h")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_train_series", type=int, default=500)
    parser.add_argument("--min_eval_series", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("a", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def validate_args(args: argparse.Namespace) -> None:
    if args.num_series <= 0:
        raise ValueError("--num_series must be positive")
    if args.min_length < args.context_length + args.prediction_length:
        raise ValueError("--min_length must be at least context_length + prediction_length")
    if args.max_length < args.min_length:
        raise ValueError("--max_length must be >= min_length")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train_ratio must be in (0, 1)")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val_ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train_ratio + --val_ratio must be < 1")


def synthetic_series(index: int, args: argparse.Namespace) -> SeriesRecord:
    rng = np.random.default_rng(args.seed + index * 9973)
    length = int(rng.integers(args.min_length, args.max_length + 1))
    t = np.arange(length, dtype=np.float32)

    family = int(index % 5)
    scale = float(rng.uniform(0.5, 3.0))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    level = float(rng.uniform(-2.0, 2.0))
    trend_slope = float(rng.uniform(-0.01, 0.01))
    noise_scale = float(rng.uniform(0.03, 0.18))

    daily = scale * np.sin(2.0 * math.pi * t / 24.0 + phase)
    weekly = 0.6 * scale * np.sin(2.0 * math.pi * t / (24.0 * 7.0) + phase / 2.0)
    trend = trend_slope * t
    noise = noise_scale * rng.standard_normal(length)

    if family == 0:
        values = level + daily + weekly + trend + noise
        pattern = "daily_weekly_trend"
    elif family == 1:
        values = level + 0.5 * daily + trend + 0.04 * np.maximum(t - length * 0.55, 0) + noise
        pattern = "piecewise_trend"
    elif family == 2:
        pulses = (rng.random(length) < 0.08).astype(np.float32) * rng.uniform(1.0, 5.0, length)
        values = level + 0.25 * daily + pulses + noise
        pattern = "intermittent_spikes"
    elif family == 3:
        regime = np.where(t > length * 0.5, rng.uniform(-1.5, 1.5), 0.0)
        values = level + daily + regime + noise
        pattern = "level_shift"
    else:
        values = level + 0.015 * np.square((t - length / 2.0) / max(length, 1)) * length + daily + noise
        pattern = "curved_trend"

    train_end = max(args.context_length + args.prediction_length, int(length * args.train_ratio))
    val_end = max(train_end + args.prediction_length, int(length * (args.train_ratio + args.val_ratio)))
    val_end = min(val_end, length - args.prediction_length)

    return SeriesRecord(
        item_id=f"series_{index:06d}",
        freq=args.freq,
        target=np.asarray(values, dtype=np.float32).round(6).tolist(),
        source=args.dataset,
        split_points={"train_end": train_end, "val_end": val_end, "length": length},
        generator={
            "pattern": pattern,
            "family": family,
            "scale": scale,
            "phase": phase,
            "level": level,
            "trend_slope": trend_slope,
            "noise_scale": noise_scale,
        },
    )


def ensure_raw_series(args: argparse.Namespace, output_dir: Path) -> tuple[Path, dict]:
    raw_path = output_dir / "raw_series.jsonl"
    progress_path = output_dir / "prepare_progress.json"

    if args.force and raw_path.exists():
        raw_path.unlink()

    existing = load_jsonl(raw_path)
    existing_ids = {row["item_id"] for row in existing}
    start_count = len(existing)

    atomic_write_json(
        progress_path,
        {
            "status": "generating",
            "schema_version": SCHEMA_VERSION,
            "dataset": args.dataset,
            "existing_series": start_count,
            "target_series": args.num_series,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    batch = []
    for idx in range(args.num_series):
        item_id = f"series_{idx:06d}"
        if item_id in existing_ids:
            continue
        batch.append(asdict(synthetic_series(idx, args)))
        if len(batch) >= 100:
            append_jsonl(raw_path, batch)
            batch.clear()
            atomic_write_json(
                progress_path,
                {
                    "status": "generating",
                    "schema_version": SCHEMA_VERSION,
                    "written_series": idx + 1,
                    "target_series": args.num_series,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
    if batch:
        append_jsonl(raw_path, batch)

    rows = load_jsonl(raw_path)
    if len(rows) < args.num_series:
        raise RuntimeError(f"Only prepared {len(rows)} / {args.num_series} raw series")

    atomic_write_json(
        progress_path,
        {
            "status": "raw_complete",
            "schema_version": SCHEMA_VERSION,
            "written_series": len(rows),
            "target_series": args.num_series,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    return raw_path, {"raw_series_count": len(rows)}


def make_eval_sample(row: dict, split: str, args: argparse.Namespace) -> dict | None:
    values = row["target"]
    split_points = row["split_points"]
    if split == "validation":
        end = split_points["val_end"]
    elif split == "test":
        end = split_points["length"]
    else:
        raise ValueError(split)

    start = end - args.context_length - args.prediction_length
    future_start = end - args.prediction_length
    if start < 0:
        return None
    context = values[start:future_start]
    future = values[future_start:end]
    if len(context) != args.context_length or len(future) != args.prediction_length:
        return None
    return {
        "item_id": row["item_id"],
        "freq": row.get("freq", args.freq),
        "context": context,
        "future": future,
        "source": row["source"],
        "split": split,
        "seasonality": 24 if args.freq.lower().startswith("h") else 1,
    }


def build_outputs(args: argparse.Namespace, output_dir: Path, raw_path: Path) -> dict:
    rows = load_jsonl(raw_path)
    train_records = []
    val_samples = []
    test_samples = []
    quality_rows = []
    skipped = {"short_train": 0, "short_validation": 0, "short_test": 0, "nan_or_inf": 0}

    for row in rows:
        values = np.asarray(row["target"], dtype=np.float32)
        if not np.isfinite(values).all():
            skipped["nan_or_inf"] += 1
            continue
        train_end = int(row["split_points"]["train_end"])
        if train_end < args.context_length + args.prediction_length:
            skipped["short_train"] += 1
            continue
        train_records.append(
            {
                "start": np.datetime64("2000-01-01 00:00", "s"),
                "target": values[:train_end],
            }
        )
        val_sample = make_eval_sample(row, "validation", args)
        test_sample = make_eval_sample(row, "test", args)
        if val_sample is None:
            skipped["short_validation"] += 1
        else:
            val_samples.append(val_sample)
        if test_sample is None:
            skipped["short_test"] += 1
        else:
            test_samples.append(test_sample)
        quality_rows.append(
            {
                "item_id": row["item_id"],
                "length": int(len(values)),
                "train_end": train_end,
                "val_end": int(row["split_points"]["val_end"]),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "pattern": row["generator"]["pattern"],
            }
        )

    if len(train_records) < args.min_train_series:
        raise RuntimeError(f"Not enough train series: {len(train_records)} < {args.min_train_series}")
    if len(test_samples) < args.min_eval_series:
        raise RuntimeError(f"Not enough test samples: {len(test_samples)} < {args.min_eval_series}")

    train_arrow = output_dir / "train.arrow"
    tmp_arrow = train_arrow.with_suffix(".arrow.tmp")
    if tmp_arrow.exists():
        tmp_arrow.unlink()
    ArrowWriter(compression="lz4").write_to_file(train_records, path=tmp_arrow)
    os.replace(tmp_arrow, train_arrow)

    validation_jsonl = output_dir / "validation.jsonl"
    test_jsonl = output_dir / "test.jsonl"
    validation_jsonl.unlink(missing_ok=True)
    test_jsonl.unlink(missing_ok=True)
    append_jsonl(validation_jsonl, val_samples)
    append_jsonl(test_jsonl, test_samples)

    quality_csv = output_dir / "data_quality.csv"
    with quality_csv.with_suffix(".csv.tmp").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["item_id", "length", "train_end", "val_end", "min", "max", "mean", "std", "pattern"],
        )
        writer.writeheader()
        writer.writerows(quality_rows)
    os.replace(quality_csv.with_suffix(".csv.tmp"), quality_csv)

    return {
        "train_series": len(train_records),
        "validation_samples": len(val_samples),
        "test_samples": len(test_samples),
        "skipped": skipped,
        "files": {
            "raw_series": str(raw_path),
            "train_arrow": str(train_arrow),
            "validation_jsonl": str(validation_jsonl),
            "test_jsonl": str(test_jsonl),
            "data_quality_csv": str(quality_csv),
        },
    }


def existing_complete_manifest(output_dir: Path, args: argparse.Namespace) -> bool:
    manifest_path = output_dir / "data_manifest.json"
    if not manifest_path.exists() or args.force:
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    config = manifest.get("config", {})
    expected = {
        "dataset": args.dataset,
        "num_series": args.num_series,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "seed": args.seed,
    }
    return manifest.get("status") == "complete" and all(config.get(k) == v for k, v in expected.items())


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if existing_complete_manifest(output_dir, args):
        print(f"Data already complete at {output_dir}. Use --force to rebuild.")
        return

    if args.force:
        for name in ["train.arrow", "validation.jsonl", "test.jsonl", "data_quality.csv", "data_manifest.json"]:
            (output_dir / name).unlink(missing_ok=True)

    raw_path, raw_stats = ensure_raw_series(args, output_dir)
    output_stats = build_outputs(args, output_dir, raw_path)

    manifest_path = output_dir / "data_manifest.json"
    file_hashes = {}
    for label, file_name in output_stats["files"].items():
        path = Path(file_name)
        file_hashes[label] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}

    manifest = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "dataset": args.dataset,
            "num_series": args.num_series,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "context_length": args.context_length,
            "prediction_length": args.prediction_length,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "freq": args.freq,
            "seed": args.seed,
        },
        "counts": {**raw_stats, **{k: v for k, v in output_stats.items() if k not in ["files", "skipped"]}},
        "skipped": output_stats["skipped"],
        "files": file_hashes,
    }
    atomic_write_json(manifest_path, manifest)

    progress_path = output_dir / "prepare_progress.json"
    atomic_write_json(
        progress_path,
        {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "manifest": str(manifest_path),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    readme = output_dir / "README.md"
    atomic_write_text(
        readme,
        "\n".join(
            [
                "# Prepared Chronos Data",
                "",
                f"- Dataset: `{args.dataset}`",
                f"- Train Arrow: `{output_stats['files']['train_arrow']}`",
                f"- Validation JSONL: `{output_stats['files']['validation_jsonl']}`",
                f"- Test JSONL: `{output_stats['files']['test_jsonl']}`",
                f"- Manifest: `{manifest_path}`",
                "",
                "This directory is resumable. If preparation is interrupted, rerun the same command.",
                "",
            ]
        ),
    )
    print(json.dumps({"manifest": str(manifest_path), "counts": manifest["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
