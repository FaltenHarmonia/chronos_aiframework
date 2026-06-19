import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from gluonts.dataset.arrow import ArrowWriter
from tqdm import tqdm


HF_DATASET_NAME = "autogluon/chronos_datasets"
TSMIXUP_CONFIG = "training_corpus_tsmixup_10m"
KERNEL_SYNTH_CONFIG = "training_corpus_kernel_synth_1m"
SCHEMA_VERSION = "chronos-corpus-subset-v1"


LEVEL_PRESETS = {
    "debug": (900, 100),
    "conservative": (45_000, 5_000),
    "recommended": (90_000, 10_000),
    "stretch": (180_000, 20_000),
    "v100": (900_000, 100_000),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Chronos official training-corpus subset.")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--data_level", choices=LEVEL_PRESETS.keys(), default="recommended")
    parser.add_argument("--tsmixup_count", type=int)
    parser.add_argument("--kernel_count", type=int)
    parser.add_argument("--min_length", type=int, default=288)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compression", default="lz4")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_nan_ratio", type=float, default=0.2)
    return parser.parse_args()


def count_label(count: int) -> str:
    return f"{count // 1000}k" if count >= 1000 and count % 1000 == 0 else str(count)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_target(row):
    for key in ["target", "value", "values", "series"]:
        if key in row and row[key] is not None:
            return row[key]
    return None


def clean_target(raw_target, min_length, max_length, max_nan_ratio, rng, stats):
    if raw_target is None:
        stats["invalid_target"] += 1
        return None
    try:
        target = np.asarray(raw_target, dtype=np.float32).squeeze()
    except Exception:
        stats["invalid_target"] += 1
        return None
    if target.ndim != 1:
        target = target.reshape(-1)
    if len(target) < min_length:
        stats["too_short"] += 1
        return None

    target = target.copy()
    target[~np.isfinite(target)] = np.nan
    nan_ratio = float(np.isnan(target).mean())
    if nan_ratio > max_nan_ratio:
        stats["too_many_nan"] += 1
        return None
    if np.isnan(target).any():
        mask = np.isnan(target)
        valid_idx = np.where(~mask, np.arange(len(target)), 0)
        np.maximum.accumulate(valid_idx, out=valid_idx)
        target = target[valid_idx]
        target[np.isnan(target)] = 0.0
        stats["nan_filled"] += 1

    if max_length is not None and max_length > 0 and len(target) > max_length:
        start = int(rng.integers(0, len(target) - max_length + 1))
        target = target[start : start + max_length]
        stats["truncated"] += 1

    if len(target) < min_length:
        stats["too_short"] += 1
        return None
    return target.astype(np.float32)


def iter_clean_records(config_name, count, args, stats, seed_offset=0):
    rng = np.random.default_rng(args.seed + seed_offset)
    ds = load_dataset(HF_DATASET_NAME, config_name, split="train", streaming=True)
    pbar = tqdm(total=count, desc=f"Collecting {config_name}")
    try:
        for row in ds:
            stats["seen"] += 1
            target = clean_target(
                extract_target(row),
                min_length=args.min_length,
                max_length=args.max_length,
                max_nan_ratio=args.max_nan_ratio,
                rng=rng,
                stats=stats,
            )
            if target is None:
                continue
            stats["written"] += 1
            pbar.update(1)
            yield {"start": np.datetime64("2000-01-01 00:00", "s"), "target": target}
            if stats["written"] >= count:
                break
        if stats["written"] < count:
            raise RuntimeError(f"{config_name}: wrote {stats['written']} / requested {count}")
    finally:
        pbar.close()


def empty_stats():
    return {
        "seen": 0,
        "written": 0,
        "invalid_target": 0,
        "too_short": 0,
        "too_many_nan": 0,
        "nan_filled": 0,
        "truncated": 0,
    }


def write_arrow(records, output_path: Path, compression: str):
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        ArrowWriter(compression=compression).write_to_file(records, path=tmp_path)
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def build_output_names(args, tsmixup_count, kernel_count):
    tsmixup_name = f"chronos_tsmixup_{count_label(tsmixup_count)}.arrow"
    kernel_name = f"chronos_kernel_synth_{count_label(kernel_count)}.arrow"
    if args.data_level == "debug":
        tsmixup_name = f"chronos_debug_tsmixup_{tsmixup_count}.arrow"
        kernel_name = f"chronos_debug_kernel_synth_{kernel_count}.arrow"
    return tsmixup_name, kernel_name


def main():
    args = parse_args()
    preset_tsmixup, preset_kernel = LEVEL_PRESETS[args.data_level]
    tsmixup_count = args.tsmixup_count if args.tsmixup_count is not None else preset_tsmixup
    kernel_count = args.kernel_count if args.kernel_count is not None else preset_kernel
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tsmixup_name, kernel_name = build_output_names(args, tsmixup_count, kernel_count)

    tsmixup_path = output_dir / tsmixup_name
    kernel_path = output_dir / kernel_name
    summary_path = output_dir / "chronos_corpus_subset_summary.json"
    progress_path = output_dir / "chronos_corpus_subset_progress.json"

    config_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": HF_DATASET_NAME,
        "data_level": args.data_level,
        "tsmixup_config": TSMIXUP_CONFIG,
        "kernel_config": KERNEL_SYNTH_CONFIG,
        "tsmixup_requested": tsmixup_count,
        "kernel_requested": kernel_count,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "seed": args.seed,
        "output_files": {"tsmixup": str(tsmixup_path), "kernel_synth": str(kernel_path)},
    }
    if args.dry_run:
        print(json.dumps(config_payload, ensure_ascii=False, indent=2))
        return

    tsmixup_stats = empty_stats()
    kernel_stats = empty_stats()

    if tsmixup_path.exists() and not args.overwrite:
        atomic_write_json(progress_path, {"status": "skipping_existing_tsmixup", **config_payload})
        tsmixup_stats["written"] = tsmixup_count
    else:
        atomic_write_json(progress_path, {"status": "collecting_tsmixup", **config_payload})
        write_arrow(
            iter_clean_records(TSMIXUP_CONFIG, tsmixup_count, args, tsmixup_stats, seed_offset=0),
            tsmixup_path,
            args.compression,
        )

    if kernel_path.exists() and not args.overwrite:
        atomic_write_json(progress_path, {"status": "skipping_existing_kernel_synth", **config_payload})
        kernel_stats["written"] = kernel_count
    else:
        atomic_write_json(progress_path, {"status": "collecting_kernel_synth", **config_payload})
        write_arrow(
            iter_clean_records(KERNEL_SYNTH_CONFIG, kernel_count, args, kernel_stats, seed_offset=1),
            kernel_path,
            args.compression,
        )

    summary = {
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **config_payload,
        "tsmixup_written": tsmixup_count,
        "kernel_written": kernel_count,
        "ratio": {"tsmixup": 0.9, "kernel_synth": 0.1},
        "stats": {"tsmixup": tsmixup_stats, "kernel_synth": kernel_stats},
        "files": {
            "tsmixup": {"path": str(tsmixup_path), "bytes": tsmixup_path.stat().st_size, "sha256": sha256_file(tsmixup_path)},
            "kernel_synth": {
                "path": str(kernel_path),
                "bytes": kernel_path.stat().st_size,
                "sha256": sha256_file(kernel_path),
            },
        },
    }
    atomic_write_json(summary_path, summary)
    atomic_write_json(progress_path, {"status": "complete", "summary": str(summary_path), **config_payload})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
