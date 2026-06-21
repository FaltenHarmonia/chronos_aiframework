import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


CONFIGS = {
    "quick-zero-shot": Path("scripts/evaluation/configs/quick-zero-shot.yaml"),
    "quick-in-domain": Path("scripts/evaluation/configs/quick-in-domain.yaml"),
    "zero-shot": Path("scripts/evaluation/configs/zero-shot.yaml"),
    "in-domain": Path("scripts/evaluation/configs/in-domain.yaml"),
}


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate selected Chronos checkpoints.")
    parser.add_argument("--run_dir", required=True, help="Training run directory containing checkpoint-* folders.")
    parser.add_argument("--model_name", required=True, help="Stable name used in output files.")
    parser.add_argument("--checkpoints", default="checkpoint-final", help="Comma-separated checkpoint names.")
    parser.add_argument(
        "--suites",
        default="quick-zero-shot,quick-in-domain",
        help=f"Comma-separated evaluation suites. Choices: {','.join(CONFIGS)}",
    )
    parser.add_argument("--output_dir", default="outputs/evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="float16")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as fp:
        return max(sum(1 for _ in fp) - 1, 0)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    checkpoints = [item.strip() for item in args.checkpoints.split(",") if item.strip()]
    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    unknown = [suite for suite in suites if suite not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown evaluation suites: {unknown}")

    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "evaluation_progress.json"
    summary_path = output_dir / "evaluation_summary.csv"

    rows = []
    atomic_write_json(
        progress_path,
        {
            "status": "started",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_dir": str(run_dir),
            "model_name": args.model_name,
            "checkpoints": checkpoints,
            "suites": suites,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
        },
    )

    for checkpoint in checkpoints:
        checkpoint_path = run_dir / checkpoint
        if not checkpoint_path.is_dir():
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "suite": "",
                    "metrics_path": "",
                    "status": "missing_checkpoint",
                    "rows": 0,
                    "seconds": 0.0,
                }
            )
            continue

        for suite in suites:
            metrics_path = output_dir / f"{args.model_name}-{checkpoint}-{suite}.csv"
            suite_progress_path = output_dir / f"{args.model_name}-{checkpoint}-{suite}.progress.json"
            if metrics_path.exists() and not args.overwrite:
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "suite": suite,
                        "metrics_path": str(metrics_path),
                        "suite_progress_path": str(suite_progress_path),
                        "status": "skipped_existing",
                        "rows": count_csv_rows(metrics_path),
                        "seconds": 0.0,
                    }
                )
                continue

            atomic_write_json(
                progress_path,
                {
                    "status": "running",
                    "checkpoint": checkpoint,
                    "suite": suite,
                    "model_id": str(checkpoint_path),
                    "metrics_path": str(metrics_path),
                    "suite_progress_path": str(suite_progress_path),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            started = time.time()
            cmd = [
                sys.executable,
                "scripts/evaluation/evaluate.py",
                "chronos",
                str(CONFIGS[suite]),
                str(metrics_path),
                "--model-id",
                str(checkpoint_path),
                "--device",
                args.device,
                "--torch-dtype",
                args.torch_dtype,
                "--batch-size",
                str(args.batch_size),
                "--num-samples",
                str(args.num_samples),
                "--progress-path",
                str(suite_progress_path),
            ]
            try:
                subprocess.run(cmd, check=True)
                status = "complete"
            except subprocess.CalledProcessError as exc:
                atomic_write_json(
                    output_dir / "evaluation_failure.json",
                    {
                        "status": "failed",
                        "checkpoint": checkpoint,
                        "suite": suite,
                        "returncode": exc.returncode,
                        "cmd": cmd,
                        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                )
                raise
            seconds = round(time.time() - started, 3)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "suite": suite,
                    "metrics_path": str(metrics_path),
                    "status": status,
                    "rows": count_csv_rows(metrics_path),
                    "seconds": seconds,
                }
            )

    with summary_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["checkpoint", "suite", "metrics_path", "status", "rows", "seconds"])
        writer.writeheader()
        writer.writerows(rows)

    atomic_write_json(
        progress_path,
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary_path": str(summary_path),
            "result_count": len(rows),
        },
    )
    print(json.dumps({"summary_path": str(summary_path), "result_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
