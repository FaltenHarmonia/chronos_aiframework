import argparse
import json
import math
import re
import time
from pathlib import Path


TQDM_RE = re.compile(r"(?P<current>\d+)\s*it\s+\[(?P<elapsed>[^,\]]+),\s*(?P<rate>[^\]]+)\]")


def parse_args():
    parser = argparse.ArgumentParser(description="Print a readable progress bar for a running Chronos evaluation.")
    parser.add_argument("--progress_json", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--output_log")
    parser.add_argument("--once", action="store_true", help="Print one progress snapshot and exit.")
    return parser.parse_args()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def last_tqdm_line(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("rb") as fp:
            fp.seek(0, 2)
            size = fp.tell()
            fp.seek(max(0, size - 128 * 1024))
            raw = fp.read()
            if raw.count(b"\x00") > len(raw) // 4:
                text = raw.decode("utf-16-le", errors="ignore")
            else:
                text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return None
    matches = list(TQDM_RE.finditer(text))
    return matches[-1] if matches else None


def bar(done, total, width=28):
    if not total or total <= 0:
        return "[" + "?" * width + "]"
    done = max(0, min(done, total))
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def emit(line: str, output_log: Path | None):
    try:
        print(line, flush=True)
    except OSError:
        pass
    if output_log is not None:
        output_log.parent.mkdir(parents=True, exist_ok=True)
        with output_log.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")


def main():
    args = parse_args()
    progress_path = Path(args.progress_json)
    log_path = Path(args.log_path)
    csv_path = Path(args.csv_path)
    output_log = Path(args.output_log) if args.output_log else None

    while True:
        progress = read_json(progress_path)
        dataset_index = progress.get("dataset_index")
        dataset_total = progress.get("dataset_total")
        dataset = progress.get("current_dataset", "unknown")
        stage = progress.get("stage", "unknown")
        test_series = progress.get("test_series")
        total_batches = math.ceil(test_series / args.batch_size) if test_series else None
        match = last_tqdm_line(log_path)
        current_batch = int(match.group("current")) if match else None
        rate = match.group("rate") if match else "unknown rate"

        if current_batch is not None and total_batches:
            pct = current_batch / total_batches * 100
            line = (
                f"{time.strftime('%H:%M:%S')} "
                f"suite {dataset_index}/{dataset_total} {dataset} {stage} "
                f"{bar(current_batch, total_batches)} "
                f"{current_batch}/{total_batches} batches ({pct:.1f}%), {rate}"
            )
        else:
            line = (
                f"{time.strftime('%H:%M:%S')} "
                f"suite {dataset_index}/{dataset_total} {dataset} {stage} "
                f"{bar(0, 0)} waiting for batch progress"
            )
        emit(line, output_log)

        if csv_path.exists():
            emit(f"{time.strftime('%H:%M:%S')} complete: {csv_path}", output_log)
            break
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
