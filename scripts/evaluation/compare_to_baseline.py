import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import gmean


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Chronos metrics CSV with a baseline CSV.")
    parser.add_argument("--model_csv", required=True)
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json")
    return parser.parse_args()


def main():
    args = parse_args()
    model_df = pd.read_csv(args.model_csv).set_index("dataset")
    baseline_df = pd.read_csv(args.baseline_csv).set_index("dataset")
    common = sorted(set(model_df.index) & set(baseline_df.index))
    if not common:
        raise ValueError("No common datasets between model and baseline CSV files.")

    metric_cols = [col for col in ["MASE", "WQL"] if col in model_df.columns and col in baseline_df.columns]
    if not metric_cols:
        raise ValueError("No comparable metric columns found. Expected MASE and/or WQL.")

    rows = []
    for dataset in common:
        row = {"dataset": dataset}
        for metric in metric_cols:
            model_value = float(model_df.loc[dataset, metric])
            baseline_value = float(baseline_df.loc[dataset, metric])
            row[f"model_{metric}"] = model_value
            row[f"baseline_{metric}"] = baseline_value
            row[f"relative_{metric}"] = model_value / baseline_value
        rows.append(row)

    result_df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    summary = {"datasets": len(common), "metrics": {}}
    for metric in metric_cols:
        summary["metrics"][metric] = {
            "geomean_relative": float(gmean(result_df[f"relative_{metric}"])),
            "mean_relative": float(result_df[f"relative_{metric}"].mean()),
        }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
