"""Quick local eval: 加载 fallback checkpoint, 在合成数据上算 MASE/WQL."""
import numpy as np
import torch
from pathlib import Path
from gluonts.dataset.common import FileDataset, ListDataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast

from chronos import ChronosPipeline

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PRED_LEN = 32
N_SERIES = 200


def load_model(checkpoint_dir: str):
    return ChronosPipeline.from_pretrained(
        checkpoint_dir, device_map="auto", torch_dtype=torch.float32,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/chronos_t5_tiny_fallback/run-3/checkpoint-final")
    parser.add_argument("--data", default="data/processed/synthetic_fallback_50k.arrow")
    parser.add_argument("--n_series", type=int, default=N_SERIES)
    parser.add_argument("--pred_len", type=int, default=PRED_LEN)
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()

    print(f"=== 快速评估 ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test series: {args.n_series}")
    print(f"  Pred length: {args.pred_len}")

    print("Loading model...")
    pipe = load_model(args.checkpoint)

    print("Preparing test data...")
    ds = FileDataset(path=Path(args.data), freq="h")
    entries = [e for i, e in enumerate(ds) if i < args.n_series]
    test_ds = ListDataset(entries, freq="h")
    offset = -args.pred_len
    _, test_template = split(test_ds, offset=offset)
    test_data = test_template.generate_instances(prediction_length=args.pred_len, windows=1)
    n_test = len(test_data.input)
    print(f"  Test instances: {n_test}")

    print("Generating forecasts...")
    forecasts = []
    for ts in test_data.input:
        context = torch.tensor(np.asarray(ts["target"], dtype=np.float32))
        with torch.no_grad():
            samples = pipe.predict(context, prediction_length=args.pred_len, num_samples=args.num_samples)
        samples = samples[0].cpu().numpy()
        fc_start = ts["start"] + len(ts["target"])
        forecasts.append(SampleForecast(samples=samples, start_date=fc_start))

    print("Computing MASE & WQL...")
    metrics = evaluate_forecasts(
        forecasts=forecasts, test_data=test_data,
        metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
    )
    mase = float(np.mean(metrics["MASE[0.5]"]))
    wql = float(np.mean(metrics["mean_weighted_sum_quantile_loss"]))

    print(f"\n{'='*40}")
    print(f"  Series evaluated: {n_test}")
    print(f"  MASE = {mase:.4f}")
    print(f"  WQL  = {wql:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
