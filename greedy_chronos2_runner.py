import argparse
import csv
import os

import numpy as np

from run import ERA5Config, ERA5DataLoader
from chronos2_proposer import Chronos2Proposer
from trend_seasonal_eval_utils import build_shared_start_indices


CSV_FIELDS = [
    "history_steps", "prediction_steps", "n_windows",
    "n_fine", "n_coarse", "k_fine",  # khong ap dung cho Greedy - luon =1/None, giu cot de gop bang de dang voi 2 script kia
    "mse_norm_mean", "mse_norm_std",
]


class GreedyChronos2Evaluator:
    """
    Baseline Greedy: KHONG search, KHONG verifier - lay dung 1 candidate
    duy nhat tu Chronos-2 (n_samples=1 -> quantile 0.5). Dung CHUNG
    Chronos-2 that (cross_learning=True) va CUNG start_indices voi cac
    runner khac (build_shared_start_indices deterministic theo config).
    """

    def __init__(self, config: ERA5Config):
        if "chronos-2" not in config.model_id and "chronos2" not in config.model_id:
            raise ValueError(
                f"config.model_id='{config.model_id}' khong phai Chronos-2 - "
                "dat ERA5Config(model_id='amazon/chronos-2', ...)."
            )
        self.config = config
        self.loader = ERA5DataLoader(config)
        self.forecaster = Chronos2Proposer(model_id=config.model_id, batch_size=100)

    def run_test_set(self, n_windows: int = 10) -> dict:
        start_indices = build_shared_start_indices(
            self.loader, self.config.history_steps, self.config.prediction_steps, n_windows
        )
        print(f"start_indices ({n_windows} cua so): {start_indices.tolist()}")

        lat, lon = self.config.grid_size
        mse_norm_list = []

        for start_idx in start_indices:
            data = self.loader.get_test_window(int(start_idx))
            history = data["history_t2m"]
            gt = data["gt_t2m"]

            _forecast_denorm, forecast_norm, mean, std = self.forecaster.predict(
                history,
                prediction_steps=self.config.prediction_steps,
                temperature=1.0,
                n_samples=1,
            )
            final_candidate_norm = forecast_norm[0]

            mean_spatial = mean.reshape(lat, lon)
            std_spatial = std.reshape(lat, lon)
            gt_norm = (gt - mean_spatial[np.newaxis, :, :]) / std_spatial[np.newaxis, :, :]

            mse_norm_list.append(float(np.mean((final_candidate_norm - gt_norm) ** 2)))

        mse_norm_mean = float(np.mean(mse_norm_list))
        mse_norm_std = float(np.std(mse_norm_list))

        print("\n" + "=" * 70)
        print(f"GREEDY (Chronos-2) - {n_windows} cua so")
        print(f"MSE (norm) : {mse_norm_mean:.6f} +/- {mse_norm_std:.6f}")
        print("=" * 70)

        return {"mse_norm_mean": mse_norm_mean, "mse_norm_std": mse_norm_std}


def append_csv_row(report_path: str, row: dict):
    file_exists = os.path.isfile(report_path)
    with open(report_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended 1 row -> {report_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Chay Greedy (Chronos-2) 1 lan, ghi 1 hang vao CSV.")
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--model-id", type=str, default="amazon/chronos-2")
    p.add_argument("--report-path", type=str, default="report_greedy_chronos2.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config_kwargs = dict(
        history_steps=args.history_steps,
        prediction_steps=args.prediction_steps,
        model_id=args.model_id,
    )
    if args.data_path is not None:
        config_kwargs["data_path"] = args.data_path
    config = ERA5Config(**config_kwargs)

    evaluator = GreedyChronos2Evaluator(config)
    metrics = evaluator.run_test_set(n_windows=args.n_windows)

    row = {
        "history_steps": args.history_steps,
        "prediction_steps": args.prediction_steps,
        "n_windows": args.n_windows,
        "n_fine": 1,
        "n_coarse": None,
        "k_fine": 1,
        "mse_norm_mean": metrics["mse_norm_mean"],
        "mse_norm_std": metrics["mse_norm_std"],
    }
    append_csv_row(args.report_path, row)
