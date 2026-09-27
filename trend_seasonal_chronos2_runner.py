import argparse
import csv
import os

import numpy as np

from run import ERA5Config, ERA5DataLoader
from chronos2_proposer import Chronos2Proposer
from multiscale_trend_seasonal_search import (
    TrendSeasonalSearchConfig,
    MultiscaleTrendSeasonalSearchEngine,
)
from trend_seasonal_eval_utils import build_shared_start_indices


CSV_FIELDS = [
    "history_steps", "prediction_steps", "n_windows",
    "n_fine", "n_coarse", "k_fine", "k_coarse",
    "w1_temporal", "w2_freq",
    "kernel_size_fine", "downsample_factor",
    "mse_norm_mean", "mse_norm_std",
]


class TrendSeasonalChronos2Evaluator:
    """
    Baseline Trend+FFT+Physics CHAY VOI CHRONOS-2 THAT - engine
    (MultiscaleTrendSeasonalSearchEngine) GIU NGUYEN, chi thay proposer
    tu ChronosProposer (Chronos T5) sang Chronos2Proposer -> ca 2 nhanh
    Fine/Coarse deu goi Chronos-2 that voi cross_learning=True tren toan
    bo V node, dung nghia "spatial-temporal".
    """

    def __init__(self, config: ERA5Config, search_config: TrendSeasonalSearchConfig = None):
        if "chronos-2" not in config.model_id and "chronos2" not in config.model_id:
            raise ValueError(
                f"config.model_id='{config.model_id}' khong phai Chronos-2 - "
                "dat ERA5Config(model_id='amazon/chronos-2', ...)."
            )
        self.config = config
        self.loader = ERA5DataLoader(config)
        self.forecaster = Chronos2Proposer(model_id=config.model_id, batch_size=100)
        self.engine = MultiscaleTrendSeasonalSearchEngine(
            proposer=self.forecaster,
            config=search_config or TrendSeasonalSearchConfig(),
        )

    def run_test_set(self, n_windows: int = 10) -> dict:
        start_indices = build_shared_start_indices(
            self.loader, self.config.history_steps, self.config.prediction_steps, n_windows
        )
        print(f"start_indices ({n_windows} cua so): {start_indices.tolist()}")

        mse_norm_list = []
        for start_idx in start_indices:
            data = self.loader.get_test_window(int(start_idx))
            result = self.engine.run(
                history=data["history_t2m"],
                u_wind_future=data["u10"],
                v_wind_future=data["v10"],
                prediction_steps=self.config.prediction_steps,
                interval_hours=self.config.interval_hours,
                ground_truth=data["gt_t2m"],
            )
            mse_norm_list.append(result["metrics"]["mse_norm"])

        mse_norm_mean = float(np.mean(mse_norm_list))
        mse_norm_std = float(np.std(mse_norm_list))

        print("\n" + "=" * 70)
        print(f"Trend+FFT+Physics (Chronos-2) - {n_windows} cua so")
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
    p = argparse.ArgumentParser(description="Chay Trend+FFT+Physics (Chronos-2) 1 lan, ghi 1 hang vao CSV.")
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--model-id", type=str, default="amazon/chronos-2")

    p.add_argument("--n-fine", type=int, default=15, help="N_A - so candidate nhanh Fine.")
    p.add_argument("--n-coarse", type=int, default=5, help="N_C - so candidate nhanh Coarse.")
    p.add_argument("--k-fine", type=int, default=5)
    p.add_argument("--k-coarse", type=int, default=5)
    p.add_argument("--w1", type=float, default=1.0, help="Trong so DTW (trend).")
    p.add_argument("--w2", type=float, default=1.0, help="Trong so FFT-L1 (seasonal).")
    p.add_argument("--kernel-size-fine", type=int, default=16)
    p.add_argument("--downsample-factor", type=int, default=4)

    p.add_argument("--report-path", type=str, default="report_trend_seasonal_chronos2.csv")
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

    search_config = TrendSeasonalSearchConfig(
        downsample_factor=args.downsample_factor,
        n_scenarios_fine=args.n_fine,
        n_scenarios_coarse=args.n_coarse,
        k_physics_fine=args.k_fine,
        k_physics_coarse=args.k_coarse,
        k_final=1,
        kernel_size_fine=args.kernel_size_fine,
        w1_temporal=args.w1,
        w2_freq=args.w2,
        w3_harmonic=0.0,
    )

    evaluator = TrendSeasonalChronos2Evaluator(config, search_config)
    metrics = evaluator.run_test_set(n_windows=args.n_windows)

    row = {
        "history_steps": args.history_steps,
        "prediction_steps": args.prediction_steps,
        "n_windows": args.n_windows,
        "n_fine": args.n_fine,
        "n_coarse": args.n_coarse,
        "k_fine": args.k_fine,
        "k_coarse": args.k_coarse,
        "w1_temporal": args.w1,
        "w2_freq": args.w2,
        "kernel_size_fine": args.kernel_size_fine,
        "downsample_factor": args.downsample_factor,
        "mse_norm_mean": metrics["mse_norm_mean"],
        "mse_norm_std": metrics["mse_norm_std"],
    }
    append_csv_row(args.report_path, row)
