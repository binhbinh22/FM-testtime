import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from run import ERA5DataLoader, ChronosProposer  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import (  # noqa: E402
    build_config, get_window_with_wind_history, generate_multivar_candidates,
    compute_multivar_mse, aggregate_window_metrics, append_csv_row, build_shared_start_indices,
)

ARCHITECTURE_NAME = "greedy"


class GreedyMultivarEvaluator:
    """
    Khong search, khong verifier - lay dung 1 candidate (temperature=0,
    n_samples=1) cho moi bien t2m/u10/v10. Dung Chronos-1 that
    (ChronosProposer, run.py) - KHONG dung Physics Filter vi chi co 1
    candidate, khong co gi de chon.
    """

    def __init__(self, config):
        self.config = config
        self.loader = ERA5DataLoader(config)
        self.forecaster = ChronosProposer(model_id=config.model_id)

    def run_test_set(self, n_windows: int = 10) -> dict:
        start_indices = build_shared_start_indices(
            self.loader, self.config.history_steps, self.config.prediction_steps, n_windows
        )
        print(f"start_indices ({n_windows} cua so): {start_indices.tolist()}")

        lat, lon = self.config.grid_size
        window_metrics = []
        for start_idx in start_indices:
            data = get_window_with_wind_history(self.loader, self.config, int(start_idx))
            candidates = generate_multivar_candidates(
                self.forecaster, data, self.config.prediction_steps, n_samples=1, temperature=0.0,
            )
            window_metrics.append(compute_multivar_mse(candidates, best_idx=0, data=data, lat=lat, lon=lon))

        metrics = aggregate_window_metrics(window_metrics)
        print("\n" + "=" * 70)
        print(f"Greedy (Chronos-1, da bien) - {n_windows} cua so")
        print(f"MSE da bien: {metrics['multivar_mse_norm_mean']:.6f} +/- {metrics['multivar_mse_norm_std']:.6f}")
        print("=" * 70)
        return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Chay Greedy (Chronos-1, da bien) 1 lan, ghi 1 hang vao CSV.")
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--setting", type=str, default="custom")
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--model-id", type=str, default="amazon/chronos-t5-base")
    p.add_argument("--report-path", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "reports", "results_greedy.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args.history_steps, args.prediction_steps, args.model_id, args.data_path)

    evaluator = GreedyMultivarEvaluator(config)
    metrics = evaluator.run_test_set(n_windows=args.n_windows)

    row = {
        "architecture": ARCHITECTURE_NAME,
        "setting": args.setting,
        "history_steps": args.history_steps,
        "prediction_steps": args.prediction_steps,
        "n_windows": args.n_windows,
        "n_fine": 0,
        "n_coarse": 0,
        **metrics,
    }
    append_csv_row(args.report_path, row)
