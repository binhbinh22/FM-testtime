import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from run import ERA5DataLoader  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import (  # noqa: E402
    build_config, get_window_with_wind_history, generate_multivar_candidates,
    compute_multivar_mse, aggregate_window_metrics, append_csv_row, build_shared_start_indices,
    compute_latitude_weights, compute_climatology,
)
from proposers import get_proposer, resolve_model_id  # noqa: E402

ARCHITECTURE_NAME = "greedy"

DEFAULT_CLIMATOLOGY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "era5_val_2015_2016.nc")


class GreedyMultivarEvaluator:
    """
    Khong search, khong verifier - lay dung 1 candidate (temperature=0,
    n_samples=1) cho moi bien t2m/u10/v10 -> SINGLE-PASS BASELINE.

    Chay duoc voi MOI backbone qua --backbone:
      - chronos  (Chronos-T5, token sampling)   -> greedy = temperature 0
      - moirai1  (Moirai-1.1-R, mixture sample)  -> greedy = 1 mau/median
      - moirai2  (Moirai-2.0, quantile)          -> baseline (median quantile)
      - timesfm  (TimesFM-2.x, quantile/point)   -> baseline (point forecast)
      - timesfm3 (TimesFM-3.0, quantile/point)   -> baseline (xem proposers.py)
    Khong dung Physics Filter (chi 1 candidate, khong co gi de chon).
    """

    def __init__(self, config, backbone: str = "chronos", climatology_path: str = DEFAULT_CLIMATOLOGY_PATH):
        self.config = config
        self.backbone = backbone
        self.loader = ERA5DataLoader(config)
        self.forecaster = get_proposer(
            backbone, model_id=config.model_id,
            history_steps=config.history_steps, prediction_steps=config.prediction_steps,
        )
        self.lat_weight = compute_latitude_weights(self.loader.ds.latitude.values)
        self.climatology = compute_climatology(climatology_path, config)

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
            window_metrics.append(compute_multivar_mse(
                candidates, best_idx=0, data=data, lat=lat, lon=lon,
                lat_weight=self.lat_weight, climatology=self.climatology,
            ))

        metrics = aggregate_window_metrics(window_metrics)
        print("\n" + "=" * 70)
        print(f"Greedy ({self.backbone}, single-pass baseline) - {n_windows} cua so")
        print(f"MSE da bien: {metrics['multivar_mse_norm_mean']:.6f} +/- {metrics['multivar_mse_norm_std']:.6f}")
        if "rmse_denorm_mean" in metrics:
            print(f"RMSE denorm: {metrics['rmse_denorm_mean']:.6f} | ACC denorm: {metrics['acc_denorm_mean']:.6f}")
        print("=" * 70)
        return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Chay Greedy/single-pass baseline (da backbone) 1 lan, ghi 1 hang vao CSV.")
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--setting", type=str, default="custom")
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--backbone", type=str, default="chronos",
                    choices=["chronos", "moirai1", "moirai2", "timesfm", "timesfm3"],
                    help="Backbone foundation model (xem proposers.py / README).")
    p.add_argument("--model-id", type=str, default=None,
                    help="Override checkpoint. Bo trong -> dung default_model_id cua backbone.")
    p.add_argument("--climatology-path", type=str, default=DEFAULT_CLIMATOLOGY_PATH,
                    help="Du lieu dung tinh climatology cho ACC (mac dinh: tap val, tach biet voi data-path dang danh gia).")
    p.add_argument("--report-path", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "reports", "results_greedy.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = resolve_model_id(args.backbone, args.model_id)
    config = build_config(args.history_steps, args.prediction_steps, model_id, args.data_path)

    evaluator = GreedyMultivarEvaluator(config, backbone=args.backbone, climatology_path=args.climatology_path)
    metrics = evaluator.run_test_set(n_windows=args.n_windows)

    row = {
        "architecture": f"{ARCHITECTURE_NAME}__{args.backbone}",
        "setting": args.setting,
        "history_steps": args.history_steps,
        "prediction_steps": args.prediction_steps,
        "n_windows": args.n_windows,
        "n_fine": 0,
        "n_coarse": 0,
        **metrics,
    }
    append_csv_row(args.report_path, row)
