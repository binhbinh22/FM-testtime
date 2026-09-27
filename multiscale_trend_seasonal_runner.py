import csv
import numpy as np
import torch

from run import ERA5Config, ERA5DataLoader, ChronosProposer
from multiscale_trend_seasonal_search import (
    TrendSeasonalSearchConfig,
    MultiscaleTrendSeasonalSearchEngine,
)
from trend_seasonal_eval_utils import build_shared_start_indices, evaluate_config


class TrendSeasonalERA5Evaluator:
    """
    Wrapper CHAY XAC NHAN (khong dung de sweep) pipeline Multiscale
    Trend-Seasonal Search tren 1 tap du lieu cho truoc, voi 1 bo
    hyperparameter DA CHON san. Khong sua doi run.py / era5_physics_energy.py.
    """

    def __init__(self, config: ERA5Config, search_config: TrendSeasonalSearchConfig = None):
        self.config = config
        self.loader = ERA5DataLoader(config)
        self.forecaster = ChronosProposer(model_id=config.model_id)
        self.engine = MultiscaleTrendSeasonalSearchEngine(
            proposer=self.forecaster,
            config=search_config or TrendSeasonalSearchConfig(),
        )

    def run_test_set(self, n_windows: int = 10, report_path: str = "report_trend_seasonal.csv"):
        start_indices = build_shared_start_indices(
            self.loader, self.config.history_steps, self.config.prediction_steps, n_windows
        )
        print(f"start_indices ({n_windows} cua so): {start_indices.tolist()}")

        metrics = evaluate_config(self.engine, self.loader, self.config, start_indices)

        print("\n" + "=" * 70)
        print(f"XAC NHAN 1 LAN tren {self.config.data_path} - {n_windows} cua so")
        print("=" * 70)
        print(f"MSE (denorm) : {metrics['mse_denorm_mean']:.6f} +/- {metrics['mse_denorm_std']:.6f}")
        print(f"MAE (denorm) : {metrics['mae_denorm_mean']:.6f} +/- {metrics['mae_denorm_std']:.6f}")
        print(f"MSE (norm)   : {metrics['mse_norm_mean']:.6f} +/- {metrics['mse_norm_std']:.6f}")
        print(f"MAE (norm)   : {metrics['mae_norm_mean']:.6f} +/- {metrics['mae_norm_std']:.6f}")
        print("=" * 70)

        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        print(f"Saved report: {report_path}")

        return metrics


if __name__ == "__main__":
    # QUAN TRONG: file nay chi dung de CHAY XAC NHAN 1 LAN DUY NHAT tren
    # tap TEST (era5_test_2017_2018.nc, dung mac dinh trong ERA5Config),
    # voi hyperparameter DA duoc chon san tu ket qua sweep tren tap
    # VALIDATION (xem multiscale_trend_seasonal_sweep.py va
    # multiscale_trend_seasonal_w1w2_sweep.py). KHONG dung file nay de do
    # tham so - lam vay se lam ro ri thong tin tu tap test (data leakage).
    BEST_N = 20        # <-- thay bang N tot nhat tim duoc tren val
    BEST_K = 5         # <-- thay bang K tot nhat tim duoc tren val
    BEST_W1 = 0.5      # <-- thay bang w1 tot nhat tim duoc tren val
    KERNEL_SIZE_FINE = 16

    config = ERA5Config(history_steps=512, prediction_steps=64)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    search_config = TrendSeasonalSearchConfig(
        n_scenarios_fine=BEST_N,
        n_scenarios_coarse=BEST_N,
        k_physics_fine=BEST_K,
        k_physics_coarse=BEST_K,
        k_final=1,
        kernel_size_fine=KERNEL_SIZE_FINE,
        w1_temporal=BEST_W1,
        w2_freq=1.0 - BEST_W1,
        w3_harmonic=0.0,
    )

    evaluator = TrendSeasonalERA5Evaluator(config, search_config)
    evaluator.run_test_set(n_windows=10, report_path="report_trend_seasonal_FINAL_TEST.csv")
