import csv
import numpy as np
import torch
import os

from run import ERA5Config, ERA5DataLoader, ChronosProposer
from multiscale_trend_seasonal_search import (
    TrendSeasonalSearchConfig,
    MultiscaleTrendSeasonalSearchEngine,
)
from trend_seasonal_eval_utils import build_shared_start_indices, evaluate_config


# ============================================================
# GRID SEARCH: w1 (trong so E_temporal - DTW tren Trend) vs
# w2 = 1 - w1 (trong so E_FFT - FreDF tren Seasonal).
#
# CHAY SAU KHI DA CHON (N*, K*) tu multiscale_trend_seasonal_sweep.py.
# Giu N*, K* CO DINH o day, chi doi w1/w2 - day la kieu tuning TUAN TU
# (coordinate-wise), khong phai joint grid search day du tren ca 4 truc
# (N, K, w1, w2) vi chi phi Chronos qua lon cho joint grid.
#
# CHAY TREN VAL DE CHON w1, CHI CHAY TREN TEST 1 LAN VOI w1 DA CHON.
# ============================================================


# TODO: thay bang (N*, K*) tot nhat tu multiscale_trend_seasonal_sweep.py
# BEST_N la TONG ngan sach (fine+coarse cong lai) - xem giai thich budget
# fairness trong multiscale_trend_seasonal_sweep.py.
BEST_N = 20
BEST_N_FINE = BEST_N // 2
BEST_N_COARSE = BEST_N - BEST_N_FINE
BEST_K = 5

W1_GRID = [0.00, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 1.00]

N_WINDOWS = 10

HISTORY_STEPS = 512
PREDICTION_STEPS = 64
KERNEL_SIZE_FINE = 16

REPORT_PATH = "report_trend_seasonal_w1w2_sweep.csv"

# Diem w1=0.50 trung voi cau hinh (N*, K*, w1=0.5) DA CHAY roi trong
# multiscale_trend_seasonal_sweep.py (Thi nghiem 1 dung w1=w2=0.5 co dinh).
# De 2 bang trong bao cao KHONG bi lech so (cung 1 cau hinh phai ra cung 1
# so), diem nay duoc DOC LAI tu CSV cua sweep N/K thay vi chay lai Chronos
# (vua tiet kiem compute, vua dam bao nhat quan tuyet doi).
NK_SWEEP_CSV_PATH = "report_trend_seasonal_sweep_VAL.csv"


def load_cached_nk_metrics(csv_path: str, N: int, K: int) -> dict:
    """Doc lai dong (N, K) tuong ung tu report cua multiscale_trend_seasonal_sweep.py."""
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["N"]) == N and int(row["K"]) == K:
                return {k: float(v) for k, v in row.items() if k not in ("his", "predict", "N", "K")}
    return None


def run_sweep(data_path: str = None, report_path: str = REPORT_PATH, nk_sweep_csv_path: str = NK_SWEEP_CSV_PATH):
    config_kwargs = dict(
        history_steps=HISTORY_STEPS,
        prediction_steps=PREDICTION_STEPS,
        model_id="amazon/chronos-t5-base",
    )
    if data_path is not None:
        config_kwargs["data_path"] = data_path
    config = ERA5Config(**config_kwargs)

    print("=" * 70)
    print(f"DATASET DUNG CHO SWEEP NAY: {config.data_path}")
    print(f"N* = {BEST_N}, K* = {BEST_K} (co dinh, lay tu sweep N/K truoc do)")
    print("=" * 70)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    loader = ERA5DataLoader(config)
    forecaster = ChronosProposer(model_id=config.model_id)

    start_indices = build_shared_start_indices(
        loader, config.history_steps, config.prediction_steps, N_WINDOWS
    )
    print(f"Shared start_indices ({N_WINDOWS} cua so, dung chung cho moi w1):")
    print(start_indices.tolist())

    fieldnames = [
        "his", "predict", "N", "K", "w1", "w2",
        "mse_norm_mean", "mse_norm_std",
        "mse_denorm_mean", "mse_denorm_std",
        "mae_norm_mean", "mae_norm_std",
        "mae_denorm_mean", "mae_denorm_std",
    ]

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for w1 in W1_GRID:
            w2 = 1.0 - w1

            # w1=0.5 la DUNG CAU HINH da chay o Thi nghiem 1 (N/K sweep,
            # co dinh w1=w2=0.5) -> tai su dung ket qua da co, KHONG chay
            # lai (Chronos co random sampling, chay lai se cho so hoi khac
            # va gay nham lan "vi sao 2 bang khong khop nhau").
            if abs(w1 - 0.5) < 1e-9:
                cached = load_cached_nk_metrics(nk_sweep_csv_path, BEST_N, BEST_K)
                if cached is not None:
                    print("\n" + "=" * 70)
                    print(
                        f"w1=0.50 w2=0.50 (N={BEST_N}, K={BEST_K}) -> TAI SU DUNG ket qua "
                        f"tu {nk_sweep_csv_path}, khong chay lai Chronos."
                    )
                    print("=" * 70)
                    metrics = cached
                    row = {
                        "his": config.history_steps,
                        "predict": config.prediction_steps,
                        "N": BEST_N,
                        "K": BEST_K,
                        "w1": w1,
                        "w2": w2,
                        **metrics,
                    }
                    writer.writerow(row)
                    f.flush()
                    print(f"  MAE(denorm) = {metrics['mae_denorm_mean']:.6f} +/- {metrics['mae_denorm_std']:.6f}")
                    print(f"  MSE(denorm) = {metrics['mse_denorm_mean']:.6f} +/- {metrics['mse_denorm_std']:.6f}")
                    continue
                print(
                    f"[canh bao] Khong tim thay {nk_sweep_csv_path} de tai su dung diem w1=0.5 "
                    "- se chay lai Chronos cho diem nay (so co the hoi lech so voi Thi nghiem 1 "
                    "do Chronos sampling ngau nhien)."
                )

            search_config = TrendSeasonalSearchConfig(
                n_scenarios_fine=BEST_N_FINE,
                n_scenarios_coarse=BEST_N_COARSE,
                k_physics_fine=min(BEST_K, BEST_N_FINE),
                k_physics_coarse=min(BEST_K, BEST_N_COARSE),
                k_final=1,
                kernel_size_fine=KERNEL_SIZE_FINE,
                w1_temporal=w1,
                w2_freq=w2,
                w3_harmonic=0.0,
            )
            engine = MultiscaleTrendSeasonalSearchEngine(
                proposer=forecaster, config=search_config
            )

            print("\n" + "=" * 70)
            print(f"RUNNING w1={w1:.2f} w2={w2:.2f} (N={BEST_N}, K={BEST_K})")
            print("=" * 70)

            metrics = evaluate_config(engine, loader, config, start_indices)

            row = {
                "his": config.history_steps,
                "predict": config.prediction_steps,
                "N": BEST_N,
                "K": BEST_K,
                "w1": w1,
                "w2": w2,
                **metrics,
            }
            writer.writerow(row)
            f.flush()

            print(
                f"  MAE(denorm) = {metrics['mae_denorm_mean']:.6f} +/- {metrics['mae_denorm_std']:.6f}"
            )
            print(
                f"  MSE(denorm) = {metrics['mse_denorm_mean']:.6f} +/- {metrics['mse_denorm_std']:.6f}"
            )

    print(f"\nSaved sweep report: {report_path}")


if __name__ == "__main__":
    # QUAN TRONG: giong sweep N/K, sweep w1/w2 PHAI chay tren tap VALIDATION.
    VAL_DATA_PATH = "/home/user18/binhnkt/era5_val_2015_2016.nc"

    run_sweep(
        data_path=VAL_DATA_PATH,
        report_path="report_trend_seasonal_w1w2_sweep_VAL.csv",
    )
