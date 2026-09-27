import csv
import numpy as np
import torch

from run import ERA5Config, ERA5DataLoader, ChronosProposer
from multiscale_trend_seasonal_search import (
    TrendSeasonalSearchConfig,
    MultiscaleTrendSeasonalSearchEngine,
)
from trend_seasonal_eval_utils import build_shared_start_indices, evaluate_config


# ============================================================
# GRID SEARCH: N (TONG ngan sach compute = tong so lan goi Chronos tren
# CA 2 nhanh fine+coarse cong lai) x K (so candidate giu lai sau physics
# filter moi nhanh), tren pipeline multiscale Trend-Seasonal-Frequency
# search.
#
# QUAN TRONG - budget fairness: N la TONG so candidate sinh ra (fine +
# coarse), KHONG PHAI so candidate moi nhanh. Kien truc nay co 2 nhanh
# (fine, coarse) trong khi Best-of-N+Physics (baseline) chi co 1 nhanh -
# neu dat n_scenarios_fine=N VA n_scenarios_coarse=N (moi nhanh deu N)
# thi tong compute thuc te se la 2N, gap doi so voi kien truc 1 nhanh
# duoc goi cung la "N" - lam sai lech hoan toan y nghia "cung ngan sach"
# khi so sanh giua cac kien truc (xem EXPERIMENT_REPORT_fixed_N_budget.md).
# Vi vay o day N duoc CHIA DOI cho 2 nhanh: n_fine = N//2, n_coarse = N -
# n_fine, tong dung bang N.
#
# CO DINH cho toan bo sweep (theo thong nhat voi nguoi dung):
#   - model_id = amazon/chronos-t5-base
#   - history_steps = 512, prediction_steps = 64
#   - w1_temporal = w2_freq = 0.5, w3_harmonic = 0 (PINT tat)
#   - k_final = 1
#   - kernel_size_fine = 16, downsample_factor = 2 (khong sweep o buoc nay)
#   - CUNG mot bo start_indices (10 cua so) duoc dung cho MOI o luoi (N, K).
#
# CHAY TREN VAL DE CHON (N,K), CHI CHAY TREN TEST 1 LAN VOI (N,K) DA CHON.
#
# LUU Y HIEU NANG: ERA5DataLoader va ChronosProposer chi duoc khoi tao
# MOT LAN duy nhat va tai su dung xuyen suot sweep (load Chronos-t5-base
# rat ton kem, khong duoc load lai cho tung o luoi).
# ============================================================


N_GRID = [10, 15, 20, 30]
K_GRID = [3, 5, 7]

N_WINDOWS = 10

HISTORY_STEPS = 512
PREDICTION_STEPS = 64
KERNEL_SIZE_FINE = 16

REPORT_PATH = "report_trend_seasonal_sweep.csv"


def run_sweep(data_path: str = None, report_path: str = REPORT_PATH):
    """
    data_path: BAT BUOC truyen ro tap du lieu dang dung khi goi sweep -
    tap VALIDATION (de chon N, K) hay tap TEST (era5_test_2017_2018.nc,
    CHI chay 1 lan duy nhat voi (N, K) da chon tu val, khong dung de
    sweep). Neu None, dung mac dinh cua ERA5Config (tap test) - luon
    kiem tra dong log ben duoi truoc khi cho chay that su tren cluster.
    """
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
    print("=" * 70)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Chi khoi tao 1 LAN: data loader + Chronos model, tai su dung cho ca sweep.
    loader = ERA5DataLoader(config)
    forecaster = ChronosProposer(model_id=config.model_id)

    start_indices = build_shared_start_indices(
        loader, config.history_steps, config.prediction_steps, N_WINDOWS
    )
    print(f"Shared start_indices ({N_WINDOWS} cua so, dung chung cho moi (N,K)):")
    print(start_indices.tolist())

    fieldnames = [
        "his", "predict", "N", "K",
        "mse_norm_mean", "mse_norm_std",
        "mse_denorm_mean", "mse_denorm_std",
        "mae_norm_mean", "mae_norm_std",
        "mae_denorm_mean", "mae_denorm_std",
    ]

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for N in N_GRID:
            n_fine = N // 2
            n_coarse = N - n_fine
            for K in K_GRID:
                k_eff = min(K, n_fine, n_coarse)

                search_config = TrendSeasonalSearchConfig(
                    n_scenarios_fine=n_fine,
                    n_scenarios_coarse=n_coarse,
                    k_physics_fine=min(K, n_fine),
                    k_physics_coarse=min(K, n_coarse),
                    k_final=1,
                    kernel_size_fine=KERNEL_SIZE_FINE,
                    w1_temporal=0.5,
                    w2_freq=0.5,
                    w3_harmonic=0.0,
                )
                engine = MultiscaleTrendSeasonalSearchEngine(
                    proposer=forecaster, config=search_config
                )

                print("\n" + "=" * 70)
                print(f"RUNNING his={config.history_steps} predict={config.prediction_steps} "
                      f"N_total={N} (fine={n_fine}, coarse={n_coarse}) K={k_eff}")
                print("=" * 70)

                metrics = evaluate_config(engine, loader, config, start_indices)

                row = {
                    "his": config.history_steps,
                    "predict": config.prediction_steps,
                    "N": N,
                    "K": k_eff,
                    **metrics,
                }
                writer.writerow(row)
                f.flush()

                print(
                    f"  MSE(denorm) = {metrics['mse_denorm_mean']:.6f} +/- {metrics['mse_denorm_std']:.6f}"
                )
                print(
                    f"  MAE(denorm) = {metrics['mae_denorm_mean']:.6f} +/- {metrics['mae_denorm_std']:.6f}"
                )

    print(f"\nSaved sweep report: {report_path}")


if __name__ == "__main__":
    # QUAN TRONG: sweep N/K de CHON tham so PHAI chay tren tap VALIDATION,
    # khong phai tap test - neu khong ket qua se bi lac quan gia (data
    # leakage). Doi VAL_DATA_PATH sang file validation cua ban (vi du tao
    # bang down.py: download_era5_slice_to_local(..., start_date="2015-01-01",
    # end_date="2016-12-31")) truoc khi chay sweep that su.
    VAL_DATA_PATH = "/home/user18/binhnkt/era5_val_2015_2016.nc"

    run_sweep(
        data_path=VAL_DATA_PATH,
        report_path="report_trend_seasonal_sweep_VAL.csv",
    )
