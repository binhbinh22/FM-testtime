import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run import ERA5Config, ERA5DataLoader  # noqa: E402
from trend_seasonal_eval_utils import build_shared_start_indices  # noqa: E402,F401


# ============================================================
# Setting chung cho toan bo chronos1_multivar/ (xem README.md):
#   Val:  2015-01-01 -> 2016-12-31 (chua tai, can download_era5_slice_to_local)
#   Test: 2017-01-01 -> 2018-12-31 (era5_test_2017_2018.nc, da co san)
#   3 setting danh gia, cung ngan sach N=20 moi kien truc.
# ============================================================
SETTINGS = [
    ("h512p64", 512, 64),
    ("h512p128", 512, 128),
    ("h1024p96", 1024, 96),
]

N_BUDGET_TOTAL = 20
# Ti le N_fine:N_coarse = 3:1 cho cac kien truc 2-scale (Trend/Seasonal/
# Trend+Seasonal, Cluster+...) - dung chung quy uoc da thong nhat truoc do
# cho Spatial-Cluster BoN. Best-of-N+Physics dung het N_BUDGET_TOTAL cho 1
# nhanh duy nhat (khong co coarse).
N_FINE_DEFAULT = 15
N_COARSE_DEFAULT = 5


RESULTS_CSV_FIELDS = [
    "architecture", "setting", "history_steps", "prediction_steps", "n_windows",
    "n_fine", "n_coarse",
    "t2m_mse_norm_mean", "u10_mse_norm_mean", "v10_mse_norm_mean",
    "multivar_mse_norm_mean", "multivar_mse_norm_std",
]


def get_window_with_wind_history(loader: ERA5DataLoader, config: ERA5Config, start_idx: int) -> dict:
    """
    Mo rong get_test_window (run.py) de lay THEM lich su cua u10/v10 - ban
    goc chi co lich su t2m, vi truoc day gio tuong lai dung thang ground
    truth (khong can du bao). KHONG sua run.py - tu slice lai tai day,
    dung HET quy uoc windowing cua ERA5DataLoader.get_test_window.
    """
    h_end = start_idx + config.history_steps
    p_end = h_end + config.prediction_steps
    window = loader.ds.isel(time=slice(start_idx, p_end))

    return {
        "history_t2m": window[config.target_var].values[: config.history_steps],
        "gt_t2m": window[config.target_var].values[config.history_steps :],
        "history_u10": window[config.u_wind_var].values[: config.history_steps],
        "gt_u10": window[config.u_wind_var].values[config.history_steps :],
        "history_v10": window[config.v_wind_var].values[: config.history_steps],
        "gt_v10": window[config.v_wind_var].values[config.history_steps :],
        "times": window.time.values,
    }


def generate_multivar_candidates(forecaster, data: dict, prediction_steps: int, n_samples: int, temperature: float):
    """
    Goi ChronosProposer.predict() 3 LAN DOC LAP (t2m, u10, v10) - Chronos-1
    khong co co che joint-multivariate, moi bien van la 2048 chuoi doc lap.
    N candidate duoc ghep theo CHI SO i (t2m[i], u10[i], v10[i]) - chi la
    quy uoc ghep, khong phan anh lien he vat ly that giua 3 bien (da giai
    thich chi tiet trong hoi thoai thiet ke).

    Return: dict voi cac key t2m/u10/v10, moi key la tuple
            (denorm, norm, mean, std) dung format cua ChronosProposer.predict.
    """
    t2m = forecaster.predict(
        data["history_t2m"], prediction_steps=prediction_steps, temperature=temperature, n_samples=n_samples,
    )
    u10 = forecaster.predict(
        data["history_u10"], prediction_steps=prediction_steps, temperature=temperature, n_samples=n_samples,
    )
    v10 = forecaster.predict(
        data["history_v10"], prediction_steps=prediction_steps, temperature=temperature, n_samples=n_samples,
    )
    return {"t2m": t2m, "u10": u10, "v10": v10}


def _mse_norm_for(forecast_norm_best: np.ndarray, gt_denorm: np.ndarray, mean_flat: np.ndarray,
                   std_flat: np.ndarray, lat: int, lon: int) -> float:
    mean_spatial = mean_flat.reshape(lat, lon)
    std_spatial = std_flat.reshape(lat, lon)
    gt_norm = (gt_denorm - mean_spatial[np.newaxis]) / std_spatial[np.newaxis]
    return float(np.mean((forecast_norm_best - gt_norm) ** 2))


def compute_multivar_mse(candidates: dict, best_idx: int, data: dict, lat: int, lon: int) -> dict:
    """
    Chuan hoa (z-score) RIENG tung bien (moi bien co mean/std rieng tinh tu
    lich su cua chinh no), roi lay trung bing cong KHONG trong so cua 3 MSE
    da chuan hoa lam diem tong hop - dung quy uoc WeatherBench/GraphCast khi
    khong co ly do rieng de uu tien bien nao hon. KHONG duoc cong MSE tho
    (K vs m/s khac don vi).
    """
    t2m_denorm, t2m_norm, t2m_mean, t2m_std = candidates["t2m"]
    u10_denorm, u10_norm, u10_mean, u10_std = candidates["u10"]
    v10_denorm, v10_norm, v10_mean, v10_std = candidates["v10"]

    t2m_mse = _mse_norm_for(t2m_norm[best_idx], data["gt_t2m"], t2m_mean, t2m_std, lat, lon)
    u10_mse = _mse_norm_for(u10_norm[best_idx], data["gt_u10"], u10_mean, u10_std, lat, lon)
    v10_mse = _mse_norm_for(v10_norm[best_idx], data["gt_v10"], v10_mean, v10_std, lat, lon)
    multivar_mse = (t2m_mse + u10_mse + v10_mse) / 3.0

    return {
        "t2m_mse_norm": t2m_mse,
        "u10_mse_norm": u10_mse,
        "v10_mse_norm": v10_mse,
        "multivar_mse_norm": multivar_mse,
    }


def aggregate_window_metrics(window_metrics: list) -> dict:
    """window_metrics: list cac dict tra ve tu compute_multivar_mse, 1 phan tu/cua so."""
    def _mean(key):
        return float(np.mean([m[key] for m in window_metrics]))

    multivar_list = [m["multivar_mse_norm"] for m in window_metrics]
    return {
        "t2m_mse_norm_mean": _mean("t2m_mse_norm"),
        "u10_mse_norm_mean": _mean("u10_mse_norm"),
        "v10_mse_norm_mean": _mean("v10_mse_norm"),
        "multivar_mse_norm_mean": float(np.mean(multivar_list)),
        "multivar_mse_norm_std": float(np.std(multivar_list)),
    }


def append_csv_row(report_path: str, row: dict, fieldnames=RESULTS_CSV_FIELDS):
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(report_path)
    with open(report_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended 1 row -> {report_path}")


def build_config(history_steps: int, prediction_steps: int, model_id: str, data_path: str = None) -> ERA5Config:
    kwargs = dict(history_steps=history_steps, prediction_steps=prediction_steps, model_id=model_id)
    if data_path is not None:
        kwargs["data_path"] = data_path
    return ERA5Config(**kwargs)
