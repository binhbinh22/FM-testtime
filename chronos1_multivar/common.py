import csv
import os
import sys

import numpy as np
import xarray as xr

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
    "t2m_rmse_denorm_mean", "u10_rmse_denorm_mean", "v10_rmse_denorm_mean",
    "rmse_denorm_mean", "rmse_denorm_std",
    "t2m_acc_denorm_mean", "u10_acc_denorm_mean", "v10_acc_denorm_mean",
    "acc_denorm_mean", "acc_denorm_std",
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


def compute_latitude_weights(lat_deg: np.ndarray) -> np.ndarray:
    """
    alpha(h) = cos(h) / mean(cos(h)) - dung quy uoc cua ClimODE (Verma et al.,
    2024, ICLR; cong thuc 13/15) de bu tru dien tich o luoi co lai gan cuc.
    """
    w = np.cos(np.deg2rad(lat_deg))
    return w / w.mean()


def compute_climatology(climatology_path: str, config: ERA5Config) -> dict:
    """
    Climatology = trung binh theo thoi gian (KHONG phan biet ngay/thang
    trong nam - dung dung dinh nghia "empirical mean" cua ClimODE) tinh TU
    TAP VAL (tach biet voi test, khong ro ri thong tin) - repo nay khong co
    10 nam train rieng nhu paper goc nen dung val lam nguon climatology
    chung cho moi lan danh gia (ca khi dang chay tren val lan test).
    """
    ds = xr.open_dataset(climatology_path).transpose("time", "latitude", "longitude")
    return {
        "t2m": ds[config.target_var].mean(dim="time").values,
        "u10": ds[config.u_wind_var].mean(dim="time").values,
        "v10": ds[config.v_wind_var].mean(dim="time").values,
    }


def _rmse_acc_denorm_for(pred_denorm: np.ndarray, gt_denorm: np.ndarray, climatology_spatial: np.ndarray,
                          lat_weight_col: np.ndarray) -> tuple:
    """
    pred_denorm, gt_denorm: (prediction_steps, lat, lon), gia tri vat ly
    that (K hoac m/s). Cong thuc dung dinh nghia ClimODE (cong thuc 13/15):
    RMSE = (1/T) sum_t sqrt(mean_hw[alpha(h)*(y-u)^2])
    ACC  = sum_thw[alpha(h)*y~*u~] / sqrt(sum_thw[alpha(h)*y~^2] * sum_thw[alpha(h)*u~^2])
    """
    diff_sq = lat_weight_col[np.newaxis] * (gt_denorm - pred_denorm) ** 2
    rmse = float(np.mean([np.sqrt(np.mean(diff_sq[t])) for t in range(diff_sq.shape[0])]))

    y_anom = gt_denorm - climatology_spatial[np.newaxis]
    u_anom = pred_denorm - climatology_spatial[np.newaxis]
    w = lat_weight_col[np.newaxis]
    numerator = np.sum(w * y_anom * u_anom)
    denominator = np.sqrt(np.sum(w * y_anom ** 2) * np.sum(w * u_anom ** 2))
    acc = float(numerator / denominator) if denominator > 0 else float("nan")

    return rmse, acc


def compute_multivar_mse(candidates: dict, best_idx: int, data: dict, lat: int, lon: int,
                          lat_weight: np.ndarray = None, climatology: dict = None) -> dict:
    """
    Chuan hoa (z-score) RIENG tung bien (moi bien co mean/std rieng tinh tu
    lich su cua chinh no), roi lay trung bing cong KHONG trong so cua 3 MSE
    da chuan hoa lam diem tong hop - dung quy uoc WeatherBench/GraphCast khi
    khong co ly do rieng de uu tien bien nao hon. KHONG duoc cong MSE tho
    (K vs m/s khac don vi).

    Neu lat_weight + climatology duoc truyen vao, tinh THEM RMSE va ACC tren
    gia tri denorm (vat ly that), co trong so vi do, theo dung dinh nghia
    ClimODE (Verma et al., 2024) - doc lap voi MSE norm o tren, khong thay
    the no.
    """
    t2m_denorm, t2m_norm, t2m_mean, t2m_std = candidates["t2m"]
    u10_denorm, u10_norm, u10_mean, u10_std = candidates["u10"]
    v10_denorm, v10_norm, v10_mean, v10_std = candidates["v10"]

    t2m_mse = _mse_norm_for(t2m_norm[best_idx], data["gt_t2m"], t2m_mean, t2m_std, lat, lon)
    u10_mse = _mse_norm_for(u10_norm[best_idx], data["gt_u10"], u10_mean, u10_std, lat, lon)
    v10_mse = _mse_norm_for(v10_norm[best_idx], data["gt_v10"], v10_mean, v10_std, lat, lon)
    multivar_mse = (t2m_mse + u10_mse + v10_mse) / 3.0

    result = {
        "t2m_mse_norm": t2m_mse,
        "u10_mse_norm": u10_mse,
        "v10_mse_norm": v10_mse,
        "multivar_mse_norm": multivar_mse,
    }

    if lat_weight is not None and climatology is not None:
        lat_weight_col = lat_weight.reshape(-1, 1)
        denorm_by_var = {"t2m": t2m_denorm[best_idx], "u10": u10_denorm[best_idx], "v10": v10_denorm[best_idx]}
        gt_by_var = {"t2m": data["gt_t2m"], "u10": data["gt_u10"], "v10": data["gt_v10"]}

        rmse_vals, acc_vals = [], []
        for var in ("t2m", "u10", "v10"):
            rmse, acc = _rmse_acc_denorm_for(
                denorm_by_var[var], gt_by_var[var], climatology[var], lat_weight_col,
            )
            result[f"{var}_rmse_denorm"] = rmse
            result[f"{var}_acc_denorm"] = acc
            rmse_vals.append(rmse)
            acc_vals.append(acc)

        result["rmse_denorm_mean"] = float(np.mean(rmse_vals))
        result["acc_denorm_mean"] = float(np.mean(acc_vals))

    return result


def aggregate_window_metrics(window_metrics: list) -> dict:
    """window_metrics: list cac dict tra ve tu compute_multivar_mse, 1 phan tu/cua so."""
    def _mean(key):
        return float(np.mean([m[key] for m in window_metrics]))

    def _std(key):
        return float(np.std([m[key] for m in window_metrics]))

    multivar_list = [m["multivar_mse_norm"] for m in window_metrics]
    result = {
        "t2m_mse_norm_mean": _mean("t2m_mse_norm"),
        "u10_mse_norm_mean": _mean("u10_mse_norm"),
        "v10_mse_norm_mean": _mean("v10_mse_norm"),
        "multivar_mse_norm_mean": float(np.mean(multivar_list)),
        "multivar_mse_norm_std": float(np.std(multivar_list)),
    }

    if "rmse_denorm_mean" in window_metrics[0]:
        result.update({
            "t2m_rmse_denorm_mean": _mean("t2m_rmse_denorm"),
            "u10_rmse_denorm_mean": _mean("u10_rmse_denorm"),
            "v10_rmse_denorm_mean": _mean("v10_rmse_denorm"),
            "rmse_denorm_mean": _mean("rmse_denorm_mean"),
            "rmse_denorm_std": _std("rmse_denorm_mean"),
            "t2m_acc_denorm_mean": _mean("t2m_acc_denorm"),
            "u10_acc_denorm_mean": _mean("u10_acc_denorm"),
            "v10_acc_denorm_mean": _mean("v10_acc_denorm"),
            "acc_denorm_mean": _mean("acc_denorm_mean"),
            "acc_denorm_std": _std("acc_denorm_mean"),
        })

    return result


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
