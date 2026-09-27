import numpy as np

from run import ERA5DataLoader, ERA5Config
from multiscale_trend_seasonal_search import MultiscaleTrendSeasonalSearchEngine


def build_shared_start_indices(
    loader: ERA5DataLoader, history_steps: int, prediction_steps: int, n_windows: int
) -> np.ndarray:
    """
    Tinh 1 LAN duy nhat bo start_indices chia deu tren toan bo do dai du
    lieu cua `loader`. Goi ham nay 1 LAN roi dung CHUNG ket qua cho moi o
    luoi hyperparameter (N, K, w1, w2, ...) trong cung 1 sweep, de dam
    bao so sanh cong bang giua cac cau hinh (khong doi cua so giua chung).
    """
    max_start = len(loader.ds.time) - (history_steps + prediction_steps)
    if max_start <= 0:
        raise ValueError("history_steps + prediction_steps vuot qua do dai du lieu.")
    return np.linspace(0, max_start - 1, n_windows, dtype=int)


def evaluate_config(
    engine: MultiscaleTrendSeasonalSearchEngine,
    loader: ERA5DataLoader,
    config: ERA5Config,
    start_indices: np.ndarray,
) -> dict:
    """
    Chay engine tren tap start_indices cho truoc, tra ve mean +- std cua
    MSE va MAE (norm & denorm) tren cac cua so do.
    """
    mse_norm_list, mse_denorm_list = [], []
    mae_norm_list, mae_denorm_list = [], []

    for start_idx in start_indices:
        data = loader.get_test_window(int(start_idx))
        result = engine.run(
            history=data["history_t2m"],
            u_wind_future=data["u10"],
            v_wind_future=data["v10"],
            prediction_steps=config.prediction_steps,
            interval_hours=config.interval_hours,
            ground_truth=data["gt_t2m"],
        )
        m = result["metrics"]
        mse_norm_list.append(m["mse_norm"])
        mse_denorm_list.append(m["mse_denorm"])
        mae_norm_list.append(m["mae_norm"])
        mae_denorm_list.append(m["mae_denorm"])

    return {
        "mse_norm_mean": float(np.mean(mse_norm_list)),
        "mse_norm_std": float(np.std(mse_norm_list)),
        "mse_denorm_mean": float(np.mean(mse_denorm_list)),
        "mse_denorm_std": float(np.std(mse_denorm_list)),
        "mae_norm_mean": float(np.mean(mae_norm_list)),
        "mae_norm_std": float(np.std(mae_norm_list)),
        "mae_denorm_mean": float(np.mean(mae_denorm_list)),
        "mae_denorm_std": float(np.std(mae_denorm_list)),
    }
