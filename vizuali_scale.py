import os
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Dict

from chronos import ChronosPipeline

# Nếu class này nằm trong file riêng thì uncomment:
# from era5_physics_energy import ERA5PhysicsEnergyScorer


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ERA5Config:

    data_path: str = "/home/user18/binhnkt/era5_test_2017_2018.nc"

    # 48 observations × 6 hours = 12 days
    history_steps: int = 512

    # 24 observations × 6 hours = 6 days
    prediction_steps: int = 64

    interval_hours: int = 6

    # Number of Chronos trajectories
    n_scenarios: int = 10

    grid_size: Tuple[int, int] = (32, 64)

    target_var: str = "2m_temperature"

    u_wind_var: str = "10m_u_component_of_wind"
    v_wind_var: str = "10m_v_component_of_wind"

    model_id: str = "amazon/chronos-t5-base"

    seed: int = 42

    # Grid point that we want to inspect
    inspect_lat: int = 16
    inspect_lon: int = 32


# ============================================================
# DATA LOADER
# ============================================================

class ERA5DataLoader:

    def __init__(self, config: ERA5Config):

        self.config = config
        self.ds = xr.open_dataset(config.data_path)
        self.ds = self.ds.transpose(
            "time",
            "latitude",
            "longitude"
        )

    def get_test_window(
        self,
        start_idx: int
    ) -> Dict[str, np.ndarray]:

        h_end = start_idx + self.config.history_steps
        p_end = h_end + self.config.prediction_steps

        window = self.ds.isel(
            time=slice(start_idx, p_end)
        )

        return {
            "history_t2m": window[self.config.target_var].values[:self.config.history_steps],
            "gt_t2m": window[self.config.target_var].values[self.config.history_steps:],
            "u10": window[self.config.u_wind_var].values[self.config.history_steps:],
            "v10": window[self.config.v_wind_var].values[self.config.history_steps:],
            "times": window.time.values
        }


# ============================================================
# CHRONOS PROPOSER
# ============================================================

class ChronosProposer:

    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-base",
        device: str = "cuda",
        batch_size: int = 16,
    ):

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.batch_size = batch_size

        print(f"Initializing ChronosProposer on {self.device}...")

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=self.device,
            torch_dtype=dtype,
        )

        print("Chronos loaded successfully.")

    def predict(
        self,
        history: np.ndarray,
        prediction_steps: int,
        temperature: float = 1.0,
        n_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        history_steps, lat, lon = history.shape
        n_series = lat * lon

        context_2d = history.reshape(history_steps, n_series).T.astype(np.float32)

        mean = np.mean(context_2d, axis=1, keepdims=True)
        std = np.std(context_2d, axis=1, keepdims=True)
        std = np.where(std < 1e-5, 1.0, std)

        normalized_context = (context_2d - mean) / std

        eff_temp = max(float(temperature), 1e-4)
        all_forecasts = []

        for start in range(0, n_series, self.batch_size):
            end = min(start + self.batch_size, n_series)
            batch_context = torch.from_numpy(normalized_context[start:end])

            with torch.inference_mode():
                forecast_tensor = self.pipeline.predict(
                    batch_context,
                    prediction_length=prediction_steps,
                    num_samples=n_samples,
                    top_k=50,
                    top_p=1.0,
                    temperature=eff_temp,
                    limit_prediction_length=False,
                )

            forecast_batch = forecast_tensor.cpu().numpy()
            all_forecasts.append(forecast_batch)

            del batch_context
            del forecast_tensor

            if self.device == "cuda":
                torch.cuda.empty_cache()

        forecast_norm = np.concatenate(all_forecasts, axis=0)

        mean_expanded = mean[:, np.newaxis, :]  
        std_expanded = std[:, np.newaxis, :]    

        forecast_denorm = forecast_norm * std_expanded + mean_expanded

        def reshape_to_spatial(arr):
            arr = arr.transpose(1, 2, 0)
            return arr.reshape(n_samples, prediction_steps, lat, lon)

        forecast_norm_spatial = reshape_to_spatial(forecast_norm)
        forecast_denorm_spatial = reshape_to_spatial(forecast_denorm)

        return forecast_denorm_spatial, forecast_norm_spatial, mean, std


# ============================================================
# MAIN EVALUATOR
# ============================================================

class PhysicsInformedEvaluator:

    def __init__(self, config: ERA5Config):
        self.config = config
        self.loader = ERA5DataLoader(config)
        self.forecaster = ChronosProposer(model_id=config.model_id)

    def downsample_series(self, data: np.ndarray, factor: int = 2) -> np.ndarray:
        """
        Thực hiện Average Pooling dọc theo trục thời gian.
        Input shape: (time, lat, lon)
        """
        t, lat, lon = data.shape
        new_t = t // factor
        data_cropped = data[-new_t * factor:]
        data_downsampled = data_cropped.reshape(new_t, factor, lat, lon).mean(axis=1)
        return data_downsampled

    def upsample_series(self, data: np.ndarray, factor: int = 2) -> np.ndarray:
        """
        Khôi phục độ phân giải bằng cách lặp lại giá trị (Repeat).
        Input shape: (time, lat, lon)
        """
        return np.repeat(data, factor, axis=0)

    def run_downsample_experiment(self, start_idx: int = 0):
        print("\n" + "=" * 70)
        print("BẮT ĐẦU THÍ NGHIỆM CHÉO MULTISCALE (A'' VÀ B'')")
        print("=" * 70)

        factor = 2

        # ----------------------------------------------------
        # 1. DỮ LIỆU LỊCH SỬ
        # ----------------------------------------------------
        data = self.loader.get_test_window(start_idx)
        series_A = data["history_t2m"]
        series_B = self.downsample_series(series_A, factor=factor)

        print(f"Lịch sử A (Raw)               : {series_A.shape}")
        print(f"Lịch sử B (Downsampled A)     : {series_B.shape}")

        # ----------------------------------------------------
        # 2. LUỒNG 1: Sinh A' và tạo A''
        # ----------------------------------------------------
        print("\nTiến hành dự báo Chuỗi A' (Từ A)...")
        pred_A_denorm, _, _, _ = self.forecaster.predict(
            history=series_A,
            prediction_steps=self.config.prediction_steps,
            temperature=0.0,
            n_samples=1
        )
        series_A_prime = pred_A_denorm[0]
        series_A_double_prime = self.downsample_series(series_A_prime, factor=factor)

        # ----------------------------------------------------
        # 3. LUỒNG 2: Sinh B' và tạo B''
        # ----------------------------------------------------
        print("Tiến hành dự báo Chuỗi B' (Từ B)...")
        b_prediction_steps = self.config.prediction_steps // factor
        pred_B_denorm, _, _, _ = self.forecaster.predict(
            history=series_B,
            prediction_steps=b_prediction_steps,
            temperature=0.0,
            n_samples=1,
        )
        series_B_prime = pred_B_denorm[0]
        series_B_double_prime = self.upsample_series(series_B_prime, factor=factor)

        print(f"\nDự báo A' (Raw)               : {series_A_prime.shape}")
        print(f"Dự báo A'' (Downsampled A')   : {series_A_double_prime.shape}")
        print(f"Dự báo B' (Raw từ B)          : {series_B_prime.shape}")
        print(f"Dự báo B'' (Upsampled B')     : {series_B_double_prime.shape}")

        # ----------------------------------------------------
        # 4. TRỰC QUAN HÓA
        # ----------------------------------------------------
        self.visualize_coarse_scale(series_B, series_B_prime, series_A_double_prime)
        # self.visualize_fine_scale(series_A, series_A_prime, series_B_double_prime)

    def visualize_coarse_scale(self, B, B_prime, A_double_prime):
        """ Visualization tại không gian Coarse Scale (So sánh B' và A'') """
        lat_idx, lon_idx = self.config.inspect_lat, self.config.inspect_lon

        point_B = B[:, lat_idx, lon_idx]
        point_B_prime = B_prime[:, lat_idx, lon_idx]
        point_A_double_prime = A_double_prime[:, lat_idx, lon_idx]

        time_hist = np.arange(len(point_B))
        time_pred = np.arange(len(point_B), len(point_B) + len(point_B_prime))

        plt.figure(figsize=(14, 6))

        # Lịch sử B
        plt.plot(time_hist, point_B, color="black", linewidth=2.5, label="Chuỗi B (Lịch sử)")

        # Kết nối
        plt.plot([time_hist[-1], time_pred[0]], [point_B[-1], point_B_prime[0]], color="red", linestyle="--", linewidth=2)
        plt.plot([time_hist[-1], time_pred[0]], [point_B[-1], point_A_double_prime[0]], color="blue", linewidth=2)

        # Dự báo tương lai
        plt.plot(time_pred, point_B_prime, color="red", linestyle="--", linewidth=2, label="Chuỗi B' (Dự báo từ B)")
        plt.plot(time_pred, point_A_double_prime, color="blue", linewidth=2, label="Chuỗi A'' (Downsampled từ A')")

        plt.axvline(time_hist[-1], color="gray", linestyle=":", linewidth=2, label="Forecast Horizon")
        plt.title(f"So sánh tại Coarse Scale (Tỷ lệ B)\nĐiểm lưới ({lat_idx}, {lon_idx})")
        plt.xlabel("Bước thời gian")
        plt.ylabel("Nhiệt độ (K)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_name = "multiscale_coarse_comparison_1024.png"
        plt.savefig(file_name, dpi=200)
        plt.show()
        print(f"\nĐã lưu: {file_name}")

    # def visualize_fine_scale(self, A, A_prime, B_double_prime):
        """ Visualization tại không gian Fine Scale (So sánh A' và B'') """
        lat_idx, lon_idx = self.config.inspect_lat, self.config.inspect_lon

        point_A = A[:, lat_idx, lon_idx]
        point_A_prime = A_prime[:, lat_idx, lon_idx]
        point_B_double_prime = B_double_prime[:, lat_idx, lon_idx]

        time_hist = np.arange(len(point_A))
        time_pred = np.arange(len(point_A), len(point_A) + len(point_A_prime))

        plt.figure(figsize=(14, 6))

        # Lịch sử A
        plt.plot(time_hist, point_A, color="black", linewidth=2.5, label="Chuỗi A (Lịch sử gốc)")

        # Kết nối
        plt.plot([time_hist[-1], time_pred[0]], [point_A[-1], point_A_prime[0]], color="blue", linewidth=2)
        plt.plot([time_hist[-1], time_pred[0]], [point_A[-1], point_B_double_prime[0]], color="red", linestyle="--", linewidth=2)

        # Dự báo tương lai
        plt.plot(time_pred, point_A_prime, color="blue", linewidth=2, label="Chuỗi A' (Dự báo từ A)")
        plt.plot(time_pred, point_B_double_prime, color="red", linestyle="--", linewidth=2, label="Chuỗi B'' (Upsampled từ B')")

        plt.axvline(time_hist[-1], color="gray", linestyle=":", linewidth=2, label="Forecast Horizon")
        plt.title(f"So sánh tại Fine Scale (Tỷ lệ A)\nĐiểm lưới ({lat_idx}, {lon_idx})")
        plt.xlabel("Bước thời gian")
        plt.ylabel("Nhiệt độ (K)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_name = "multiscale_fine_comparison_192.png"
        plt.savefig(file_name, dpi=200)
        plt.show()
        print(f"Đã lưu: {file_name}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    config = ERA5Config()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    evaluator = PhysicsInformedEvaluator(config)
    evaluator.run_downsample_experiment(start_idx=0)