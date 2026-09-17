# Cần cài trước:
#   pip install uni2ts gluonts
#
# Lưu ý: tên repo model trên HuggingFace có thể thay đổi theo phiên bản uni2ts
# (ví dụ "Salesforce/moirai-1.1-R-small", "-base", "-large"). Nếu gặp lỗi
# not-found khi from_pretrained, kiểm tra lại tên chính xác trên trang
# HuggingFace của Salesforce/uni2ts tại thời điểm chạy.

import os
import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Dict

from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# Nếu class này nằm trong file riêng thì uncomment:
# from era5_physics_energy import ERA5PhysicsEnergyScorer


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ERA5Config:

    data_path: str = "/home/user18/binhnkt/era5_test_2017_2018.nc"

    # 96 observations × 6 hours = 24 ngày lịch sử
    history_steps: int = 96

    # 48 observations × 6 hours = 12 ngày dự báo tương lai
    prediction_steps: int = 48

    interval_hours: int = 6

    # Number of Moirai sample trajectories
    n_scenarios: int = 10

    grid_size: Tuple[int, int] = (32, 64)

    target_var: str = "2m_temperature"

    u_wind_var: str = "10m_u_component_of_wind"
    v_wind_var: str = "10m_v_component_of_wind"

    # Model Moirai (uni2ts)
    model_id: str = "Salesforce/moirai-2.0-R-small"
    patch_size: str = "auto"  # hoặc số cụ thể: 8, 16, 32, 64, 128

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
# MOIRAI PROPOSER
# ============================================================

class MoiraiProposer:
    """
    Proposer dùng Moirai (uni2ts) thay cho Chronos.

    Giữ nguyên interface .predict(history, prediction_steps, temperature, n_samples)
    -> (forecast_denorm_spatial, forecast_norm_spatial, mean, std) để cắm thẳng
    vào PhysicsInformedEvaluator đã có, KHÔNG cần sửa phần còn lại của pipeline.

    Lưu ý quan trọng: Moirai không phải autoregressive token-based như Chronos,
    nên KHÔNG có khái niệm decoding `temperature`. Tham số `temperature` được
    giữ lại trong chữ ký hàm chỉ để tương thích interface — Moirai bỏ qua giá
    trị này. Độ đa dạng giữa các sample đến từ việc lấy mẫu trực tiếp từ phân
    phối xác suất (mixture distribution) mà model dự đoán, kiểm soát qua
    `n_samples`.
    """

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-1.1-R-small",
        device: str = "cuda",
        patch_size="auto",
        predictor_batch_size: int = 512,
    ):

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model_id = model_id
        self.patch_size = patch_size
        self.predictor_batch_size = predictor_batch_size

        print(f"Initializing MoiraiProposer ({model_id}) on {self.device}...")

        self.module = MoiraiModule.from_pretrained(model_id)

        print("Moirai module loaded successfully.")

    def predict(
        self,
        history: np.ndarray,
        prediction_steps: int,
        temperature: float = 1.0,
        n_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if temperature != 1.0:
            print(
                f"[MoiraiProposer] Lưu ý: temperature={temperature} bị bỏ qua "
                f"(Moirai không có decoding temperature)."
            )

        history_steps, lat, lon = history.shape
        n_series = lat * lon

        context_2d = history.reshape(history_steps, n_series).T.astype(np.float32)  # (n_series, history_steps)

        # mean/std chỉ dùng để trả về forecast_norm cho tương thích interface,
        # KHÔNG dùng để chuẩn hóa đầu vào -> Moirai tự chuẩn hóa patch nội bộ.
        mean = np.mean(context_2d, axis=1, keepdims=True)
        std = np.std(context_2d, axis=1, keepdims=True)
        std = np.where(std < 1e-5, 1.0, std)

        n_samples_eff = max(int(n_samples), 1)

        model = MoiraiForecast(
            module=self.module,
            prediction_length=prediction_steps,
            context_length=history_steps,
            patch_size=self.patch_size,
            num_samples=n_samples_eff,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=self.predictor_batch_size, device=self.device)

        # Mốc thời gian giả định chỉ để hợp lệ với GluonTS, không ảnh hưởng đến forecast
        start = pd.Period("2000-01-01", freq="6H")
        dataset = ListDataset(
            [{"start": start, "target": context_2d[i]} for i in range(n_series)],
            freq="6H",
        )

        forecasts = list(predictor.predict(dataset))

        # forecasts[i].samples có shape (n_samples_eff, prediction_steps)
        all_samples = np.stack([f.samples for f in forecasts], axis=0)  # (n_series, n_samples_eff, prediction_steps)

        forecast_denorm = all_samples.astype(np.float32)  # Moirai output đã ở đơn vị gốc

        mean_expanded = mean[:, np.newaxis, :]  # (n_series, 1, 1)
        std_expanded = std[:, np.newaxis, :]    # (n_series, 1, 1)

        forecast_norm = (forecast_denorm - mean_expanded) / std_expanded

        def reshape_to_spatial(arr):
            arr = arr.transpose(1, 2, 0)  # -> (n_samples, prediction_steps, n_series)
            return arr.reshape(n_samples_eff, prediction_steps, lat, lon)

        forecast_denorm_spatial = reshape_to_spatial(forecast_denorm)
        forecast_norm_spatial = reshape_to_spatial(forecast_norm)

        return forecast_denorm_spatial, forecast_norm_spatial, mean, std


# ============================================================
# MAIN EVALUATOR
# ============================================================

class PhysicsInformedEvaluator:

    def __init__(self, config: ERA5Config):
        self.config = config
        self.loader = ERA5DataLoader(config)
        self.forecaster = MoiraiProposer(
            model_id=config.model_id,
            patch_size=config.patch_size,
        )

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
        print("BẮT ĐẦU THÍ NGHIỆM CHÉO MULTISCALE (A'' VÀ B'') — MOIRAI")
        print("=" * 70)

        factor = 2

        # ----------------------------------------------------
        # 1. DỮ LIỆU LỊCH SỬ + GROUND TRUTH THỰC TẾ
        # ----------------------------------------------------
        data = self.loader.get_test_window(start_idx)
        series_A = data["history_t2m"]
        gt_A = data["gt_t2m"]

        series_B = self.downsample_series(series_A, factor=factor)
        # Downsample luôn ground truth để có bản đối chiếu ở coarse scale (tỷ lệ B)
        gt_B = self.downsample_series(gt_A, factor=factor)

        print(f"Lịch sử A (Raw)               : {series_A.shape}")
        print(f"Lịch sử B (Downsampled A)     : {series_B.shape}")
        print(f"Ground truth A (thực tế)      : {gt_A.shape}")
        print(f"Ground truth B (Downsampled)  : {gt_B.shape}")

        # ----------------------------------------------------
        # 2. LUỒNG 1: Sinh A' và tạo A''
        # ----------------------------------------------------
        print("\nTiến hành dự báo Chuỗi A' (Từ A)...")
        pred_A_denorm, _, _, _ = self.forecaster.predict(
            history=series_A,
            prediction_steps=self.config.prediction_steps,
            temperature=1.0,
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
            temperature=1.0,
            n_samples=10,
        )
        # Lấy trung bình các kịch bản để thu được dự báo B'
        series_B_prime = pred_B_denorm.mean(axis=0)
        series_B_double_prime = self.upsample_series(series_B_prime, factor=factor)

        print(f"\nDự báo A' (Raw)               : {series_A_prime.shape}")
        print(f"Dự báo A'' (Downsampled A')   : {series_A_double_prime.shape}")
        print(f"Dự báo B' (Raw từ B)          : {series_B_prime.shape}")
        print(f"Dự báo B'' (Upsampled B')     : {series_B_double_prime.shape}")

        # ----------------------------------------------------
        # 4. TRỰC QUAN HÓA (CÓ GROUND TRUTH ĐỂ ĐỐI CHIẾU)
        # ----------------------------------------------------
        self.visualize_coarse_scale(series_B, series_B_prime, series_A_double_prime, gt_B)
        self.visualize_fine_scale(series_A, series_A_prime, series_B_double_prime, gt_A)

    def visualize_coarse_scale(self, B, B_prime, A_double_prime, gt_B):
        """ Visualization tại không gian Coarse Scale (So sánh B' và A'' với ground truth) """
        lat_idx, lon_idx = self.config.inspect_lat, self.config.inspect_lon

        point_B = B[:, lat_idx, lon_idx]
        point_B_prime = B_prime[:, lat_idx, lon_idx]
        point_A_double_prime = A_double_prime[:, lat_idx, lon_idx]
        point_gt_B = gt_B[:, lat_idx, lon_idx]

        time_hist = np.arange(len(point_B))
        time_pred = np.arange(len(point_B), len(point_B) + len(point_B_prime))

        plt.figure(figsize=(14, 6))

        # Lịch sử B
        plt.plot(time_hist, point_B, color="black", linewidth=2.5, label="Chuỗi B (Lịch sử)")

        # Ground truth thực tế (đã downsample về tỷ lệ B)
        plt.plot(
            [time_hist[-1], time_pred[0]], [point_B[-1], point_gt_B[0]],
            color="green", linewidth=2.2,
        )
        plt.plot(time_pred, point_gt_B, color="green", linewidth=2.2, label="Ground truth (thực tế, tỷ lệ B)")

        # Kết nối
        plt.plot([time_hist[-1], time_pred[0]], [point_B[-1], point_B_prime[0]], color="red", linestyle="--", linewidth=2)
        plt.plot([time_hist[-1], time_pred[0]], [point_B[-1], point_A_double_prime[0]], color="blue", linewidth=2)

        # Dự báo tương lai
        plt.plot(time_pred, point_B_prime, color="red", linestyle="--", linewidth=2, label="Chuỗi B' (Dự báo từ B)")
        plt.plot(time_pred, point_A_double_prime, color="blue", linewidth=2, label="Chuỗi A'' (Downsampled từ A')")

        plt.axvline(time_hist[-1], color="gray", linestyle=":", linewidth=2, label="Forecast Horizon")
        plt.title(f"[Moirai] So sánh tại Coarse Scale (Tỷ lệ B)\nĐiểm lưới ({lat_idx}, {lon_idx})")
        plt.xlabel("Bước thời gian")
        plt.ylabel("Nhiệt độ (K)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_name = "moirai_multiscale_coarse_comparison.png"
        plt.savefig(file_name, dpi=200)
        plt.show()
        print(f"\nĐã lưu: {file_name}")

    def visualize_fine_scale(self, A, A_prime, B_double_prime, gt_A):
        """ Visualization tại không gian Fine Scale (So sánh A' và B'' với ground truth) """
        lat_idx, lon_idx = self.config.inspect_lat, self.config.inspect_lon

        point_A = A[:, lat_idx, lon_idx]
        point_A_prime = A_prime[:, lat_idx, lon_idx]
        point_B_double_prime = B_double_prime[:, lat_idx, lon_idx]
        point_gt_A = gt_A[:, lat_idx, lon_idx]

        time_hist = np.arange(len(point_A))
        time_pred = np.arange(len(point_A), len(point_A) + len(point_A_prime))

        plt.figure(figsize=(14, 6))

        # Lịch sử A
        plt.plot(time_hist, point_A, color="black", linewidth=2.5, label="Chuỗi A (Lịch sử gốc)")

        # Ground truth thực tế (tỷ lệ A)
        plt.plot(
            [time_hist[-1], time_pred[0]], [point_A[-1], point_gt_A[0]],
            color="green", linewidth=2.2,
        )
        plt.plot(time_pred, point_gt_A, color="green", linewidth=2.2, label="Ground truth (thực tế)")

        # Kết nối
        plt.plot([time_hist[-1], time_pred[0]], [point_A[-1], point_A_prime[0]], color="blue", linewidth=2)
        plt.plot([time_hist[-1], time_pred[0]], [point_A[-1], point_B_double_prime[0]], color="red", linestyle="--", linewidth=2)

        # Dự báo tương lai
        plt.plot(time_pred, point_A_prime, color="blue", linewidth=2, label="Chuỗi A' (Dự báo từ A)")
        plt.plot(time_pred, point_B_double_prime, color="red", linestyle="--", linewidth=2, label="Chuỗi B'' (Upsampled từ B')")

        plt.axvline(time_hist[-1], color="gray", linestyle=":", linewidth=2, label="Forecast Horizon")
        plt.title(f"[Moirai] So sánh tại Fine Scale (Tỷ lệ A)\nĐiểm lưới ({lat_idx}, {lon_idx})")
        plt.xlabel("Bước thời gian")
        plt.ylabel("Nhiệt độ (K)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_name = "moirai_multiscale_fine_comparison.png"
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