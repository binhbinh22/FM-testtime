import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Dict

from chronos import ChronosPipeline


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ERA5Config:

    data_path: str = "/home/user18/binhnkt/era5_test_2017_2018.nc"

    # Cập nhật chiều dài lịch sử theo yêu cầu (512 bước)
    history_steps: int = 512

    # 24 observations × 6 hours = 6 days
    prediction_steps: int = 128

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
    # === THÊM MỚI: cấu hình cho Scale C (coarse) ===
    downsample_factor: int = 2      # 6h × 2 = 12h/bước → chuyển scale A (6h) sang scale C (12h/bước)
    kernel_size_A: int = 24         # kernel decomposition ở scale A (fine)

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
# DOWNSAMPLER (tạo Scale C từ Scale A)
# ============================================================

class DownSampler:
    """
    Downsample theo đúng công thức TimeMixer: average pooling,
    factor = 2^m tại mỗi scale m.
    x_m ∈ R^⌊P/2^m⌋×C
    """
    @staticmethod
    def downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
        t = x.shape[0]
        usable_len = (t // factor) * factor
        x_trimmed = x[:usable_len]
        new_shape = (usable_len // factor, factor) + tuple(x.shape[1:])
        x_reshaped = x_trimmed.reshape(new_shape)
        return x_reshaped.mean(dim=1)

    @staticmethod
    def multiscale(x: torch.Tensor, M: int):
        """
        Sinh ra toàn bộ tập multiscale X = {x_0, ..., x_M} theo đúng TimeMixer.
        x_0 = x (chuỗi gốc, mịn nhất)
        x_m = downsample(x, factor=2^m)
        """
        scales = {0: x}
        for m in range(1, M + 1):
            factor = 2 ** m
            scales[m] = DownSampler.downsample(x, factor=factor)
        return scales

# ============================================================
# MOVING AVERAGE DECOMPOSITION (Autoformer / TimeMixer Style)
# ============================================================

class MovingAvgDecomp(nn.Module):
    """
    Phân rã chuỗi thời gian thành Trend và Seasonal bằng Moving Average.
    Áp dụng chuẩn lý thuyết từ các paper Autoformer, FEDformer và TimeMixer.
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input x shape: (time, lat, lon) hoặc (time,)
        """
        # Padding đối xứng hai đầu để giữ nguyên kích thước chuỗi
        front = x[:1].repeat(self.kernel_size // 2, *([1] * (x.dim() - 1)))
        end = x[-1:].repeat(self.kernel_size - self.kernel_size // 2 - 1, *([1] * (x.dim() - 1)))
        
        x_padded = torch.cat([front, x, end], dim=0)
        
        if x.dim() == 3:  # Định dạng (time, lat, lon)
            t, lat, lon = x_padded.shape
            x_padded = x_padded.permute(1, 2, 0).reshape(lat * lon, 1, t)
            trend = self.avg(x_padded)
            trend = trend.reshape(lat, lon, -1).permute(2, 0, 1)
        else:  # Định dạng 1D cho từng điểm lưới đơn lẻ: (time,)
            x_padded = x_padded.unsqueeze(0).unsqueeze(0)
            trend = self.avg(x_padded).squeeze(0).squeeze(0)

        seasonal = x - trend
        return trend, seasonal


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

    def run_decomposition_experiment(self, start_idx: int = 0):
        print("\n" + "=" * 70)
        print("BẮT ĐẦU THÍ NGHIỆM PHÂN RÃ CHUỖI A (TREND & SEASONAL) - 512 BƯỚC")
        print("=" * 70)

        # 1. Lấy dữ liệu lịch sử A
        data = self.loader.get_test_window(start_idx)
        series_A = torch.tensor(data["history_t2m"], dtype=torch.float32)

        print(f"Kích thước lịch sử A         : {series_A.shape}")

        # 2. Thực hiện phân rã (Kernel size = 24 tương ứng chu kỳ 1 ngày với dữ liệu 6 tiếng/lần)
        kernel_size = 24
        decomp = MovingAvgDecomp(kernel_size=kernel_size)
        trend_A, seasonal_A = decomp(series_A)

        print(f"Kích thước thành phần Trend   : {trend_A.shape}")
        print(f"Kích thước thành phần Seasonal: {seasonal_A.shape}")

        # 3. Trực quan hóa
        self.visualize_decomposition(series_A, trend_A, seasonal_A)

    def visualize_decomposition(self, A, trend_A, seasonal_A):
        lat_idx, lon_idx = self.config.inspect_lat, self.config.inspect_lon

        point_A = A[:, lat_idx, lon_idx].numpy()
        point_trend = trend_A[:, lat_idx, lon_idx].numpy()
        point_seasonal = seasonal_A[:, lat_idx, lon_idx].numpy()

        time_steps = np.arange(len(point_A))

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(time_steps, point_A, color="black", linewidth=1.5)
        axes[0].set_title(f"Chuỗi A Gốc (Raw Temperature) tại điểm lưới ({lat_idx}, {lon_idx}) - History Length: 512")
        axes[0].set_ylabel("Nhiệt độ (K)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_steps, point_trend, color="blue", linewidth=2)
        axes[1].set_title(f"Thành phần Xu hướng (Trend - Moving Average, kernel={24})")
        axes[1].set_ylabel("Nhiệt độ (K)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(time_steps, point_seasonal, color="red", linewidth=1.2)
        axes[2].set_title("Thành phần Mùa vụ / Chu kỳ (Seasonal = Raw - Trend)")
        axes[2].set_xlabel("Bước thời gian")
        axes[2].set_ylabel("Biên độ (K)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        file_name = "decomposition_A_trend_seasonal_512.png"
        plt.savefig(file_name, dpi=200)
        plt.show()
        print(f"\nĐã lưu biểu đồ phân rã: {file_name}")

    def run_decomposition_experiment_scaleC(self, start_idx: int = 0):
        print("\n" + "=" * 70)
        print("BẮT ĐẦU THÍ NGHIỆM PHÂN RÃ CHUỖI C (TREND & SEASONAL) - SCALE COARSE")
        print("=" * 70)

        # 1. Lấy dữ liệu lịch sử A (giống scale A)
        data = self.loader.get_test_window(start_idx)
        series_A = torch.tensor(data["history_t2m"], dtype=torch.float32)

        # 2. Downsample A -> C
        factor = self.config.downsample_factor
        series_C = DownSampler.downsample(series_A, factor=factor)

        print(f"Kích thước lịch sử A (fine)   : {series_A.shape}")
        print(f"Downsample factor              : {factor}")
        print(f"Kích thước lịch sử C (coarse)  : {series_C.shape}")

        # 3. Decompose C — kernel scale theo đúng tỉ lệ downsample so với kernel của A
        kernel_size_C = max(2, self.config.kernel_size_A // factor)
        print(f"Kernel size dùng cho Scale C   : {kernel_size_C}")

        decomp_C = MovingAvgDecomp(kernel_size=kernel_size_C)
        trend_C, seasonal_C = decomp_C(series_C)

        print(f"Kích thước thành phần Trend_C   : {trend_C.shape}")
        print(f"Kích thước thành phần Seasonal_C: {seasonal_C.shape}")

        # 4. Trực quan hóa riêng Scale C
        self.visualize_decomposition_scaleC(series_C, trend_C, seasonal_C)

        # 5. Trực quan hóa so sánh A vs C (đặt cạnh nhau)
        self.visualize_compare_scales(series_A, series_C, trend_C, seasonal_C)

        return series_C, trend_C, seasonal_C

    def visualize_decomposition_scaleC(self, C, trend_C, seasonal_C):
        lat_idx, lon_idx = self.config.inspect_lat, self.config.inspect_lon

        point_C = C[:, lat_idx, lon_idx].numpy()
        point_trend = trend_C[:, lat_idx, lon_idx].numpy()
        point_seasonal = seasonal_C[:, lat_idx, lon_idx].numpy()

        time_steps = np.arange(len(point_C))

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(time_steps, point_C, color="black", linewidth=1.5)
        axes[0].set_title(f"Chuỗi C Coarse (Downsampled x{self.config.downsample_factor}) tại điểm lưới ({lat_idx}, {lon_idx})")
        axes[0].set_ylabel("Nhiệt độ (K)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_steps, point_trend, color="blue", linewidth=2)
        axes[1].set_title("Thành phần Xu hướng (Trend_C - Moving Average)")
        axes[1].set_ylabel("Nhiệt độ (K)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(time_steps, point_seasonal, color="red", linewidth=1.2)
        axes[2].set_title("Thành phần Mùa vụ / Chu kỳ (Seasonal_C = Raw_C - Trend_C)")
        axes[2].set_xlabel("Bước thời gian (coarse)")
        axes[2].set_ylabel("Biên độ (K)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        file_name = "decomposition_C_trend_seasonal.png"
        plt.savefig(file_name, dpi=200)
        plt.show()
        print(f"\nĐã lưu biểu đồ phân rã Scale C: {file_name}")

    def visualize_compare_scales(self, A, C, trend_C, seasonal_C):
        """
        So sánh trực quan giữa Scale A (fine, gốc) và Scale C (coarse, downsampled)
        trên cùng 1 hình để thấy rõ quan hệ giữa 2 độ phân giải.
        """
        lat_idx, lon_idx = self.config.inspect_lat, self.config.inspect_lon
        factor = self.config.downsample_factor

        point_A = A[:, lat_idx, lon_idx].numpy()
        point_C = C[:, lat_idx, lon_idx].numpy()

        # Trục thời gian của C quy đổi về cùng đơn vị bước với A để so sánh đúng vị trí
        time_A = np.arange(len(point_A))
        time_C = np.arange(len(point_C)) * factor

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(time_A, point_A, color="gray", alpha=0.5, linewidth=1, label="Scale A (fine, gốc)")
        ax.plot(time_C, point_C, color="darkorange", linewidth=2.2, marker="o", markersize=3,
                label=f"Scale C (coarse, downsample x{factor})")

        ax.set_title(f"So sánh Scale A vs Scale C tại điểm lưới ({lat_idx}, {lon_idx})")
        ax.set_xlabel("Bước thời gian (theo đơn vị Scale A)")
        ax.set_ylabel("Nhiệt độ (K)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        file_name = "compare_scaleA_scaleC.png"
        plt.savefig(file_name, dpi=200)
        plt.show()
        print(f"\nĐã lưu biểu đồ so sánh A vs C: {file_name}")
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

    evaluator.run_decomposition_experiment(start_idx=0)

    # Scale C (coarse) — thêm mới
    evaluator.run_decomposition_experiment_scaleC(start_idx=0)