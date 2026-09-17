import os
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataclasses import dataclass
from typing import Tuple, Dict, List

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
    history_steps: int = 192

    # 24 observations × 6 hours = 6 days
    prediction_steps: int = 96

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

    # ----------------------------------------------------
    # Cấu hình cho thí nghiệm N-candidate với temperature ngẫu nhiên
    # ----------------------------------------------------
    n_candidates: int = 10
    temp_min: float = 0.3
    temp_max: float = 1.5


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

    def run_B_candidates_experiment(self, start_idx: int = 0):
        """
        Thí nghiệm sinh N candidate cho chuỗi B, mỗi candidate dùng một
        temperature ngẫu nhiên riêng (lấy uniform trong [temp_min, temp_max]),
        rồi visualize toàn bộ candidate dạng spaghetti plot cùng với B (lịch sử)
        và đường mean của các candidate.
        """
        print("\n" + "=" * 70)
        print(f"SINH {self.config.n_candidates} CANDIDATE CHO CHUỖI B (TEMPERATURE NGẪU NHIÊN)")
        print("=" * 70)

        factor = 2

        # ----------------------------------------------------
        # 1. DỮ LIỆU LỊCH SỬ A -> DOWNSAMPLE THÀNH B
        # ----------------------------------------------------
        data = self.loader.get_test_window(start_idx)
        series_A = data["history_t2m"]
        series_B = self.downsample_series(series_A, factor=factor)

        print(f"Lịch sử A (Raw)               : {series_A.shape}")
        print(f"Lịch sử B (Downsampled A)     : {series_B.shape}")

        # ----------------------------------------------------
        # 2. SINH N CANDIDATE, MỖI CANDIDATE 1 TEMPERATURE NGẪU NHIÊN
        # ----------------------------------------------------
        b_prediction_steps = self.config.prediction_steps // factor

        rng = np.random.default_rng(self.config.seed)
        temperatures = rng.uniform(
            self.config.temp_min,
            self.config.temp_max,
            size=self.config.n_candidates,
        )

        candidates: List[np.ndarray] = []

        for i, temp in enumerate(temperatures):
            print(f"\nCandidate {i + 1}/{self.config.n_candidates} — temperature = {temp:.3f}")
            pred_B_denorm, _, _, _ = self.forecaster.predict(
                history=series_B,
                prediction_steps=b_prediction_steps,
                temperature=float(temp),
                n_samples=1,
            )
            candidates.append(pred_B_denorm[0])  # (prediction_steps, lat, lon)

        candidates_arr = np.stack(candidates, axis=0)  # (n_candidates, prediction_steps, lat, lon)
        candidates_mean = candidates_arr.mean(axis=0)  # (prediction_steps, lat, lon)

        print(f"\nLịch sử B                     : {series_B.shape}")
        print(f"Tất cả candidate B'           : {candidates_arr.shape}")
        print(f"Mean của các candidate         : {candidates_mean.shape}")

        # ----------------------------------------------------
        # 3. TRỰC QUAN HÓA DẠNG SPAGHETTI PLOT
        # ----------------------------------------------------
        self.visualize_B_candidates(series_B, candidates_arr, candidates_mean, temperatures)

    def visualize_B_candidates(
        self,
        B: np.ndarray,
        candidates: np.ndarray,
        candidates_mean: np.ndarray,
        temperatures: np.ndarray,
    ):
        """
        Spaghetti plot: lịch sử B + từng candidate B' (mỗi đường ứng với 1 temperature
        ngẫu nhiên riêng, tô màu theo giá trị temperature) + đường mean của tất cả candidate.
        """
        lat_idx, lon_idx = self.config.inspect_lat, self.config.inspect_lon

        point_B = B[:, lat_idx, lon_idx]
        n_candidates = candidates.shape[0]
        points_candidates = candidates[:, :, lat_idx, lon_idx]  # (n_candidates, prediction_steps)
        point_mean = candidates_mean[:, lat_idx, lon_idx]

        time_hist = np.arange(len(point_B))
        time_pred = np.arange(len(point_B), len(point_B) + points_candidates.shape[1])

        plt.figure(figsize=(14, 7))

        # Lịch sử B
        plt.plot(time_hist, point_B, color="black", linewidth=2.5, label="Chuỗi B (Lịch sử)", zorder=5)

        # Colormap theo temperature để dễ quan sát ảnh hưởng của temperature
        norm = plt.Normalize(vmin=temperatures.min(), vmax=temperatures.max())
        cmap = cm.get_cmap("plasma")

        for i in range(n_candidates):
            color = cmap(norm(temperatures[i]))

            # Đoạn nối lịch sử -> candidate
            plt.plot(
                [time_hist[-1], time_pred[0]],
                [point_B[-1], points_candidates[i, 0]],
                color=color, linewidth=1.2, alpha=0.6, zorder=2,
            )

            plt.plot(
                time_pred, points_candidates[i],
                color=color, linewidth=1.4, alpha=0.75, zorder=2,
                label=f"T={temperatures[i]:.2f}" if n_candidates <= 12 else None,
            )

        # Đường mean của tất cả candidate
        plt.plot(
            [time_hist[-1], time_pred[0]],
            [point_B[-1], point_mean[0]],
            color="black", linestyle="--", linewidth=2.5, zorder=6,
        )
        plt.plot(
            time_pred, point_mean,
            color="black", linestyle="--", linewidth=2.5,
            label=f"Mean của {n_candidates} candidate", zorder=6,
        )

        # Colorbar cho temperature
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label("Temperature")

        plt.axvline(time_hist[-1], color="gray", linestyle=":", linewidth=2, label="Forecast Horizon")
        plt.title(
            f"Spaghetti plot: {n_candidates} candidate của B' (temperature ngẫu nhiên "
            f"trong [{self.config.temp_min}, {self.config.temp_max}])\n"
            f"Điểm lưới ({lat_idx}, {lon_idx})"
        )
        plt.xlabel("Bước thời gian")
        plt.ylabel("Nhiệt độ (K)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left", fontsize=8, ncol=2)
        plt.tight_layout()

        file_name = "B_candidates_spaghetti_192.png"
        plt.savefig(file_name, dpi=200)
        plt.show()
        print(f"\nĐã lưu: {file_name}")


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
    evaluator.run_B_candidates_experiment(start_idx=0)