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
    history_steps: int = 96

    # 24 observations × 6 hours = 6 days
    prediction_steps: int = 48

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

    def run_A_candidates_experiment(self, start_idx: int = 0):
        """
        Thí nghiệm đối chứng: sinh N candidate cho chuỗi A (KHÔNG downsample),
        mỗi candidate dùng một temperature ngẫu nhiên riêng (uniform trong
        [temp_min, temp_max]) — y hệt cách làm với B trước đó — rồi visualize
        dạng spaghetti plot cùng với ground truth thực tế (gt_t2m) để đối chiếu
        xem A có bị thu hẹp độ bất định (underdispersion) giống B hay không.
        """
        print("\n" + "=" * 70)
        print(f"SINH {self.config.n_candidates} CANDIDATE CHO CHUỖI A (ĐỐI CHỨNG VỚI B)")
        print("=" * 70)

        # ----------------------------------------------------
        # 1. DỮ LIỆU LỊCH SỬ A + GROUND TRUTH THỰC TẾ
        # ----------------------------------------------------
        data = self.loader.get_test_window(start_idx)
        series_A = data["history_t2m"]
        gt_A = data["gt_t2m"]

        print(f"Lịch sử A (Raw)               : {series_A.shape}")
        print(f"Ground truth A (thực tế)      : {gt_A.shape}")

        # ----------------------------------------------------
        # 2. SINH N CANDIDATE, MỖI CANDIDATE 1 TEMPERATURE NGẪU NHIÊN
        # ----------------------------------------------------
        rng = np.random.default_rng(self.config.seed)
        temperatures = rng.uniform(
            self.config.temp_min,
            self.config.temp_max,
            size=self.config.n_candidates,
        )

        candidates: List[np.ndarray] = []

        for i, temp in enumerate(temperatures):
            print(f"\nCandidate {i + 1}/{self.config.n_candidates} — temperature = {temp:.3f}")
            pred_A_denorm, _, _, _ = self.forecaster.predict(
                history=series_A,
                prediction_steps=self.config.prediction_steps,
                temperature=float(temp),
                n_samples=1,
            )
            candidates.append(pred_A_denorm[0])  # (prediction_steps, lat, lon)

        candidates_arr = np.stack(candidates, axis=0)  # (n_candidates, prediction_steps, lat, lon)
        candidates_mean = candidates_arr.mean(axis=0)  # (prediction_steps, lat, lon)

        print(f"\nLịch sử A                     : {series_A.shape}")
        print(f"Tất cả candidate A'           : {candidates_arr.shape}")
        print(f"Mean của các candidate         : {candidates_mean.shape}")

        # ----------------------------------------------------
        # 3. TRỰC QUAN HÓA DẠNG SPAGHETTI PLOT + GROUND TRUTH
        # ----------------------------------------------------
        self.visualize_A_candidates(series_A, candidates_arr, candidates_mean, temperatures, gt_A)

    def visualize_A_candidates(
        self,
        A: np.ndarray,
        candidates: np.ndarray,
        candidates_mean: np.ndarray,
        temperatures: np.ndarray,
        gt_A: np.ndarray,
    ):
        """
        Spaghetti plot: lịch sử A + từng candidate A' (mỗi đường ứng với 1 temperature
        ngẫu nhiên riêng, tô màu theo giá trị temperature) + đường mean của tất cả
        candidate + ground truth thực tế (gt_A) để đối chiếu độ bao phủ của candidate.
        """
        lat_idx, lon_idx = self.config.inspect_lat, self.config.inspect_lon

        point_A = A[:, lat_idx, lon_idx]
        n_candidates = candidates.shape[0]
        points_candidates = candidates[:, :, lat_idx, lon_idx]  # (n_candidates, prediction_steps)
        point_mean = candidates_mean[:, lat_idx, lon_idx]
        point_gt = gt_A[:, lat_idx, lon_idx]

        time_hist = np.arange(len(point_A))
        time_pred = np.arange(len(point_A), len(point_A) + points_candidates.shape[1])

        plt.figure(figsize=(14, 7))

        # Lịch sử A
        plt.plot(time_hist, point_A, color="black", linewidth=2.5, label="Chuỗi A (Lịch sử)", zorder=5)

        # Ground truth thực tế (đối chiếu)
        plt.plot(
            [time_hist[-1], time_pred[0]],
            [point_A[-1], point_gt[0]],
            color="green", linewidth=2.2, zorder=6,
        )
        plt.plot(
            time_pred, point_gt,
            color="green", linewidth=2.2, label="Ground truth (thực tế)", zorder=6,
        )

        # Colormap theo temperature để dễ quan sát ảnh hưởng của temperature
        norm = plt.Normalize(vmin=temperatures.min(), vmax=temperatures.max())
        cmap = cm.get_cmap("plasma")

        for i in range(n_candidates):
            color = cmap(norm(temperatures[i]))

            # Đoạn nối lịch sử -> candidate
            plt.plot(
                [time_hist[-1], time_pred[0]],
                [point_A[-1], points_candidates[i, 0]],
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
            [point_A[-1], point_mean[0]],
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
            f"Spaghetti plot (Chuỗi A - đối chứng): {n_candidates} candidate "
            f"(temperature ngẫu nhiên trong [{self.config.temp_min}, {self.config.temp_max}])\n"
            f"Điểm lưới ({lat_idx}, {lon_idx})"
        )
        plt.xlabel("Bước thời gian")
        plt.ylabel("Nhiệt độ (K)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left", fontsize=8, ncol=2)
        plt.tight_layout()

        file_name = "A_candidates_spaghetti_vs_groundtruth.png"
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
    evaluator.run_A_candidates_experiment(start_idx=0)