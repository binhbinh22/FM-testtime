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
import matplotlib.cm as cm

from dataclasses import dataclass
from typing import Tuple, Dict

from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

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

    grid_size: Tuple[int, int] = (32, 64)

    target_var: str = "2m_temperature"

    u_wind_var: str = "10m_u_component_of_wind"
    v_wind_var: str = "10m_v_component_of_wind"

    # Model Moirai 2.0 (uni2ts.model.moirai2)
    model_id: str = "Salesforce/moirai-2.0-R-small"

    seed: int = 42

    # Grid point that we want to inspect
    inspect_lat: int = 16
    inspect_lon: int = 32

    # ----------------------------------------------------
    # Số candidate sinh ra cho spaghetti plot.
    # LƯU Ý: khác với Chronos, Moirai KHÔNG có tham số temperature nên không
    # cần dải [temp_min, temp_max] — toàn bộ N candidate được lấy mẫu trực
    # tiếp từ phân phối xác suất (mixture distribution) mà model dự đoán,
    # chỉ trong 1 lần gọi predict() (num_samples=n_candidates).
    # ----------------------------------------------------
    n_candidates: int = 10


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
    Proposer dùng Moirai 2.0 (uni2ts) thay cho Chronos.

    Giữ nguyên interface .predict(history, prediction_steps, temperature, n_samples)
    -> (forecast_denorm_spatial, forecast_norm_spatial, mean, std) để cắm thẳng
    vào PhysicsInformedEvaluator, KHÔNG cần sửa phần còn lại của pipeline.

    LƯU Ý KIẾN TRÚC (khác Moirai 1.x):
    - Moirai 2.0 dùng class Moirai2Forecast/Moirai2Module (module uni2ts.model.moirai2),
      KHÔNG phải MoiraiForecast/MoiraiModule của bản 1.x.
    - Theo ví dụ chính thức của Salesforce, Moirai2Forecast KHÔNG nhận tham số
      `patch_size` trong constructor (kiến trúc decoder-only tự xử lý patching nội bộ).
    - Moirai 2.0 chuyển từ distributional loss (mixture distribution) sang quantile
      loss + multi-token prediction. Điều này có thể ảnh hưởng đến cách sinh nhiều
      sample: chưa có xác nhận chắc chắn liệu `num_samples` có được hỗ trợ y hệt bản
      1.x hay không. Code dưới đây có cơ chế fallback: nếu truyền num_samples gây lỗi,
      sẽ tự động thử lại KHÔNG truyền num_samples và log cảnh báo — bạn nên kiểm tra kỹ
      lại forecast trả về (forecasts[i].samples) có thực sự đa dạng giữa các sample hay
      không, vì bản chất quantile loss có thể khiến các "sample" ít ngẫu nhiên hơn.
    - Không autoregressive token-based như Chronos, nên KHÔNG có khái niệm decoding
      `temperature`. Tham số này được giữ trong chữ ký hàm chỉ để tương thích interface.
    """

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-2.0-R-small",
        device: str = "cuda",
        predictor_batch_size: int = 512,
    ):

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model_id = model_id
        self.predictor_batch_size = predictor_batch_size

        print(f"Initializing MoiraiProposer (Moirai 2.0: {model_id}) on {self.device}...")

        self.module = Moirai2Module.from_pretrained(model_id)

        print("Moirai 2.0 module loaded successfully.")

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

        common_kwargs = dict(
            module=self.module,
            prediction_length=prediction_steps,
            context_length=history_steps,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        try:
            model = Moirai2Forecast(num_samples=n_samples_eff, **common_kwargs)
        except TypeError:
            print(
                "[MoiraiProposer] Moirai2Forecast không nhận tham số num_samples ở "
                "phiên bản uni2ts này -> tạo model không truyền num_samples. Kiểm tra "
                "lại forecasts[i].samples sau khi chạy để xác nhận số lượng/độ đa dạng "
                "sample thực tế."
            )
            model = Moirai2Forecast(**common_kwargs)

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
        )

    def run_A_candidates_experiment(self, start_idx: int = 0):
        """
        Thí nghiệm đối chứng: sinh N candidate cho chuỗi A (KHÔNG downsample)
        bằng Moirai, visualize dạng spaghetti plot cùng ground truth thực tế
        (gt_t2m) để đối chiếu độ bao phủ / độ bất định của candidate.

        Khác với bản Chronos: vì Moirai không có temperature, N candidate được
        sinh ra CHỈ TRONG 1 LẦN GỌI predict() (num_samples=n_candidates) thay
        vì phải lặp N lần với N temperature khác nhau.
        """
        print("\n" + "=" * 70)
        print(f"SINH {self.config.n_candidates} CANDIDATE CHO CHUỖI A (MOIRAI)")
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
        # 2. SINH N CANDIDATE TRONG 1 LẦN GỌI PREDICT
        # ----------------------------------------------------
        print(f"\nTiến hành dự báo {self.config.n_candidates} candidate (Từ A, Moirai)...")
        pred_A_denorm, _, _, _ = self.forecaster.predict(
            history=series_A,
            prediction_steps=self.config.prediction_steps,
            n_samples=self.config.n_candidates,
        )
        candidates_arr = pred_A_denorm  # (n_candidates, prediction_steps, lat, lon)
        candidates_mean = candidates_arr.mean(axis=0)  # (prediction_steps, lat, lon)

        print(f"\nLịch sử A                     : {series_A.shape}")
        print(f"Tất cả candidate A'           : {candidates_arr.shape}")
        print(f"Mean của các candidate         : {candidates_mean.shape}")

        # ----------------------------------------------------
        # 3. TRỰC QUAN HÓA DẠNG SPAGHETTI PLOT + GROUND TRUTH
        # ----------------------------------------------------
        self.visualize_A_candidates(series_A, candidates_arr, candidates_mean, gt_A)

    def visualize_A_candidates(
        self,
        A: np.ndarray,
        candidates: np.ndarray,
        candidates_mean: np.ndarray,
        gt_A: np.ndarray,
    ):
        """
        Spaghetti plot: lịch sử A + từng candidate A' (mỗi đường là 1 sample
        độc lập lấy từ phân phối xác suất của Moirai, tô màu theo chỉ số
        candidate vì không còn temperature để tô theo) + đường mean của tất cả
        candidate + ground truth thực tế (gt_A) để đối chiếu độ bao phủ.
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

        # Colormap theo CHỈ SỐ candidate (không còn temperature để tô theo như Chronos)
        norm = plt.Normalize(vmin=0, vmax=max(n_candidates - 1, 1))
        cmap = cm.get_cmap("plasma")

        for i in range(n_candidates):
            color = cmap(norm(i))

            # Đoạn nối lịch sử -> candidate
            plt.plot(
                [time_hist[-1], time_pred[0]],
                [point_A[-1], points_candidates[i, 0]],
                color=color, linewidth=1.2, alpha=0.6, zorder=2,
            )

            plt.plot(
                time_pred, points_candidates[i],
                color=color, linewidth=1.4, alpha=0.75, zorder=2,
                label=f"Candidate {i + 1}" if n_candidates <= 12 else None,
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

        # Colorbar theo chỉ số candidate
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label("Chỉ số candidate")

        plt.axvline(time_hist[-1], color="gray", linestyle=":", linewidth=2, label="Forecast Horizon")
        plt.title(
            f"[Moirai] Spaghetti plot (Chuỗi A): {n_candidates} candidate "
            f"(sinh trong 1 lần gọi predict, không có temperature)\n"
            f"Điểm lưới ({lat_idx}, {lon_idx})"
        )
        plt.xlabel("Bước thời gian")
        plt.ylabel("Nhiệt độ (K)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left", fontsize=8, ncol=2)
        plt.tight_layout()

        file_name = "moirai_A_candidates_spaghetti_vs_groundtruth.png"
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