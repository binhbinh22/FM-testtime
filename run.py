import os
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Dict

from chronos import ChronosPipeline

# Nếu class này nằm trong file riêng thì uncomment:
from era5_physics_energy import ERA5PhysicsEnergyScorer


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ERA5Config:

    data_path: str = "/home/user18/binhnkt/era5_test_2017_2018.nc"

    # 48 observations × 6 hours = 12 days
    history_steps: int = 48

    # 24 observations × 6 hours = 6 days
    prediction_steps: int = 24

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

            "history_t2m":
                window[
                    self.config.target_var
                ].values[
                    :self.config.history_steps
                ],

            "gt_t2m":
                window[
                    self.config.target_var
                ].values[
                    self.config.history_steps:
                ],

            "u10":
                window[
                    self.config.u_wind_var
                ].values[
                    self.config.history_steps:
                ],

            "v10":
                window[
                    self.config.v_wind_var
                ].values[
                    self.config.history_steps:
                ],

            "times":
                window.time.values
        }


# ============================================================
# CHRONOS PROPOSER (WITH NORMALIZE / DENORMALIZE)
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

        print(
            f"Initializing ChronosProposer "
            f"on {self.device}..."
        )

        # Better dtype handling
        dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.pipeline = ChronosPipeline.from_pretrained(

            model_id,

            device_map=self.device,

            dtype=dtype,
        )

        print(
            "Chronos loaded successfully."
        )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(
        self,
        history: np.ndarray,
        prediction_steps: int,
        temperature: float = 1.0,
        n_samples: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Input:
            history:
                (history_steps, lat, lon)

        Output:
            forecast:
                (n_samples, prediction_steps, lat, lon)
            mean:
                (n_series, 1)
            std:
                (n_series, 1)
        """

        history_steps, lat, lon = history.shape

        n_series = lat * lon

        # ----------------------------------------------------
        # Flatten spatial dimensions
        # [T, H, W] -> [H*W, T]
        # ----------------------------------------------------

        context_2d = (
            history
            .reshape(
                history_steps,
                n_series
            )
            .T
            .astype(np.float32)
        )

        # ====================================================
        # NORMALIZE (Z-score per time series)
        # ====================================================
        mean = np.mean(context_2d, axis=1, keepdims=True)
        std = np.std(context_2d, axis=1, keepdims=True)

        # Tránh chia cho 0 nếu chuỗi phẳng tuyệt đối
        std = np.where(std < 1e-5, 1.0, std)

        normalized_context = (context_2d - mean) / std

        # ----------------------------------------------------
        # Avoid exactly zero temperature
        # ----------------------------------------------------

        eff_temp = max(
            float(temperature),
            1e-4
        )

        all_forecasts = []

        # ----------------------------------------------------
        # Process grid points in batches
        # ----------------------------------------------------

        for start in range(
            0,
            n_series,
            self.batch_size
        ):

            end = min(
                start + self.batch_size,
                n_series
            )

            batch_context = torch.from_numpy(
                normalized_context[start:end]
            )

            with torch.inference_mode():

                forecast_tensor = (
                    self.pipeline.predict(

                        batch_context,

                        prediction_length=
                            prediction_steps,

                        num_samples=
                            n_samples,

                        # Chronos default-style sampling
                        top_k=50,

                        top_p=1.0,

                        temperature=eff_temp,

                        limit_prediction_length=False,
                    )
                )

            # ------------------------------------------------
            # Shape: [batch_series, n_samples, prediction_steps]
            # ------------------------------------------------

            forecast_batch = (
                forecast_tensor
                .cpu()
                .numpy()
            )

            all_forecasts.append(
                forecast_batch
            )

            del batch_context
            del forecast_tensor

            if self.device == "cuda":
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Combine batches
        # [lat*lon, n_samples, prediction_steps]
        # ----------------------------------------------------

        forecast_norm = np.concatenate(
            all_forecasts,
            axis=0
        )

        # ====================================================
        # DENORMALIZE FOR FINAL METRICS/PLOTS
        # ====================================================
        mean_expanded = mean[:, np.newaxis, :]  # [n_series, 1, 1]
        std_expanded = std[:, np.newaxis, :]    # [n_series, 1, 1]

        forecast_denorm = forecast_norm * std_expanded + mean_expanded

        # Reshape back to [n_samples, prediction_steps, lat, lon]
        def reshape_to_spatial(arr):
            arr = arr.transpose(1, 2, 0)
            return arr.reshape(n_samples, prediction_steps, lat, lon)

        forecast_norm_spatial = reshape_to_spatial(forecast_norm)
        forecast_denorm_spatial = reshape_to_spatial(forecast_denorm)

        return forecast_denorm_spatial, forecast_norm_spatial, mean, std


# ============================================================
# VISUALIZATION
# ============================================================

def plot_full_candidates(
    history,
    candidates,
    ground_truth,
    lat_idx,
    lon_idx,
    history_steps,
    prediction_steps,
    temperature,
):
    """
    Plot ALL Chronos candidates before Physics Verifier.
    """

    history_point = (
        history[:, lat_idx, lon_idx]
    )

    gt_point = (
        ground_truth[:, lat_idx, lon_idx]
    )

    n_candidates = candidates.shape[0]

    candidate_points = np.array([
        candidates[
            i,
            :,
            lat_idx,
            lon_idx
        ]
        for i in range(n_candidates)
    ])

    x_history = np.arange(
        history_steps
    )

    x_future = np.arange(
        history_steps,
        history_steps + prediction_steps
    )

    plt.figure(
        figsize=(15, 7)
    )

    plt.plot(
        x_history,
        history_point,
        color="black",
        linewidth=2.5,
        label="History"
    )

    plt.plot(
        x_future,
        gt_point,
        color="green",
        linestyle="--",
        linewidth=2.5,
        label="Ground Truth"
    )

    for i in range(n_candidates):

        plt.plot(
            x_future,
            candidate_points[i],
            linewidth=1.5,
            alpha=0.65,
            label=f"Candidate {i}"
        )

    plt.axvline(
        history_steps,
        color="gray",
        linestyle=":",
        linewidth=2,
        label="Forecast Horizon"
    )

    plt.title(
        "Raw Chronos Candidates "
        f"Before Physics Verifier\n"
        f"Grid Point ({lat_idx}, {lon_idx}) | "
        f"Temperature={temperature} | "
        f"N={n_candidates}"
    )

    plt.xlabel(
        "Time step (6-hour interval)"
    )

    plt.ylabel(
        "Temperature (K)"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        "chronos_full_candidates.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "\nSaved:"
        " chronos_full_candidates.png"
    )

    print("\n")
    print("=" * 70)
    print("RAW CHRONOS CANDIDATE STATISTICS")
    print("=" * 70)

    for i in range(n_candidates):

        candidate = candidate_points[i]

        print(
            f"Candidate {i:02d} | "
            f"Mean={np.mean(candidate):.4f} | "
            f"Std={np.std(candidate):.4f} | "
            f"Min={np.min(candidate):.4f} | "
            f"Max={np.max(candidate):.4f} | "
            f"Range={np.ptp(candidate):.4f} | "
            f"DiffStd={np.std(np.diff(candidate)):.4f}"
        )

    print("=" * 70)


# ============================================================
# MAIN EVALUATOR
# ============================================================

class PhysicsInformedEvaluator:

    def __init__(
        self,
        config: ERA5Config
    ):

        self.config = config

        self.loader = ERA5DataLoader(
            config
        )

        self.forecaster = ChronosProposer(
            model_id=config.model_id
        )

        self.scorer = ERA5PhysicsEnergyScorer(
            dt=config.interval_hours * 3600
        )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    def calculate_rmse(
        self,
        pred: np.ndarray,
        gt: np.ndarray
    ) -> float:

        return float(
            np.sqrt(
                np.mean(
                    (pred - gt) ** 2
                )
            )
        )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run_evaluation(
        self,
        n_test_samples: int = 1
    ):

        print(
            "\nStarting Chronos evaluation..."
        )

        greedy_rmses = []

        best_n_rmses = []

        max_start = (
            len(self.loader.ds.time)
            -
            (
                self.config.history_steps
                +
                self.config.prediction_steps
            )
        )

        start_indices = np.linspace(
            0,
            max_start - 1,
            n_test_samples,
            dtype=int
        )

        last_data = None

        for idx in start_indices:

            data = (
                self.loader
                .get_test_window(idx)
            )

            gt_t2m = data["gt_t2m"]
            u10 = data["u10"]
            v10 = data["v10"]

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"Inference window:"
                f" {data['times'][0]}"
            )

            print(
                "=" * 70
            )

            # =================================================
            # 1. LOW-TEMPERATURE / GREEDY-LIKE BASELINE
            # =================================================

            greedy_pred_denorm, greedy_pred_norm, mean, std = (
                self.forecaster.predict(

                    data["history_t2m"],

                    prediction_steps=
                        self.config.prediction_steps,

                    temperature=0.0,

                    n_samples=1
                )
            )

            greedy_pred_norm_single = greedy_pred_norm[0]

            # Chuẩn hóa Ground Truth để tính RMSE trên miền Norm
            # gt_t2m shape: [pred_steps, lat, lon]
            # mean, std shape: [lat*lon, 1] -> reshape lại thành [lat, lon]
            lat, lon = self.config.grid_size
            mean_spatial = mean.reshape(lat, lon)
            std_spatial = std.reshape(lat, lon)

            # Broadcast theo chiều thời gian
            gt_t2m_norm = (gt_t2m - mean_spatial[np.newaxis, :, :]) / std_spatial[np.newaxis, :, :]

            greedy_rmse = (
                self.calculate_rmse(
                    greedy_pred_norm_single,
                    gt_t2m_norm
                )
            )

            greedy_rmses.append(
                greedy_rmse
            )

            # =================================================
            # 2. GENERATE FULL CHRONOS CANDIDATES (NORM & DENORM)
            # =================================================

            print(
                "\nGenerating "
                f"{self.config.n_scenarios} "
                "Chronos candidates..."
            )

            candidates_denorm, candidates_norm, _, _ = (
                self.forecaster.predict(

                    data["history_t2m"],

                    prediction_steps=
                        self.config.prediction_steps,

                    temperature=1.0,

                    n_samples=
                        self.config.n_scenarios
                )
            )

            print(
                "Candidate tensor shape:",
                candidates_denorm.shape
            )

            # =================================================
            # 3. PLOT FULL RAW CANDIDATES (Dùng bản Denorm để plot trực quan theo nhiệt độ K)
            # =================================================

            plot_full_candidates(

                history=
                    data["history_t2m"],

                candidates=
                    candidates_denorm,

                ground_truth=
                    gt_t2m,

                lat_idx=
                    self.config.inspect_lat,

                lon_idx=
                    self.config.inspect_lon,

                history_steps=
                    self.config.history_steps,

                prediction_steps=
                    self.config.prediction_steps,

                temperature=1.0
            )

            # =================================================
            # 4. PHYSICS VERIFIER (Dùng Candidates Denorm để tính năng lượng vật lý thực tế)
            # =================================================

            print(
                "\nRunning Physics Verifier..."
            )

            top_idx, top_scenarios, details = (
                self.scorer.select_top_k_composite(

                    candidates_denorm,

                    u10,

                    v10,

                    history_t2m=
                        data["history_t2m"],

                    k=1,

                    lambda_activity=1.0
                )
            )

            best_idx = top_idx[0]

            best_n_pred_denorm = (
                top_scenarios[0]
            )

            # Lấy bản candidate tương ứng ở miền Norm để tính RMSE norm
            best_n_pred_norm = candidates_norm[best_idx]

            # =================================================
            # 5. BEST-OF-N RMSE (Trên miền Norm)
            # =================================================

            best_n_rmse = (
                self.calculate_rmse(
                    best_n_pred_norm,
                    gt_t2m_norm
                )
            )

            best_n_rmses.append(
                best_n_rmse
            )

            # =================================================
            # 6. PRINT RESULT
            # =================================================

            print(
                "\n"
                + "-" * 70
            )

            print(
                "RESULT (ON NORMALIZED DOMAIN)"
            )

            print(
                "-" * 70
            )

            print(
                f"Greedy-like RMSE (Norm) : "
                f"{greedy_rmse:.4f}"
            )

            print(
                f"Best-of-N RMSE (Norm)   : "
                f"{best_n_rmse:.4f}"
            )

            print(
                f"Selected candidate: "
                f"{best_idx}"
            )

            print(
                "-" * 70
            )

            last_data = (
                data,
                greedy_pred_denorm[0],
                candidates_denorm,
                best_n_pred_denorm,
                best_idx
            )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"CHRONOS EVALUATION "
            f"OVER {n_test_samples} WINDOWS (NORM)"
        )

        print(
            "=" * 70
        )

        print(
            f"Mean Greedy RMSE (Norm): "
            f"{np.mean(greedy_rmses):.4f}"
        )

        print(
            f"Mean Best-of-N RMSE (Norm): "
            f"{np.mean(best_n_rmses):.4f}"
        )

        print(
            "=" * 70
        )

        if last_data is not None:

            self.visualize_results(
                *last_data
            )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    def visualize_results(
        self,
        data,
        greedy_pred,
        candidates,
        best_n_pred,
        best_idx
    ):

        lat_idx = (
            self.config.inspect_lat
        )

        lon_idx = (
            self.config.inspect_lon
        )

        history = (
            data["history_t2m"]
            [:, lat_idx, lon_idx]
        )

        gt = (
            data["gt_t2m"]
            [:, lat_idx, lon_idx]
        )

        greedy = (
            greedy_pred
            [:, lat_idx, lon_idx]
        )

        best_n = (
            best_n_pred
            [:, lat_idx, lon_idx]
        )

        x_hist = np.arange(
            self.config.history_steps
        )

        x_pred = np.arange(
            self.config.history_steps,

            self.config.history_steps
            +
            self.config.prediction_steps
        )

        plt.figure(
            figsize=(15, 7)
        )

        plt.plot(
            x_hist,
            history,
            color="black",
            linewidth=2.5,
            label="History"
        )

        plt.plot(
            x_pred,
            gt,
            color="green",
            linestyle="--",
            linewidth=2.5,
            label="Ground Truth"
        )

        for i in range(
            candidates.shape[0]
        ):

            candidate = (
                candidates[
                    i,
                    :,
                    lat_idx,
                    lon_idx
                ]
            )

            if i == best_idx:

                plt.plot(
                    x_pred,
                    candidate,
                    color="blue",
                    linewidth=3,
                    label=
                        f"Selected Candidate "
                        f"{i}"
                )

            else:

                plt.plot(
                    x_pred,
                    candidate,
                    alpha=0.35,
                    linewidth=1
                )

        plt.axvline(
            self.config.history_steps,
            color="gray",
            linestyle=":",
            linewidth=2
        )

        plt.title(
            "Chronos Candidates + "
            "Selected Candidate\n"
            f"Grid Point ({lat_idx}, {lon_idx})"
        )

        plt.xlabel(
            "Time step (6-hour interval)"
        )

        plt.ylabel(
            "Temperature (K)"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            "chronos_candidates_selected.png",
            dpi=200
        )

        plt.show()

        plt.figure(
            figsize=(14, 6)
        )

        plt.plot(
            x_hist,
            history,
            color="black",
            linewidth=2.5,
            label="History"
        )

        plt.plot(
            x_pred,
            gt,
            color="green",
            linestyle="--",
            linewidth=2.5,
            label="Ground Truth"
        )

        plt.plot(
            x_pred,
            greedy,
            color="red",
            linewidth=2,
            label="Chronos Low-Temperature"
        )

        plt.plot(
            x_pred,
            best_n,
            color="blue",
            linewidth=3,
            label=
                f"Best-of-{candidates.shape[0]} "
                f"(Candidate {best_idx})"
        )

        plt.axvline(
            self.config.history_steps,
            color="gray",
            linestyle=":",
            linewidth=2,
            label="Forecast Horizon"
        )

        plt.title(
            "Chronos Final Comparison\n"
            f"Grid Point ({lat_idx}, {lon_idx})"
        )

        plt.xlabel(
            "Time step (6-hour interval)"
        )

        plt.ylabel(
            "Temperature (K)"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            "evaluation_result.png",
            dpi=200
        )

        plt.show()

        print(
            "\nSaved:"
        )

        print(
            "  chronos_full_candidates.png"
        )

        print(
            "  chronos_candidates_selected.png"
        )

        print(
            "  evaluation_result.png"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    config = ERA5Config()

    np.random.seed(
        config.seed
    )

    torch.manual_seed(
        config.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            config.seed
        )

    evaluator = (
        PhysicsInformedEvaluator(
            config
        )
    )

    evaluator.run_evaluation(
        n_test_samples=10
    )