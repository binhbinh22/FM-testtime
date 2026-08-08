import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MultiscaleSearchConfig:
    """Configuration for the 3-scale test-time search pipeline."""

    scales: int = 3
    pool_size: int = 2
    n_scenarios_coarse: int = 8
    n_scenarios_intermediate: int = 8
    n_scenarios_fine: int = 8
    k_coarse: int = 3
    k_intermediate: int = 3
    k_fine: int = 1
    dtw_threshold: float = 1.5
    lambda_activity: float = 1.0
    temperature_coarse: float = 1.0
    temperature_intermediate: float = 1.0
    temperature_fine: float = 1.0
    adaptive_sampling: bool = True
    adaptive_multiplier: int = 2


class MultiscaleTimeMixerDownsampler:
    """
    TimeMixer-style downsampling over the time axis.

    The input is pooled along the time dimension using average pooling with
    a fixed window size. This produces a hierarchy of sequences:

    - Scale 0: finest raw sequence (A)
    - Scale 1: one downsampling step (B)
    - Scale 2: two downsampling steps (C)
    """

    def __init__(self, pool_size: int = 2, scales: int = 3):
        self.pool_size = pool_size
        self.scales = scales

    def downsample_along_axis(self, x: np.ndarray, factor: int, axis: int = 0) -> np.ndarray:
        """Average-pool a tensor along a given axis using a non-overlapping window."""
        arr = np.asarray(x)
        if factor <= 0:
            raise ValueError("factor must be positive")
        if arr.shape[axis] % factor != 0:
            raise ValueError(
                f"Axis length {arr.shape[axis]} is not divisible by factor {factor}."
            )

        if arr.ndim == 1:
            reshaped = arr.reshape(arr.shape[0] // factor, factor)
            return reshaped.mean(axis=1)

        shape = list(arr.shape)
        new_shape = shape[:axis] + [shape[axis] // factor, factor] + shape[axis + 1 :]
        reshaped = arr.reshape(new_shape)
        return reshaped.mean(axis=axis + 1)

    def build_multiscale_views(self, history: np.ndarray, forecast: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
        """
        Build scale views for history and optionally for a forecast batch.

        For a history of shape (T, H, W), the returned views have lengths
        T, T/2, T/4 for scales 0, 1, 2 when divisible by 4.
        """
        if history.ndim < 2:
            raise ValueError("history must be at least 2D")

        history_views = {0: np.asarray(history)}
        current = np.asarray(history)
        for scale in range(1, self.scales):
            current = self.downsample_along_axis(current, self.pool_size, axis=0)
            history_views[scale] = current

        if forecast is None:
            return history_views

        forecast_views = {0: np.asarray(forecast)}
        current_forecast = np.asarray(forecast)
        for scale in range(1, self.scales):
            current_forecast = self.downsample_along_axis(current_forecast, self.pool_size, axis=1)
            forecast_views[scale] = current_forecast

        return forecast_views


class MultiscaleTimeSearchEngine:
    """
    Three-stage multiscale search pipeline inspired by TimeMixer-style decomposition.

    Stage 1 (Coarse): search on scale C, select top-k anchors.
    Stage 2 (Intermediate): search on scale B with DTW consistency against coarse anchors.
    Stage 3 (Fine): search on scale A with DTW consistency against intermediate anchors,
               then select the single best trajectory.
    """

    def __init__(
        self,
        proposer,
        scorer,
        config: Optional[MultiscaleSearchConfig] = None,
    ):
        self.proposer = proposer
        self.scorer = scorer
        self.config = config or MultiscaleSearchConfig()
        self.downsampler = MultiscaleTimeMixerDownsampler(
            pool_size=self.config.pool_size,
            scales=self.config.scales,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prediction_length_for_scale(self, prediction_steps: int, scale: int) -> int:
        return max(1, int(prediction_steps // (2 ** scale)))

    def _build_scale_views(self, history: np.ndarray, future_u: np.ndarray, future_v: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        scale_views: Dict[int, Dict[str, np.ndarray]] = {}
        current_hist = np.asarray(history)
        current_u = np.asarray(future_u)
        current_v = np.asarray(future_v)

        for scale in range(self.config.scales):
            scale_views[scale] = {
                "history": current_hist,
                "u_wind": current_u,
                "v_wind": current_v,
            }
            if scale < self.config.scales - 1:
                current_hist = self.downsampler.downsample_along_axis(current_hist, self.config.pool_size, axis=0)
                current_u = self.downsampler.downsample_along_axis(current_u, self.config.pool_size, axis=0)
                current_v = self.downsampler.downsample_along_axis(current_v, self.config.pool_size, axis=0)

        return scale_views

    def _reshape_single_trajectory(self, traj: np.ndarray) -> np.ndarray:
        arr = np.asarray(traj)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return arr.reshape(arr.shape[0], -1)
        if arr.ndim == 4:
            return arr.reshape(arr.shape[0], -1)
        raise ValueError(f"Unsupported trajectory shape: {arr.shape}")

    def _dtw_distance(self, traj_a: np.ndarray, traj_b: np.ndarray) -> float:
        """Compute DTW distance between two trajectories after flattening the spatial grid."""
        a = self._reshape_single_trajectory(traj_a)
        b = self._reshape_single_trajectory(traj_b)

        if a.shape[0] != b.shape[0]:
            raise ValueError(f"Trajectory lengths differ: {a.shape[0]} vs {b.shape[0]}")

        n = a.shape[0]
        m = b.shape[0]

        cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
        dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
        dtw[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                local_cost = cost[i - 1, j - 1]
                dtw[i, j] = local_cost + min(
                    dtw[i - 1, j],
                    dtw[i, j - 1],
                    dtw[i - 1, j - 1],
                )

        return float(dtw[n, m])

    def _downsample_batch_to_target_scale(self, trajectories: np.ndarray, factor: int) -> np.ndarray:
        """Downsample a batch of trajectories along the temporal axis."""
        if trajectories.ndim == 3:
            return self.downsampler.downsample_along_axis(trajectories, factor, axis=0)
        if trajectories.ndim == 4:
            return self.downsampler.downsample_along_axis(trajectories, factor, axis=1)
        raise ValueError(f"Unsupported trajectory batch shape: {trajectories.shape}")

    def _filter_by_dtw(self, candidates: np.ndarray, anchors: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keep only candidates whose best DTW distance to the anchor set is below threshold."""
        if candidates.shape[0] == 0:
            return (
                np.empty((0, *candidates.shape[1:]), dtype=candidates.dtype),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        valid_indices: List[int] = []
        distances: List[float] = []

        for idx in range(candidates.shape[0]):
            candidate = candidates[idx]
            best_distance = min(self._dtw_distance(candidate, anchor) for anchor in anchors)
            if best_distance <= threshold:
                valid_indices.append(idx)
                distances.append(best_distance)

        if len(valid_indices) == 0:
            return (
                np.empty((0, *candidates.shape[1:]), dtype=candidates.dtype),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        kept = candidates[np.array(valid_indices)]
        kept_distances = np.array(distances, dtype=np.float32)
        kept_idx = np.array(valid_indices, dtype=np.int64)
        return kept, kept_distances, kept_idx

    def _adaptive_sampling(self, base_n: int, attempted_n: int) -> int:
        if not self.config.adaptive_sampling:
            return base_n
        return max(base_n, attempted_n * self.config.adaptive_multiplier)

    def _search_stage(
        self,
        history: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        prediction_steps: int,
        temperature: float,
        n_scenarios: int,
        k_keep: int,
        scale_name: str,
    ) -> Dict[str, np.ndarray]:
        """Run one search stage using the existing Chronos proposer + physics scorer."""
        n_attempt = n_scenarios
        candidates_denorm = None
        candidates_norm = None
        mean = None
        std = None

        while True:
            candidates_denorm, candidates_norm, mean, std = self.proposer.predict(
                history,
                prediction_steps=prediction_steps,
                temperature=temperature,
                n_samples=n_attempt,
            )

            if candidates_denorm.shape[0] >= k_keep or not self.config.adaptive_sampling:
                break
            n_attempt = self._adaptive_sampling(n_scenarios, n_attempt)

        top_idx, top_scenarios, details = self.scorer.select_top_k_composite(
            candidates_denorm,
            u_wind,
            v_wind,
            history_t2m=history,
            k=k_keep,
            lambda_activity=self.config.lambda_activity,
        )

        return {
            "candidates_denorm": candidates_denorm,
            "candidates_norm": candidates_norm,
            "top_idx": top_idx,
            "top_scenarios": top_scenarios,
            "details": details,
            "mean": mean,
            "std": std,
            "scale_label": scale_name,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_multiscale_search(
        self,
        history: np.ndarray,
        u_wind_future: np.ndarray,
        v_wind_future: np.ndarray,
        prediction_steps: int,
        ground_truth: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Main routine for the full 3-stage multiscale search pipeline.

        Parameters
        ----------
        history : ndarray
            Shape (history_steps, lat, lon). The raw history at the finest scale.
        u_wind_future : ndarray
            Shape (prediction_steps, lat, lon) for the future wind field.
        v_wind_future : ndarray
            Shape (prediction_steps, lat, lon) for the future wind field.
        prediction_steps : int
            Forecast horizon on the finest scale.
        ground_truth : ndarray, optional
            Shape (prediction_steps, lat, lon), used to report RMSE metrics.
        """
        if history.ndim != 3:
            raise ValueError("history must be rank-3: (time, lat, lon)")

        scale_views = self._build_scale_views(history, u_wind_future, v_wind_future)

        # -----------------------------
        # Stage 1: coarse-scale search
        # -----------------------------
        coarse_scale = 2
        coarse_view = scale_views[coarse_scale]
        pred_len_c = self._prediction_length_for_scale(prediction_steps, coarse_scale)
        coarse_stage = self._search_stage(
            history=coarse_view["history"],
            u_wind=coarse_view["u_wind"],
            v_wind=coarse_view["v_wind"],
            prediction_steps=pred_len_c,
            temperature=self.config.temperature_coarse,
            n_scenarios=self.config.n_scenarios_coarse,
            k_keep=self.config.k_coarse,
            scale_name="C",
        )
        coarse_anchors = coarse_stage["top_scenarios"]

        # -----------------------------
        # Stage 2: intermediate-scale search with DTW consistency
        # -----------------------------
        intermediate_scale = 1
        intermediate_view = scale_views[intermediate_scale]
        pred_len_b = self._prediction_length_for_scale(prediction_steps, intermediate_scale)
        intermediate_stage = self._search_stage(
            history=intermediate_view["history"],
            u_wind=intermediate_view["u_wind"],
            v_wind=intermediate_view["v_wind"],
            prediction_steps=pred_len_b,
            temperature=self.config.temperature_intermediate,
            n_scenarios=self.config.n_scenarios_intermediate,
            k_keep=self.config.k_intermediate,
            scale_name="B",
        )

        candidates_b = intermediate_stage["candidates_denorm"]
        if coarse_anchors.shape[0] > 0:
            coarse_resampled = coarse_anchors
            candidates_b_for_dtw = self._downsample_batch_to_target_scale(candidates_b, factor=2)
            filtered_candidates_b, _, filtered_idx_b = self._filter_by_dtw(
                candidates_b_for_dtw,
                coarse_resampled,
                threshold=self.config.dtw_threshold,
            )
            if filtered_candidates_b.shape[0] == 0:
                filtered_candidates_b = candidates_b_for_dtw[: max(1, self.config.k_intermediate)]
                filtered_idx_b = np.arange(min(candidates_b.shape[0], max(1, self.config.k_intermediate)))
        else:
            filtered_candidates_b = self._downsample_batch_to_target_scale(candidates_b, factor=2)[: max(1, self.config.k_intermediate)]
            filtered_idx_b = np.arange(min(candidates_b.shape[0], max(1, self.config.k_intermediate)))

        if filtered_candidates_b.shape[0] > 0:
            # Keep the original-resolution candidates for physics verification.
            keep_indices = filtered_idx_b[: min(len(filtered_idx_b), candidates_b.shape[0])]
            if len(keep_indices) == 0:
                keep_indices = np.array([0])
            physics_candidates_b = candidates_b[keep_indices]
            physics_filtered_b = self.scorer.select_top_k_composite(
                physics_candidates_b,
                intermediate_view["u_wind"],
                intermediate_view["v_wind"],
                history_t2m=intermediate_view["history"],
                k=min(self.config.k_intermediate, physics_candidates_b.shape[0]),
                lambda_activity=self.config.lambda_activity,
            )[1]
            intermediate_anchors = physics_filtered_b
        else:
            intermediate_anchors = candidates_b[:1]

        # -----------------------------
        # Stage 3: fine-scale search with DTW consistency
        # -----------------------------
        fine_scale = 0
        fine_view = scale_views[fine_scale]
        pred_len_a = self._prediction_length_for_scale(prediction_steps, fine_scale)
        fine_stage = self._search_stage(
            history=fine_view["history"],
            u_wind=fine_view["u_wind"],
            v_wind=fine_view["v_wind"],
            prediction_steps=pred_len_a,
            temperature=self.config.temperature_fine,
            n_scenarios=self.config.n_scenarios_fine,
            k_keep=self.config.k_fine,
            scale_name="A",
        )

        candidates_a = fine_stage["candidates_denorm"]
        if intermediate_anchors.shape[0] > 0:
            candidates_a_for_dtw = self._downsample_batch_to_target_scale(candidates_a, factor=2)
            filtered_candidates_a, _, filtered_idx_a = self._filter_by_dtw(
                candidates_a_for_dtw,
                intermediate_anchors,
                threshold=self.config.dtw_threshold,
            )
            if filtered_candidates_a.shape[0] == 0:
                filtered_candidates_a = candidates_a_for_dtw[: max(1, self.config.k_fine)]
                filtered_idx_a = np.arange(min(candidates_a.shape[0], max(1, self.config.k_fine)))
        else:
            filtered_candidates_a = self._downsample_batch_to_target_scale(candidates_a, factor=2)[: max(1, self.config.k_fine)]
            filtered_idx_a = np.arange(min(candidates_a.shape[0], max(1, self.config.k_fine)))

        if filtered_candidates_a.shape[0] > 0:
            keep_indices = filtered_idx_a[: min(len(filtered_idx_a), candidates_a.shape[0])]
            if len(keep_indices) == 0:
                keep_indices = np.array([0])
            physics_candidates_a = candidates_a[keep_indices]
            top_idx_final, top_scenarios_final, _ = self.scorer.select_top_k_composite(
                physics_candidates_a,
                fine_view["u_wind"],
                fine_view["v_wind"],
                history_t2m=fine_view["history"],
                k=min(self.config.k_fine, physics_candidates_a.shape[0]),
                lambda_activity=self.config.lambda_activity,
            )
            final_candidate = top_scenarios_final[0]
        else:
            top_idx_final = np.array([0])
            final_candidate = candidates_a[0]

        # -----------------------------
        # RMSE metrics
        # -----------------------------
        metrics = {}
        if ground_truth is not None:
            metrics = self.compute_rmse_metrics(
                selected_prediction=final_candidate,
                ground_truth=ground_truth,
                mean=fine_stage["mean"],
                std=fine_stage["std"],
                history=history,
            )

        return {
            "coarse_stage": coarse_stage,
            "intermediate_stage": intermediate_stage,
            "fine_stage": fine_stage,
            "coarse_anchors": coarse_anchors,
            "intermediate_anchors": intermediate_anchors,
            "final_candidate": final_candidate,
            "final_index": top_idx_final[0],
            "metrics": metrics,
        }

    def compute_rmse_metrics(
        self,
        selected_prediction: np.ndarray,
        ground_truth: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        history: np.ndarray,
    ) -> Dict[str, float]:
        """Compute RMSE in normalized and denormalized spaces for the selected trajectory."""
        pred = np.asarray(selected_prediction)
        gt = np.asarray(ground_truth)

        if pred.shape != gt.shape:
            raise ValueError(f"Prediction shape {pred.shape} does not match ground truth {gt.shape}")

        lat, lon = history.shape[1], history.shape[2]
        mean_spatial = mean.reshape(lat, lon)
        std_spatial = std.reshape(lat, lon)

        pred_norm = (pred - mean_spatial[np.newaxis, :, :]) / std_spatial[np.newaxis, :, :]
        gt_norm = (gt - mean_spatial[np.newaxis, :, :]) / std_spatial[np.newaxis, :, :]

        rmse_norm = float(np.sqrt(np.mean((pred_norm - gt_norm) ** 2)))
        rmse_denorm = float(np.sqrt(np.mean((pred - gt) ** 2)))

        return {
            "rmse_norm": rmse_norm,
            "rmse_denorm": rmse_denorm,
        }

    def _resample_series(self, values: np.ndarray, target_length: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("values must be 1D")
        if values.size == 0:
            return np.zeros(target_length, dtype=np.float32)
        if values.size == 1:
            return np.full(target_length, values[0], dtype=np.float32)

        x_old = np.arange(values.size, dtype=np.float32)
        x_new = np.linspace(0, values.size - 1, target_length, dtype=np.float32)
        return np.interp(x_new, x_old, values)

    def save_multiscale_visuals(
        self,
        history: np.ndarray,
        ground_truth: np.ndarray,
        result: Dict[str, object],
        lat_idx: int = 0,
        lon_idx: int = 0,
        output_dir: str = ".",
    ) -> Dict[str, str]:
        """Create 3 plots summarizing the multiscale search outcome."""
        history = np.asarray(history)
        ground_truth = np.asarray(ground_truth)
        final_candidate = np.asarray(result["final_candidate"])

        if history.ndim != 3:
            raise ValueError("history must be rank-3")
        if ground_truth.ndim != 3:
            raise ValueError("ground_truth must be rank-3")
        if final_candidate.ndim != 3:
            raise ValueError("final_candidate must be rank-3")

        history_series = history[:, lat_idx, lon_idx]
        gt_series = ground_truth[:, lat_idx, lon_idx]
        selected_series = final_candidate[:, lat_idx, lon_idx]

        history_steps = history_series.shape[0]
        pred_steps = gt_series.shape[0]

        fine_stage = result.get("fine_stage", {})
        candidates = fine_stage.get("candidates_denorm")
        if candidates is not None:
            candidate_series = np.array([
                candidates[i, :, lat_idx, lon_idx] for i in range(candidates.shape[0])
            ])
        else:
            candidate_series = np.array([selected_series])

        x_history = np.arange(history_steps)
        x_future = np.arange(history_steps, history_steps + pred_steps)

        os.makedirs(output_dir, exist_ok=True)

        # Plot 1: all fine-scale candidates
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(x_history, history_series, color="black", linewidth=2.5, label="History")
        ax.plot(x_future, gt_series, color="green", linestyle="--", linewidth=2.5, label="Ground Truth")
        for i in range(candidate_series.shape[0]):
            ax.plot(x_future, candidate_series[i], linewidth=1.2, alpha=0.65, label=f"Candidate {i}" if i == 0 else None)
        ax.plot(x_future, selected_series, color="blue", linewidth=3, label="Selected Multiscale")
        ax.axvline(history_steps, color="gray", linestyle=":", linewidth=2, label="Forecast Horizon")
        ax.set_title("Multiscale Search Candidates")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Temperature")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if l]
        if filtered:
            ax.legend(*zip(*filtered))
        fig.tight_layout()
        path1 = os.path.join(output_dir, "multiscale_candidates.png")
        fig.savefig(path1, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Plot 2: selected trajectory only
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x_history, history_series, color="black", linewidth=2.5, label="History")
        ax.plot(x_future, gt_series, color="green", linestyle="--", linewidth=2.5, label="Ground Truth")
        ax.plot(x_future, selected_series, color="blue", linewidth=3, label="Selected Multiscale")
        ax.axvline(history_steps, color="gray", linestyle=":", linewidth=2)
        ax.set_title("Selected Multiscale Trajectory")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Temperature")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path2 = os.path.join(output_dir, "multiscale_selected.png")
        fig.savefig(path2, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Plot 3: comparison across scales
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(x_history, history_series, color="black", linewidth=2.5, label="History")
        ax.plot(x_future, gt_series, color="green", linestyle="--", linewidth=2.5, label="Ground Truth")
        ax.plot(x_future, selected_series, color="blue", linewidth=3, label="Selected")

        coarse_stage = result.get("coarse_stage", {})
        intermediate_stage = result.get("intermediate_stage", {})
        coarse_scenarios = coarse_stage.get("top_scenarios")
        intermediate_scenarios = intermediate_stage.get("top_scenarios")
        if coarse_scenarios is not None and coarse_scenarios.shape[0] > 0:
            coarse_series = coarse_scenarios[0, :, lat_idx, lon_idx]
            coarse_series_resampled = self._resample_series(coarse_series, pred_steps)
            ax.plot(x_future, coarse_series_resampled, color="orange", linewidth=1.8, alpha=0.8, label="Coarse Anchor")
        if intermediate_scenarios is not None and intermediate_scenarios.shape[0] > 0:
            intermediate_series = intermediate_scenarios[0, :, lat_idx, lon_idx]
            intermediate_series_resampled = self._resample_series(intermediate_series, pred_steps)
            ax.plot(x_future, intermediate_series_resampled, color="purple", linewidth=1.8, alpha=0.8, label="Intermediate Anchor")

        ax.axvline(history_steps, color="gray", linestyle=":", linewidth=2, label="Forecast Horizon")
        ax.set_title("Multiscale Comparison")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Temperature")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path3 = os.path.join(output_dir, "multiscale_comparison.png")
        fig.savefig(path3, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return {
            "multiscale_candidates": path1,
            "multiscale_selected": path2,
            "multiscale_comparison": path3,
        }
