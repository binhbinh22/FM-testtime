import os
import sys

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from multiscale import downsample_avgpool  # noqa: E402
from era5_physics_energy import ERA5PhysicsEnergyScorer  # noqa: E402
from trend_seasonal_decompose import centered_moving_average_decompose, kernel_size_for_scale  # noqa: E402
from dtw import ERA5DTWEnergyScorer  # noqa: E402
from frequency import ERA5FrequencyEnergyScorer  # noqa: E402
from spatial_decompose import (  # noqa: E402
    aggregate_to_clusters,
    spatial_decompose,
    spatial_consistency_score,
    history_reference_Q,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import generate_multivar_candidates, compute_multivar_mse  # noqa: E402


# ============================================================
# BIEN THE DA BIEN cua SpatialClusterSearchEngine goc
# (../../spatial_cluster_search.py) - KHONG sua file goc (dang dung o noi
# khac voi Chronos-2). S (ma tran gan cum) va W (do thi kNN) van build 1
# LAN ben ngoai, truyen vao qua constructor - giong het ban goc.
#
# Khac bản goc: nhanh Fine sinh CA 3 bien, Physics Filter dung gio DA
# DUOC DU BAO tung candidate. Nhanh Coarse van CHI t2m (gop cum), khong
# can du bao gio o cap cum.
# ============================================================


@dataclass
class SpatialClusterMultivarConfig:
    n_clusters: int = 32
    downsample_factor: int = 4
    n_fine: int = 15
    n_coarse: int = 5
    k_fine: int = 5
    kernel_size_fine: int = 16
    w1_dtw: float = 1.0
    w2_fft: float = 1.0
    w3_spatial: float = 1.0
    lambda_activity: float = 1.0
    dtw_window: Optional[float] = None


class SpatialClusterMultivarEngine:
    """
    Cluster+Physics (w3_spatial=0) / Cluster+Spatial+Physics (w3_spatial=1)
    - cung 1 engine, chi khac trong so (xem architectures/cluster_physics.py,
    cluster_spatial_physics.py).
    """

    def __init__(self, forecaster, S: np.ndarray, W: np.ndarray, config: Optional[SpatialClusterMultivarConfig] = None):
        self.forecaster = forecaster
        self.S = S
        self.W = W
        self.config = config or SpatialClusterMultivarConfig()
        self.dtw_scorer = ERA5DTWEnergyScorer(window=self.config.dtw_window, distance="abs")
        self.freq_scorer = ERA5FrequencyEnergyScorer(factor=self.config.downsample_factor, mode="complex_abs")

    def _flatten_grid(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1)

    def _cluster_series_as_grid(self, X_c: np.ndarray) -> np.ndarray:
        return X_c[:, :, None]

    def run(self, data: Dict, prediction_steps: int, interval_hours: float, lat: int, lon: int) -> Dict:
        cfg = self.config
        factor = cfg.downsample_factor
        history_flat = self._flatten_grid(data["history_t2m"])

        # ------------------------------------------------------------
        # Nhanh Coarse: gop cum (CHI t2m) -> downsample thoi gian -> Chronos(N_C)
        # ------------------------------------------------------------
        history_cluster = aggregate_to_clusters(history_flat, self.S)
        history_cluster_ds = downsample_avgpool(history_cluster, factor=factor)
        prediction_steps_coarse = max(1, prediction_steps // factor)
        history_cluster_grid = self._cluster_series_as_grid(history_cluster_ds)

        coarse_denorm, _coarse_norm, _coarse_mean, _coarse_std = self.forecaster.predict(
            history_cluster_grid, prediction_steps=prediction_steps_coarse, temperature=1.0, n_samples=cfg.n_coarse,
        )
        coarse_denorm = coarse_denorm[..., 0]

        kernel_fine = cfg.kernel_size_fine
        kernel_coarse = kernel_size_for_scale(kernel_fine, factor)
        coarse_trend, coarse_seasonal = [], []
        for j in range(coarse_denorm.shape[0]):
            t, s = centered_moving_average_decompose(coarse_denorm[j], kernel_coarse)
            coarse_trend.append(t)
            coarse_seasonal.append(s)

        # ------------------------------------------------------------
        # Nhanh Fine: sinh CA 3 bien, Physics Filter dung gio du bao
        # ------------------------------------------------------------
        candidates = generate_multivar_candidates(
            self.forecaster, data, prediction_steps, n_samples=cfg.n_fine, temperature=1.0,
        )
        t2m_denorm = candidates["t2m"][0]
        u10_denorm = candidates["u10"][0]
        v10_denorm = candidates["v10"][0]

        physics_scorer = ERA5PhysicsEnergyScorer(dt=interval_hours * 3600.0)
        physics_scores = np.empty(cfg.n_fine, dtype=np.float64)
        for i in range(cfg.n_fine):
            details = physics_scorer.composite_score(
                t2m_denorm[i], u10_denorm[i], v10_denorm[i],
                history_t2m=data["history_t2m"], lambda_activity=cfg.lambda_activity,
            )
            physics_scores[i] = details["composite_score"]
        k_eff = min(cfg.k_fine, cfg.n_fine)
        fine_idx = np.argsort(physics_scores)[:k_eff]
        fine_top_denorm = t2m_denorm[fine_idx]

        # ------------------------------------------------------------
        # Spatial Decomp tren moi candidate fine con lai
        # ------------------------------------------------------------
        history_local_flat = history_flat - aggregate_to_clusters(history_flat, self.S) @ self.S.T
        hist_Q = history_reference_Q(history_local_flat, self.W)

        n_fine_kept = fine_top_denorm.shape[0]
        fine_trend_cluster, fine_seasonal_cluster = [], []
        e_spatial = np.zeros(n_fine_kept, dtype=np.float64)

        for i in range(n_fine_kept):
            Y_fine_flat = self._flatten_grid(fine_top_denorm[i])
            Y_global_cluster, _Y_global_broadcast, Y_local = spatial_decompose(Y_fine_flat, self.S)
            trend_c, seasonal_c = centered_moving_average_decompose(Y_global_cluster, kernel_fine)
            fine_trend_cluster.append(trend_c)
            fine_seasonal_cluster.append(seasonal_c)
            e_spatial[i] = spatial_consistency_score(Y_local, self.W, hist_Q)

        # ------------------------------------------------------------
        # Verifier E = w1*DTW + w2*FFT + w3*spatial, best-anchor-per-fine
        # ------------------------------------------------------------
        n_coarse = len(coarse_trend)
        E_matrix = np.zeros((n_fine_kept, n_coarse), dtype=np.float64)
        for i in range(n_fine_kept):
            trend_fine_grid = self._cluster_series_as_grid(fine_trend_cluster[i])
            seasonal_fine_grid = self._cluster_series_as_grid(fine_seasonal_cluster[i])
            for j in range(n_coarse):
                trend_coarse_grid = self._cluster_series_as_grid(coarse_trend[j])
                seasonal_coarse_grid = self._cluster_series_as_grid(coarse_seasonal[j])
                e_dtw = self.dtw_scorer.dtw_field_score(trend_fine_grid, trend_coarse_grid)
                e_fft = self.freq_scorer.low_freq_field_score(seasonal_fine_grid, seasonal_coarse_grid)
                E_matrix[i, j] = cfg.w1_dtw * e_dtw + cfg.w2_fft * e_fft + cfg.w3_spatial * e_spatial[i]

        E_fine = E_matrix.min(axis=1)
        best_local = int(np.argmin(E_fine))
        best_idx_original = int(fine_idx[best_local])

        metrics = compute_multivar_mse(candidates, best_idx_original, data, lat, lon)
        metrics["final_candidate_t2m"] = t2m_denorm[best_idx_original]
        metrics["E_fine"] = E_fine
        metrics["spatial_scores"] = e_spatial
        return metrics
