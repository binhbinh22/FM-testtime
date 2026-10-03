import os
import sys

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from multiscale import downsample_avgpool  # noqa: E402
from era5_physics_energy import ERA5PhysicsEnergyScorer  # noqa: E402
from trend_seasonal_decompose import centered_moving_average_decompose, kernel_size_for_scale  # noqa: E402
from dtw import ERA5DTWEnergyScorer  # noqa: E402
from frequency import ERA5FrequencyEnergyScorer  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import generate_multivar_candidates, compute_multivar_mse  # noqa: E402


# ============================================================
# BIEN THE DA BIEN cua MultiscaleTrendSeasonalSearchEngine goc
# (../../multiscale_trend_seasonal_search.py) - KHONG sua file goc (dang
# dung o noi khac voi Chronos-2/ground-truth wind).
#
# Khac bản goc:
# 1) Fine branch sinh CA 3 bien (t2m,u10,v10) qua generate_multivar_candidates
#    (3 lan goi ChronosProposer doc lap), khong chi t2m.
# 2) Physics Filter dung gio DA DUOC DU BAO (tung candidate), khong dung
#    ground truth tuong lai dung chung.
# 3. DON GIAN HOA CO CHU DICH: Physics Filter CHI ap dung nhanh Fine,
#    KHONG ap dung nhanh Coarse (khac ban goc loc ca 2 nhanh) - vi nhanh
#    Coarse o day chi du bao t2m (khong du bao gio o cap do coarse), giu
#    do phuc tap trong tam kiem soat.
# ============================================================


@dataclass
class TrendSeasonalMultivarConfig:
    downsample_factor: int = 4
    n_fine: int = 15
    n_coarse: int = 5
    k_fine: int = 5
    kernel_size_fine: int = 16
    w1_dtw: float = 1.0
    w2_fft: float = 1.0
    lambda_activity: float = 1.0
    dtw_window: Optional[int] = None


class TrendSeasonalMultivarEngine:
    """
    Trend+Physics / Seasonal+Physics / Trend+Seasonal+Physics - 3 kien truc
    chung 1 engine nay, chi khac w1_dtw/w2_fft (xem architectures/
    trend_physics.py, seasonal_physics.py, trend_seasonal_physics.py).
    """

    def __init__(self, forecaster, config: Optional[TrendSeasonalMultivarConfig] = None):
        self.forecaster = forecaster
        self.config = config or TrendSeasonalMultivarConfig()
        self.dtw_scorer = ERA5DTWEnergyScorer(window=self.config.dtw_window, distance="abs")
        self.freq_scorer = ERA5FrequencyEnergyScorer(factor=self.config.downsample_factor, mode="complex_abs")

    def _decompose_batch(self, candidates: np.ndarray, kernel_size: int):
        trends, seasonals = [], []
        for i in range(candidates.shape[0]):
            trend, seasonal = centered_moving_average_decompose(candidates[i], kernel_size)
            trends.append(trend)
            seasonals.append(seasonal)
        return trends, seasonals

    def run(self, data: Dict, prediction_steps: int, interval_hours: float, lat: int, lon: int) -> Dict:
        cfg = self.config
        factor = cfg.downsample_factor

        # ------------------------------------------------------------
        # 1) Fine branch: sinh CA 3 bien, Physics Filter dung gio du bao
        # ------------------------------------------------------------
        candidates = generate_multivar_candidates(
            self.forecaster, data, prediction_steps, n_samples=cfg.n_fine, temperature=1.0,
        )
        t2m_denorm, t2m_norm, t2m_mean, t2m_std = candidates["t2m"]
        u10_denorm = candidates["u10"][0]
        v10_denorm = candidates["v10"][0]

        physics_scorer = ERA5PhysicsEnergyScorer(dt=interval_hours * 3600.0)
        scores = np.empty(cfg.n_fine, dtype=np.float64)
        for i in range(cfg.n_fine):
            details = physics_scorer.composite_score(
                t2m_denorm[i], u10_denorm[i], v10_denorm[i],
                history_t2m=data["history_t2m"], lambda_activity=cfg.lambda_activity,
            )
            scores[i] = details["composite_score"]
        k_eff = min(cfg.k_fine, cfg.n_fine)
        fine_idx = np.argsort(scores)[:k_eff]
        fine_top_denorm = t2m_denorm[fine_idx]

        # ------------------------------------------------------------
        # 2) Coarse branch: CHI t2m, downsample thoi gian, KHONG Physics Filter
        # ------------------------------------------------------------
        history_coarse = downsample_avgpool(data["history_t2m"], factor=factor)
        prediction_steps_coarse = max(1, prediction_steps // factor)
        coarse_denorm, _coarse_norm, _coarse_mean, _coarse_std = self.forecaster.predict(
            history_coarse, prediction_steps=prediction_steps_coarse, temperature=1.0, n_samples=cfg.n_coarse,
        )

        # ------------------------------------------------------------
        # 3) Temporal Decomp (Trend/Seasonal) ca 2 nhanh
        # ------------------------------------------------------------
        kernel_fine = cfg.kernel_size_fine
        kernel_coarse = kernel_size_for_scale(kernel_fine, factor)
        fine_trend, fine_seasonal = self._decompose_batch(fine_top_denorm, kernel_fine)
        coarse_trend, coarse_seasonal = self._decompose_batch(coarse_denorm, kernel_coarse)

        # ------------------------------------------------------------
        # 4) Verifier E = w1*DTW(trend) + w2*FFT(seasonal), best-anchor-per-fine
        # ------------------------------------------------------------
        n_fine_kept = len(fine_trend)
        n_coarse = len(coarse_trend)
        E_matrix = np.zeros((n_fine_kept, n_coarse), dtype=np.float64)
        for i in range(n_fine_kept):
            for j in range(n_coarse):
                e_dtw = self.dtw_scorer.dtw_field_score(fine_trend[i], coarse_trend[j])
                e_fft = self.freq_scorer.low_freq_field_score(fine_seasonal[i], coarse_seasonal[j])
                E_matrix[i, j] = cfg.w1_dtw * e_dtw + cfg.w2_fft * e_fft

        E_fine = E_matrix.min(axis=1)
        best_local = int(np.argmin(E_fine))
        best_idx_original = int(fine_idx[best_local])  # index trong N_fine ban dau (chua loc)

        metrics = compute_multivar_mse(candidates, best_idx_original, data, lat, lon)
        metrics["final_candidate_t2m"] = t2m_denorm[best_idx_original]
        metrics["E_fine"] = E_fine
        return metrics
