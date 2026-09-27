import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from down import downsample_avgpool
from era5_physics_energy import ERA5PhysicsEnergyScorer
from trend_seasonal_decompose import centered_moving_average_decompose, kernel_size_for_scale
from dtw import ERA5DTWEnergyScorer
from frequency import ERA5FrequencyEnergyScorer
from pint_harmonic import PINTHarmonicOscillatorScorer


# ============================================================
# KIEN TRUC (theo dung mo ta):
#
# X (history) --downsample--> X_coarse
#
#   X_coarse --Chronos(N_C)--> candidates_coarse --Physics filter(k=5)--> coarse_top
#   X        --Chronos(N_A)--> candidates_fine   --Physics filter(k=5)--> fine_top
#
#   moi candidate con lai (ca 2 nhanh) --CMA decompose--> (Trend, Seasonal)
#
#   Trend_fine  vs Trend_coarse   --DTW (mien thoi gian)--> E_temporal
#   Seasonal_fine vs Seasonal_coarse --FFT/FreDF L1 (mien tan so)--> E_FFT
#
#   E(Y_fine, anchor) = w1 * E_temporal + w2 * E_FFT [+ w3 * E_harmonic(S_fine)]
#
#   Voi 5 anchor coarse cho 1 candidate fine, lay E_fine = min qua cac anchor
#   (candidate fine duoc "giai thich tot nhat" boi it nhat 1 anchor coarse -
#   cung quy uoc "best-distance-to-any-anchor" da dung trong
#   multiscale_search.py._filter_by_dtw; day la GIA DINH THIET KE, doi
#   thanh mean/weighted-mean qua cac anchor neu ban muon nghiem ngat hon).
#
#   Top-K theo E_fine thap nhat -> final candidate(s).
# ============================================================


@dataclass
class TrendSeasonalSearchConfig:
    """Cau hinh cho pipeline 2-scale Trend/Seasonal search."""

    downsample_factor: int = 2

    n_scenarios_fine: int = 10
    n_scenarios_coarse: int = 10

    k_physics_fine: int = 5
    k_physics_coarse: int = 5
    k_final: int = 1

    # kernel_size_fine=4 (voi interval_hours=6 -> 1 ngay/4 buoc): CMA 1 ngay
    # tach duoc dung chu ky ngay-dem (diurnal) ra Seasonal, giu Trend la
    # thanh phan bien thien cham (nhieu ngay). kernel coarse duoc quy doi
    # tuong ung qua kernel_size_for_scale.
    kernel_size_fine: int = 4

    lambda_activity: float = 1.0
    temperature_fine: float = 1.0
    temperature_coarse: float = 1.0

    # Trong so verifier: E = w1*E_temporal(DTW trend) + w2*E_FFT(seasonal)
    w1_temporal: float = 1.0
    w2_freq: float = 1.0

    # PINT harmonic oscillator: regularizer TUY CHON tren thanh phan
    # seasonal cua nhanh fine (mac dinh TAT, w3=0, de dung dung cong thuc
    # 2 so hang da neu). Bat bang cach set w3_harmonic > 0.
    w3_harmonic: float = 0.0
    period_days: float = 365.0

    dtw_window: Optional[int] = None


class MultiscaleTrendSeasonalSearchEngine:
    """
    Pipeline: Multiscale raw (2 scale) -> Physics filter moi nhanh ->
    Decompose Trend/Seasonal -> So sanh Trend (DTW) + Seasonal (FFT L1) ->
    Physics verifier tuyen tinh -> Top-K.

    Tai su dung nguyen ve: ChronosProposer (run.py), ERA5PhysicsEnergyScorer
    (era5_physics_energy.py), downsample_avgpool (down.py),
    ERA5DTWEnergyScorer (dtw.py), ERA5FrequencyEnergyScorer (frequency.py).
    """

    def __init__(self, proposer, config: Optional[TrendSeasonalSearchConfig] = None):
        self.proposer = proposer
        self.config = config or TrendSeasonalSearchConfig()

        self.dtw_scorer = ERA5DTWEnergyScorer(window=self.config.dtw_window, distance="abs")
        self.freq_scorer = ERA5FrequencyEnergyScorer(
            factor=self.config.downsample_factor, mode="complex_abs"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _physics_scorer_for(self, interval_hours: float) -> ERA5PhysicsEnergyScorer:
        """
        QUAN TRONG: dt cua continuity equation PHAI ty le voi do dai buoc
        thoi gian THUC TE cua nhanh dang xet. Nhanh coarse co buoc thoi
        gian dai gap `downsample_factor` lan nhanh fine, nen dt phai nhan
        theo factor tuong ung - neu dung chung 1 scorer voi dt co dinh cho
        ca 2 nhanh (nhu multiscale_search.py dang lam), residual PDE o
        nhanh coarse se bi tinh SAI ty le.
        """
        return ERA5PhysicsEnergyScorer(dt=interval_hours * 3600.0)

    def _physics_filter(
        self,
        candidates_denorm: np.ndarray,
        candidates_norm: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        history: np.ndarray,
        interval_hours: float,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
        scorer = self._physics_scorer_for(interval_hours)
        k_eff = min(k, candidates_denorm.shape[0])
        top_idx, top_denorm, details = scorer.select_top_k_composite(
            candidates_denorm,
            u_wind,
            v_wind,
            history_t2m=history,
            k=k_eff,
            lambda_activity=self.config.lambda_activity,
        )
        top_norm = candidates_norm[top_idx]
        return top_idx, top_denorm, top_norm, details

    def _decompose_batch(self, candidates: np.ndarray, kernel_size: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        trends, seasonals = [], []
        for i in range(candidates.shape[0]):
            trend, seasonal = centered_moving_average_decompose(candidates[i], kernel_size)
            trends.append(trend)
            seasonals.append(seasonal)
        return trends, seasonals

    def compute_metrics(
        self,
        selected_denorm: np.ndarray,
        selected_norm: np.ndarray,
        ground_truth: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        lat: int,
        lon: int,
    ) -> Dict[str, float]:
        gt = np.asarray(ground_truth)
        if selected_denorm.shape != gt.shape:
            raise ValueError(
                f"selected_denorm {selected_denorm.shape} khong khop ground_truth {gt.shape}"
            )

        mean_spatial = mean.reshape(lat, lon)
        std_spatial = std.reshape(lat, lon)
        gt_norm = (gt - mean_spatial[np.newaxis, :, :]) / std_spatial[np.newaxis, :, :]

        mse_denorm = float(np.mean((selected_denorm - gt) ** 2))
        mse_norm = float(np.mean((selected_norm - gt_norm) ** 2))
        mae_denorm = float(np.mean(np.abs(selected_denorm - gt)))
        mae_norm = float(np.mean(np.abs(selected_norm - gt_norm)))

        return {
            "mse_denorm": mse_denorm,
            "mse_norm": mse_norm,
            "rmse_denorm": float(np.sqrt(mse_denorm)),
            "rmse_norm": float(np.sqrt(mse_norm)),
            "mae_denorm": mae_denorm,
            "mae_norm": mae_norm,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        history: np.ndarray,
        u_wind_future: np.ndarray,
        v_wind_future: np.ndarray,
        prediction_steps: int,
        interval_hours: float,
        ground_truth: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        if history.ndim != 3:
            raise ValueError("history phai la rank-3: (time, lat, lon)")

        cfg = self.config
        factor = cfg.downsample_factor
        lat, lon = history.shape[1], history.shape[2]

        # -----------------------------------------------------
        # 1) Multiscale raw (2 scale): X -> X_coarse
        # -----------------------------------------------------
        history_coarse = downsample_avgpool(history, factor=factor)
        u_coarse = downsample_avgpool(u_wind_future, factor=factor)
        v_coarse = downsample_avgpool(v_wind_future, factor=factor)
        prediction_steps_coarse = max(1, prediction_steps // factor)

        # -----------------------------------------------------
        # 2) Chronos sinh candidate DOC LAP tren 2 scale
        # -----------------------------------------------------
        fine_denorm, fine_norm, fine_mean, fine_std = self.proposer.predict(
            history,
            prediction_steps=prediction_steps,
            temperature=cfg.temperature_fine,
            n_samples=cfg.n_scenarios_fine,
        )
        coarse_denorm, coarse_norm, _coarse_mean, _coarse_std = self.proposer.predict(
            history_coarse,
            prediction_steps=prediction_steps_coarse,
            temperature=cfg.temperature_coarse,
            n_samples=cfg.n_scenarios_coarse,
        )

        # -----------------------------------------------------
        # 3) Physics filter moi nhanh (k=5)
        # -----------------------------------------------------
        fine_idx, fine_top_denorm, fine_top_norm, fine_details = self._physics_filter(
            fine_denorm, fine_norm, u_wind_future, v_wind_future, history,
            interval_hours=interval_hours, k=cfg.k_physics_fine,
        )
        coarse_idx, coarse_top_denorm, coarse_top_norm, coarse_details = self._physics_filter(
            coarse_denorm, coarse_norm, u_coarse, v_coarse, history_coarse,
            interval_hours=interval_hours * factor, k=cfg.k_physics_coarse,
        )

        # -----------------------------------------------------
        # 4) Decompose Trend / Seasonal (Centered Moving Average)
        # -----------------------------------------------------
        kernel_fine = cfg.kernel_size_fine
        kernel_coarse = kernel_size_for_scale(kernel_fine, factor)

        fine_trend, fine_seasonal = self._decompose_batch(fine_top_denorm, kernel_fine)
        coarse_trend, coarse_seasonal = self._decompose_batch(coarse_top_denorm, kernel_coarse)

        # -----------------------------------------------------
        # 5) PINT harmonic oscillator (TUY CHON) tren Seasonal fine
        # -----------------------------------------------------
        harmonic_scores = None
        if cfg.w3_harmonic > 0:
            harmonic_scorer = PINTHarmonicOscillatorScorer(
                dt_days=interval_hours / 24.0, period_days=cfg.period_days
            )
            harmonic_scores = np.array(
                [harmonic_scorer.harmonic_residual_score(s) for s in fine_seasonal]
            )

        # -----------------------------------------------------
        # 6) Physics Verifier:
        #    E(Y_fine, anchor) = w1*E_temporal(T_fine,T_coarse)
        #                      + w2*E_FFT(S_fine,S_coarse)
        #                      [+ w3*E_harmonic(S_fine)]
        # -----------------------------------------------------
        n_fine = len(fine_trend)
        n_coarse = len(coarse_trend)
        temporal_matrix = np.zeros((n_fine, n_coarse), dtype=np.float64)
        freq_matrix = np.zeros((n_fine, n_coarse), dtype=np.float64)
        E_matrix = np.zeros((n_fine, n_coarse), dtype=np.float64)

        for i in range(n_fine):
            for j in range(n_coarse):
                e_temporal = self.dtw_scorer.dtw_field_score(fine_trend[i], coarse_trend[j])
                e_freq = self.freq_scorer.low_freq_field_score(fine_seasonal[i], coarse_seasonal[j])

                temporal_matrix[i, j] = e_temporal
                freq_matrix[i, j] = e_freq

                e_total = cfg.w1_temporal * e_temporal + cfg.w2_freq * e_freq
                if harmonic_scores is not None:
                    e_total += cfg.w3_harmonic * harmonic_scores[i]
                E_matrix[i, j] = e_total

        # Moi candidate fine duoc cham theo anchor coarse KHOP TOT NHAT.
        best_anchor_per_fine = E_matrix.argmin(axis=1)
        E_fine = E_matrix.min(axis=1)

        ranking = np.argsort(E_fine)
        k_final = min(cfg.k_final, n_fine)
        top_final_local_idx = ranking[:k_final]

        best_local = top_final_local_idx[0]
        best_anchor_j = best_anchor_per_fine[best_local]

        final_candidate = fine_top_denorm[best_local]
        final_candidate_norm = fine_top_norm[best_local]

        # -----------------------------------------------------
        # 7) MSE tren ground truth
        # -----------------------------------------------------
        metrics: Dict[str, float] = {}
        if ground_truth is not None:
            metrics = self.compute_metrics(
                selected_denorm=final_candidate,
                selected_norm=final_candidate_norm,
                ground_truth=ground_truth,
                mean=fine_mean,
                std=fine_std,
                lat=lat,
                lon=lon,
            )

        return {
            "final_candidate": final_candidate,
            "final_candidate_norm": final_candidate_norm,
            "final_fine_index": int(fine_idx[best_local]),
            "final_matched_coarse_index": int(coarse_idx[best_anchor_j]),
            "top_final_local_idx": top_final_local_idx,
            "E_fine": E_fine,
            "E_matrix": E_matrix,
            "temporal_matrix": temporal_matrix,
            "freq_matrix": freq_matrix,
            "harmonic_scores": harmonic_scores,
            "fine_top_idx": fine_idx,
            "coarse_top_idx": coarse_idx,
            "fine_physics_details": fine_details,
            "coarse_physics_details": coarse_details,
            "metrics": metrics,
        }
