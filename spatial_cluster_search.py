import inspect
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from multiscale import downsample_avgpool
from era5_physics_energy import ERA5PhysicsEnergyScorer
from trend_seasonal_decompose import centered_moving_average_decompose, kernel_size_for_scale
from dtw import ERA5DTWEnergyScorer
from frequency import ERA5FrequencyEnergyScorer
from spectral_clustering import spectral_cluster_grid, cluster_pooling_matrix
from spatial_decompose import (
    aggregate_to_clusters,
    spatial_decompose,
    spatial_consistency_score,
    history_reference_Q,
)


# ============================================================
# KIEN TRUC (dung theo bao cao "thiet ke luong" da thong nhat, muc 2.1-2.5):
#
# X (2048 node) --Spectral Clustering--> S (V,C)
#
#   COARSE: S.X -> C sieu nut --Temporal Downsample--> Chronos(N_C) -> candidates_coarse
#            --Temporal Decomp--> T_coarse, S_coarse --FFT--> FFT(S_coarse)
#
#   FINE:   X --Chronos(N_A)--> candidates_fine
#            --PHYSIC Filter (node-level, composite_score)--> Top K_A
#            --Spatial Decomp (dung S)--> Y_global (--Temporal Decomp+FFT-->
#                                                     T_fine_cluster, FFT(S_fine_cluster))
#                                          Y_local (--spatial consistency score-->)
#
#   ST Verifier E = w1*DTW(T_coarse,T_fine_cluster)
#                 + w2*FFT_L1(S_coarse,S_fine_cluster)
#                 + w3*spatial_consistency_score(Y_local)
#   -> Y* = candidate trong Top K_A co E thap nhat.
#
# Tai su dung nguyen ve: ChronosProposer (run.py), ERA5PhysicsEnergyScorer
# (era5_physics_energy.py), centered_moving_average_decompose
# (trend_seasonal_decompose.py), ERA5DTWEnergyScorer (dtw.py),
# ERA5FrequencyEnergyScorer (frequency.py), downsample_avgpool (multiscale.py).
# ============================================================


@dataclass
class SpatialClusterSearchConfig:
    """Cau hinh cho pipeline Spatial-Cluster BoN (khong MCTS, sinh 1 lan)."""

    # --- Spectral Clustering (muc 2.1) ---
    n_clusters: int = 32  # C - xem trao doi truoc: patch ~8x8, can bang chi tiet vs compute
    k_neighbors: int = 8

    # --- Nhanh Coarse (muc 2.2) ---
    downsample_factor: int = 4
    n_scenarios_coarse: int = 5  # N_C

    # --- Nhanh Fine (muc 2.3) ---
    n_scenarios_fine: int = 15  # N_A
    k_physics_fine: int = 5  # K_A - top-K giu lai sau Physics Filter
    lambda_activity: float = 1.0  # chong trivial-flat-solution (era5_physics_energy.py)

    # --- Temporal Decomp (dung chung Trend/Seasonal, ca 2 nhanh) ---
    kernel_size_fine: int = 4

    # --- ST Verifier E (muc 2.5) ---
    w1_dtw: float = 1.0
    w2_fft: float = 1.0
    w3_spatial: float = 1.0
    k_final: int = 1

    temperature_fine: float = 1.0
    temperature_coarse: float = 1.0
    dtw_window: Optional[int] = None


class SpatialClusterSearchEngine:
    """
    Chay 1 cua so du bao theo dung 5 buoc 2.1-2.5. S (ma tran gan cum) va
    W (do thi kNN) duoc build 1 LAN ben ngoai (xem spatial_cluster_runner.py)
    va truyen vao qua constructor - KHONG build lai moi cua so, vi Spectral
    Clustering chi phu thuoc cau truc luoi (lat, lon), khong doi theo thoi
    gian.

    proposer PHAI la Chronos2Proposer (chronos2_proposer.py) de nhanh
    Fine/Coarse thuc su dung group attention da node (cross_learning=True)
    - day la yeu cau rieng, KHONG dung ChronosProposer (Chronos T5 co
    trong run.py, du bao tung node doc lap). Engine van chap nhan bat ky
    object nao co .predict(history, prediction_steps, temperature,
    n_samples) cung interface, nhung neu proposer KHONG ho tro tham so
    group_order (xem duoi), nhanh Fine se mat loi ich gom nhom theo cum
    khi chia batch cross-learning.
    """

    def __init__(
        self,
        proposer,
        S: np.ndarray,
        W: np.ndarray,
        config: Optional[SpatialClusterSearchConfig] = None,
    ):
        self.proposer = proposer
        self.S = S
        self.W = W
        self.config = config or SpatialClusterSearchConfig()

        if S.shape[1] != self.config.n_clusters:
            raise ValueError(
                f"S co C={S.shape[1]} cot nhung config.n_clusters={self.config.n_clusters} - khong khop."
            )

        self.dtw_scorer = ERA5DTWEnergyScorer(window=self.config.dtw_window, distance="abs")
        self.freq_scorer = ERA5FrequencyEnergyScorer(
            factor=self.config.downsample_factor, mode="complex_abs"
        )

        # Thu tu node gom theo cum (argmax S = nhan cum, one-hot) - dung
        # de truyen group_order cho Chronos2Proposer nhanh Fine, sao cho
        # moi batch cross_learning (~100 node, xem chronos2_proposer.py)
        # uu tien gom cac node CUNG 1 vung khi hau lai gan nhau, thay vi
        # chia theo thu tu phang lat/lon ngau nhien. Khong hoan hao (1
        # batch van co the lan sang cum ke ben neu cum > 100 node) nhung
        # tot hon khong gom gi ca.
        cluster_labels = np.argmax(self.S, axis=1)
        self._fine_group_order = np.argsort(cluster_labels)

    def _predict_with_optional_group_order(
        self, history: np.ndarray, prediction_steps: int, temperature: float, n_samples: int,
        group_order: Optional[np.ndarray],
    ):
        """Chi truyen group_order neu proposer.predict thuc su nhan tham so nay (Chronos2Proposer)."""
        params = inspect.signature(self.proposer.predict).parameters
        kwargs = {}
        if group_order is not None and "group_order" in params:
            kwargs["group_order"] = group_order
        return self.proposer.predict(
            history, prediction_steps=prediction_steps, temperature=temperature,
            n_samples=n_samples, **kwargs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _physics_scorer_for(self, interval_hours: float) -> ERA5PhysicsEnergyScorer:
        return ERA5PhysicsEnergyScorer(dt=interval_hours * 3600.0)

    def _flatten_grid(self, x: np.ndarray) -> np.ndarray:
        """(T, ny, nx) -> (T, V), V=ny*nx, thu tu khop voi era5_graph.flatten_latlon."""
        T = x.shape[0]
        return x.reshape(T, -1)

    def _unflatten_grid(self, x_flat: np.ndarray, ny: int, nx: int) -> np.ndarray:
        T = x_flat.shape[0]
        return x_flat.reshape(T, ny, nx)

    def _cluster_series_as_grid(self, X_c: np.ndarray) -> np.ndarray:
        """
        (T, C) -> (T, C, 1) de tai dung ChronosProposer.predict (nhan
        input rank-3 (time, lat, lon)) va ERA5DTWEnergyScorer/
        ERA5FrequencyEnergyScorer (nhan (T, ny, nx)) MA KHONG sua doi cac
        class do - coi C cum nhu 1 "luoi" C x 1.
        """
        return X_c[:, :, None]

    def _decompose_batch(
        self, candidates: np.ndarray, kernel_size: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
        T_hist, ny, nx = history.shape
        V = ny * nx
        if self.S.shape[0] != V:
            raise ValueError(f"S co V={self.S.shape[0]} nhung history co V={V} - khong khop luoi.")

        history_flat = self._flatten_grid(history)  # (T_hist, V)

        # -----------------------------------------------------
        # 2.1 (da lam ngoai engine): S da co san qua constructor.
        # -----------------------------------------------------

        # -----------------------------------------------------
        # 2.2 Nhanh Coarse: gop cum -> downsample thoi gian -> Chronos(N_C)
        # -----------------------------------------------------
        history_cluster = aggregate_to_clusters(history_flat, self.S)  # (T_hist, C)
        history_cluster_ds = downsample_avgpool(history_cluster, factor=factor)  # (T_hist/factor, C)
        prediction_steps_coarse = max(1, prediction_steps // factor)

        history_cluster_grid = self._cluster_series_as_grid(history_cluster_ds)  # (t, C, 1)
        # Khong can group_order o day: C thuong <= 100 (mac dinh C=32),
        # nam gon trong 1 batch cross_learning (chronos2_proposer.py) -
        # moi cum deu duoc du bao dong thoi, thu tu khong quan trong.
        coarse_denorm, coarse_norm, _coarse_mean, _coarse_std = self.proposer.predict(
            history_cluster_grid,
            prediction_steps=prediction_steps_coarse,
            temperature=cfg.temperature_coarse,
            n_samples=cfg.n_scenarios_coarse,
        )
        # (n_samples, pred_coarse, C, 1) -> (n_samples, pred_coarse, C)
        coarse_denorm = coarse_denorm[..., 0]

        # -----------------------------------------------------
        # 2.2 Temporal Decomp + FFT tren nhanh Coarse
        # -----------------------------------------------------
        kernel_fine = cfg.kernel_size_fine
        kernel_coarse = kernel_size_for_scale(kernel_fine, factor)
        coarse_trend, coarse_seasonal = self._decompose_batch(coarse_denorm, kernel_coarse)

        # -----------------------------------------------------
        # 2.3 Nhanh Fine: Chronos(N_A) tren toan luoi -> PHYSIC Filter -> Top K_A
        # -----------------------------------------------------
        fine_denorm, fine_norm, fine_mean, fine_std = self._predict_with_optional_group_order(
            history,
            prediction_steps=prediction_steps,
            temperature=cfg.temperature_fine,
            n_samples=cfg.n_scenarios_fine,
            group_order=self._fine_group_order,
        )

        physics_scorer = self._physics_scorer_for(interval_hours)
        k_eff = min(cfg.k_physics_fine, fine_denorm.shape[0])
        fine_idx, fine_top_denorm, physics_details = physics_scorer.select_top_k_composite(
            fine_denorm,
            u_wind_future,
            v_wind_future,
            history_t2m=history,
            k=k_eff,
            lambda_activity=cfg.lambda_activity,
        )
        fine_top_norm = fine_norm[fine_idx]

        # -----------------------------------------------------
        # 2.4 Spatial Decomp: Y_global (--Temporal Decomp+FFT-->), Y_local (--spatial score-->)
        # -----------------------------------------------------
        history_local_flat = history_flat - aggregate_to_clusters(history_flat, self.S) @ self.S.T
        hist_Q = history_reference_Q(history_local_flat, self.W)

        n_fine = fine_top_denorm.shape[0]
        fine_trend_cluster: List[np.ndarray] = []
        fine_seasonal_cluster: List[np.ndarray] = []
        e_spatial_per_fine = np.zeros(n_fine, dtype=np.float64)

        for i in range(n_fine):
            Y_fine_flat = self._flatten_grid(fine_top_denorm[i])  # (pred, V)
            Y_global_cluster, _Y_global_broadcast, Y_local = spatial_decompose(Y_fine_flat, self.S)

            trend_c, seasonal_c = centered_moving_average_decompose(Y_global_cluster, kernel_fine)
            fine_trend_cluster.append(trend_c)
            fine_seasonal_cluster.append(seasonal_c)

            e_spatial_per_fine[i] = spatial_consistency_score(Y_local, self.W, hist_Q)

        # -----------------------------------------------------
        # 2.5 ST Verifier E: DTW(trend) + FFT-L1(seasonal) + spatial consistency
        # -----------------------------------------------------
        n_coarse = len(coarse_trend)
        temporal_matrix = np.zeros((n_fine, n_coarse), dtype=np.float64)
        freq_matrix = np.zeros((n_fine, n_coarse), dtype=np.float64)
        E_matrix = np.zeros((n_fine, n_coarse), dtype=np.float64)

        for i in range(n_fine):
            trend_fine_grid = self._cluster_series_as_grid(fine_trend_cluster[i])
            seasonal_fine_grid = self._cluster_series_as_grid(fine_seasonal_cluster[i])
            for j in range(n_coarse):
                trend_coarse_grid = self._cluster_series_as_grid(coarse_trend[j])
                seasonal_coarse_grid = self._cluster_series_as_grid(coarse_seasonal[j])

                e_dtw = self.dtw_scorer.dtw_field_score(trend_fine_grid, trend_coarse_grid)
                e_fft = self.freq_scorer.low_freq_field_score(seasonal_fine_grid, seasonal_coarse_grid)

                temporal_matrix[i, j] = e_dtw
                freq_matrix[i, j] = e_fft
                E_matrix[i, j] = (
                    cfg.w1_dtw * e_dtw + cfg.w2_fft * e_fft + cfg.w3_spatial * e_spatial_per_fine[i]
                )

        # Moi candidate fine duoc cham theo anchor coarse KHOP TOT NHAT
        # (cung quy uoc "best-distance-to-any-anchor" voi
        # multiscale_trend_seasonal_search.py).
        best_anchor_per_fine = E_matrix.argmin(axis=1)
        E_fine = E_matrix.min(axis=1)

        ranking = np.argsort(E_fine)
        k_final = min(cfg.k_final, n_fine)
        top_final_local_idx = ranking[:k_final]

        best_local = int(top_final_local_idx[0])
        best_anchor_j = int(best_anchor_per_fine[best_local])

        final_candidate = fine_top_denorm[best_local]
        final_candidate_norm = fine_top_norm[best_local]

        metrics: Dict[str, float] = {}
        if ground_truth is not None:
            metrics = self.compute_metrics(
                selected_denorm=final_candidate,
                selected_norm=final_candidate_norm,
                ground_truth=ground_truth,
                mean=fine_mean,
                std=fine_std,
                lat=ny,
                lon=nx,
            )

        return {
            "final_candidate": final_candidate,
            "final_candidate_norm": final_candidate_norm,
            "final_fine_index": int(fine_idx[best_local]),
            "final_matched_coarse_index": best_anchor_j,
            "top_final_local_idx": top_final_local_idx,
            "E_fine": E_fine,
            "E_matrix": E_matrix,
            "temporal_matrix": temporal_matrix,
            "freq_matrix": freq_matrix,
            "spatial_scores": e_spatial_per_fine,
            "fine_top_idx": fine_idx,
            "fine_physics_details": physics_details,
            "history_reference_Q": hist_Q,
            "metrics": metrics,
        }


if __name__ == "__main__":
    # Smoke-test end-to-end voi Chronos proposer GIA (stub deterministic,
    # khong tai model that / khong can torch+GPU) - muc dich CHI kiem tra
    # wiring 5 buoc (shape khop, khong crash, verifier tra ve so huu han),
    # KHONG kiem tra chat luong du bao thuc te.
    class _StubProposer:
        """Gia lap ChronosProposer.predict: sinh N_samples bang history + nhieu."""

        def predict(self, history, prediction_steps, temperature, n_samples):
            T, a, b = history.shape
            mean = history.mean(axis=0, keepdims=True)
            std = history.std(axis=0, keepdims=True)
            std = np.where(std < 1e-5, 1.0, std)

            base = history[-1:].repeat(prediction_steps, axis=0)  # (pred, a, b)
            noise = temperature * 0.3 * np.random.randn(n_samples, prediction_steps, a, b)
            forecast_denorm = base[None] + noise
            forecast_norm = (forecast_denorm - mean[None]) / std[None]

            mean_flat = mean.reshape(a * b, 1)
            std_flat = std.reshape(a * b, 1)
            return forecast_denorm, forecast_norm, mean_flat, std_flat

    np.random.seed(0)

    ny, nx = 8, 16
    lat = np.linspace(-90, 90, ny)
    lon = np.linspace(0, 354.375, nx)
    C = 4
    T_hist = 24
    prediction_steps = 8
    interval_hours = 6.0

    S, labels, W = spectral_cluster_grid(lat, lon, n_clusters=C, k_neighbors=6)

    t = np.arange(T_hist)
    seasonal_true = np.sin(2 * np.pi * t / 4.0)[:, None, None]
    history = 280.0 + seasonal_true * np.ones((1, ny, nx)) + 0.5 * np.random.randn(T_hist, ny, nx)
    u10 = 2.0 * np.random.randn(prediction_steps, ny, nx)
    v10 = 2.0 * np.random.randn(prediction_steps, ny, nx)
    ground_truth = 280.0 + 0.5 * np.random.randn(prediction_steps, ny, nx)

    config = SpatialClusterSearchConfig(
        n_clusters=C,
        downsample_factor=2,
        n_scenarios_coarse=3,
        n_scenarios_fine=5,
        k_physics_fine=3,
        kernel_size_fine=4,
        w1_dtw=1.0,
        w2_fft=1.0,
        w3_spatial=1.0,
        k_final=1,
    )

    engine = SpatialClusterSearchEngine(proposer=_StubProposer(), S=S, W=W, config=config)

    result = engine.run(
        history=history,
        u_wind_future=u10,
        v_wind_future=v10,
        prediction_steps=prediction_steps,
        interval_hours=interval_hours,
        ground_truth=ground_truth,
    )

    print("final_candidate.shape:", result["final_candidate"].shape)
    assert result["final_candidate"].shape == (prediction_steps, ny, nx)

    print("E_fine:", np.round(result["E_fine"], 4))
    print("E_matrix.shape:", result["E_matrix"].shape)
    assert np.all(np.isfinite(result["E_fine"])), "E_fine phai huu han (khong NaN/Inf)"

    print("spatial_scores:", np.round(result["spatial_scores"], 4))
    print("history_reference_Q:", round(result["history_reference_Q"], 6))

    print("\nmetrics:", {k: round(v, 4) for k, v in result["metrics"].items()})
    assert result["metrics"]["mse_denorm"] >= 0

    print("\nfinal_fine_index:", result["final_fine_index"])
    print("final_matched_coarse_index:", result["final_matched_coarse_index"])
    print("\nOK - wiring 5 buoc (2.1-2.5) chay het khong crash.")
