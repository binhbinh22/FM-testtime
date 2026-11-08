import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run import ERA5DataLoader  # noqa: E402
from era5_physics_energy import ERA5PhysicsEnergyScorer  # noqa: E402
from spectral_clustering import spectral_cluster_grid  # noqa: E402
from multiscale import downsample_avgpool  # noqa: E402
from trend_seasonal_decompose import centered_moving_average_decompose, kernel_size_for_scale  # noqa: E402
from spatial_decompose import (  # noqa: E402
    aggregate_to_clusters, spatial_decompose, spatial_consistency_score, history_reference_Q,
)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common import (  # noqa: E402
    build_config, get_window_with_wind_history, generate_multivar_candidates,
    compute_multivar_mse, aggregate_window_metrics, append_csv_row, build_shared_start_indices,
    N_BUDGET_TOTAL, compute_latitude_weights, compute_climatology,
)
from engines.spatial_cluster_multivar import (  # noqa: E402
    SpatialClusterMultivarEngine, SpatialClusterMultivarConfig,
)
from proposers_quantile import (  # noqa: E402
    get_proposer_any, generate_multivar_candidates_algo3, DEFAULT_QUANTILE_LEVELS,
)

"""
RUNNER HOP NHAT (ERA5, Chronos-1) - 1 LAN CHAY ra NHIEU ket qua tren CUNG
pool candidate, de so sanh CONG BANG:

    1. Greedy                  (temperature=0, 1 candidate tat dinh)
    2. Best-of-N + Physics     (chon trong 20 fine bang physics energy)
    3. Cluster+Spatial+Physics (20 fine + 5 coarse anchor tu Chronos)

(Ban free-anchor da TAM BO theo yeu cau - method _build_coarse_anchors_free
van giu lai trong class de bat lai de dang sau nay, hien khong duoc goi.)

DIEM MAU CHOT (cong bang): Best-of-N va Cluster deu chon dap an cuoi tu
CUNG 1 pool 20 fine candidate (sinh DUY NHAT 1 LAN/cua so, seed co dinh).
Khong con nhieu sampling giua cac phuong phap - khac biet CHI do verifier.
Coarse anchor (ban 3) KHONG BAO GIO la dap an cuoi (chi la moc neo cho
verifier da tang), chi phi <0.1% (gop 32 cum, downsample 4x).

File nay KHONG sua engine/script cu: tai dung SpatialClusterMultivarEngine
bang SUBCLASS (them ham cham diem tren candidate CHO SAN), khong dong vao
spatial_cluster_multivar.py / best_of_n_physics.py / greedy.py.
"""

DEFAULT_CLIMATOLOGY_PATH = os.path.join(os.path.dirname(__file__), "..", "era5_val_2015_2016.nc")


class SharedCandidateClusterEngine(SpatialClusterMultivarEngine):
    """
    Mo rong engine goc: thay vi tu sinh candidate trong run(), nhan candidate
    CHO SAN tu ben ngoai (de chia se voi Best-of-N), va ho tro 2 che do moc neo:
      - 'tsfm' : moc neo = Chronos forecast tren lich su downsample (nhu goc)
      - 'free' : moc neo = consensus (median) cua chinh cac fine candidate,
                 KHONG goi Chronos them -> budget bang Best-of-N.
    """

    def _build_coarse_anchors_tsfm(self, data, prediction_steps):
        cfg = self.config
        factor = cfg.downsample_factor
        history_flat = self._flatten_grid(data["history_t2m"])
        history_cluster = aggregate_to_clusters(history_flat, self.S)
        history_cluster_ds = downsample_avgpool(history_cluster, factor=factor)
        prediction_steps_coarse = max(1, prediction_steps // factor)
        history_cluster_grid = self._cluster_series_as_grid(history_cluster_ds)

        coarse_denorm, _n, _m, _s = self.forecaster.predict(
            history_cluster_grid, prediction_steps=prediction_steps_coarse,
            temperature=1.0, n_samples=cfg.n_coarse,
        )
        coarse_denorm = coarse_denorm[..., 0]  # (n_coarse, pred_coarse, n_clusters)
        return self._decompose_coarse(coarse_denorm)

    def _build_coarse_anchors_free(self, candidates, prediction_steps):
        """
        [HIEN KHONG DUNG - tam bo free-anchor theo yeu cau, giu lai de bat lai]
        Moc neo mien phi: median cua 20 fine t2m -> gop cum -> downsample -> 1 anchor.
        """
        cfg = self.config
        factor = cfg.downsample_factor
        t2m_denorm = candidates["t2m"][0]  # (n_fine, pred, lat, lon)
        consensus = np.median(t2m_denorm, axis=0)  # (pred, lat, lon)
        consensus_flat = self._flatten_grid(consensus)  # (pred, lat*lon)
        consensus_cluster = aggregate_to_clusters(consensus_flat, self.S)  # (pred, n_clusters)
        consensus_cluster_ds = downsample_avgpool(consensus_cluster, factor=factor)
        return self._decompose_coarse(consensus_cluster_ds[None, ...])  # 1 anchor

    def _decompose_coarse(self, coarse_denorm):
        """coarse_denorm: (n_anchor, pred_coarse, n_clusters) -> list trend, list seasonal."""
        kernel_coarse = kernel_size_for_scale(self.config.kernel_size_fine, self.config.downsample_factor)
        coarse_trend, coarse_seasonal = [], []
        for j in range(coarse_denorm.shape[0]):
            t, s = centered_moving_average_decompose(coarse_denorm[j], kernel_coarse)
            coarse_trend.append(t)
            coarse_seasonal.append(s)
        return coarse_trend, coarse_seasonal

    def score_shared(self, candidates, coarse_trend, coarse_seasonal, data,
                     prediction_steps, interval_hours, lat, lon, lat_weight, climatology):
        """
        Replicate nhanh Fine + E-matrix + chon cua engine goc, NHUNG dung
        candidate CHO SAN + anchor CHO SAN (khong tu sinh gi). Tra ve metrics.
        """
        cfg = self.config
        t2m_denorm = candidates["t2m"][0]
        u10_denorm = candidates["u10"][0]
        v10_denorm = candidates["v10"][0]
        n_fine = t2m_denorm.shape[0]

        # Physics pre-filter -> top-k_fine (y het engine goc)
        physics_scorer = ERA5PhysicsEnergyScorer(dt=interval_hours * 3600.0)
        physics_scores = np.empty(n_fine, dtype=np.float64)
        for i in range(n_fine):
            d = physics_scorer.composite_score(
                t2m_denorm[i], u10_denorm[i], v10_denorm[i],
                history_t2m=data["history_t2m"], lambda_activity=cfg.lambda_activity,
            )
            physics_scores[i] = d["composite_score"]
        k_eff = min(cfg.k_fine, n_fine)
        fine_idx = np.argsort(physics_scores)[:k_eff]
        fine_top_denorm = t2m_denorm[fine_idx]

        # Spatial decomp tren tung fine candidate con lai
        history_flat = self._flatten_grid(data["history_t2m"])
        history_local_flat = history_flat - aggregate_to_clusters(history_flat, self.S) @ self.S.T
        hist_Q = history_reference_Q(history_local_flat, self.W)

        kernel_fine = cfg.kernel_size_fine
        n_fine_kept = fine_top_denorm.shape[0]
        fine_trend_cluster, fine_seasonal_cluster = [], []
        e_spatial = np.zeros(n_fine_kept, dtype=np.float64)
        for i in range(n_fine_kept):
            Y_fine_flat = self._flatten_grid(fine_top_denorm[i])
            Y_global_cluster, _bc, Y_local = spatial_decompose(Y_fine_flat, self.S)
            tc, sc = centered_moving_average_decompose(Y_global_cluster, kernel_fine)
            fine_trend_cluster.append(tc)
            fine_seasonal_cluster.append(sc)
            e_spatial[i] = spatial_consistency_score(Y_local, self.W, hist_Q)

        # E = w1*DTW(trend) + w2*FFT(seasonal) + w3*spatial, min over anchors
        n_anchor = len(coarse_trend)
        E_matrix = np.zeros((n_fine_kept, n_anchor), dtype=np.float64)
        for i in range(n_fine_kept):
            tf = self._cluster_series_as_grid(fine_trend_cluster[i])
            sf = self._cluster_series_as_grid(fine_seasonal_cluster[i])
            for j in range(n_anchor):
                tcj = self._cluster_series_as_grid(coarse_trend[j])
                scj = self._cluster_series_as_grid(coarse_seasonal[j])
                e_dtw = self.dtw_scorer.dtw_field_score(tf, tcj)
                e_fft = self.freq_scorer.low_freq_field_score(sf, scj)
                E_matrix[i, j] = cfg.w1_dtw * e_dtw + cfg.w2_fft * e_fft + cfg.w3_spatial * e_spatial[i]

        E_fine = E_matrix.min(axis=1)
        best_local = int(np.argmin(E_fine))
        best_idx_original = int(fine_idx[best_local])

        return compute_multivar_mse(
            candidates, best_idx_original, data, lat, lon,
            lat_weight=lat_weight, climatology=climatology,
        )


def set_global_seed(seed: int):
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_all(config, n_windows, n_clusters, k_neighbors, n_fine, n_coarse, k_fine,
            climatology_path, seed, report_path, backbone="chronos",
            quantile_levels=None):
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    quantile_levels = quantile_levels or DEFAULT_QUANTILE_LEVELS

    loader = ERA5DataLoader(config)
    forecaster = get_proposer_any(
        backbone, model_id=config.model_id, history_steps=config.history_steps,
        prediction_steps=config.prediction_steps,
    )
    lat_arr = loader.ds.latitude.values
    lon_arr = loader.ds.longitude.values
    lat_weight = compute_latitude_weights(lat_arr)
    climatology = compute_climatology(climatology_path, config)
    lat, lon = config.grid_size

    print(f"Spectral Clustering: V={lat_arr.size * lon_arr.size}, C={n_clusters}...")
    S, _labels, W = spectral_cluster_grid(lat_arr, lon_arr, n_clusters=n_clusters, k_neighbors=k_neighbors)

    cluster_cfg = SpatialClusterMultivarConfig(
        n_clusters=n_clusters, n_fine=n_fine, n_coarse=n_coarse, k_fine=k_fine,
        w1_dtw=1.0, w2_fft=1.0, w3_spatial=1.0,
    )
    engine = SharedCandidateClusterEngine(forecaster, S, W, cluster_cfg)
    physics_scorer = ERA5PhysicsEnergyScorer(dt=config.interval_hours * 3600.0)

    start_indices = build_shared_start_indices(
        loader, config.history_steps, config.prediction_steps, n_windows
    )
    print(f"start_indices ({n_windows} cua so): {start_indices.tolist()}")

    buckets = {name: [] for name in
               ["greedy", "best_of_n_physics", "cluster_spatial_physics"]}

    for start_idx in start_indices:
        data = get_window_with_wind_history(loader, config, int(start_idx))

        # ---- (1) Greedy: temperature=0, 1 candidate tat dinh (rieng, khong trong pool) ----
        greedy_cands = generate_multivar_candidates(
            forecaster, data, config.prediction_steps, n_samples=1, temperature=0.0,
        )
        buckets["greedy"].append(compute_multivar_mse(
            greedy_cands, 0, data, lat, lon, lat_weight=lat_weight, climatology=climatology,
        ))

        # ---- Pool CHUNG: n_fine candidate theo Algorithm 3 (sinh DUY NHAT 1 LAN) ----
        # Native sampler (chronos/moirai1): (n_fine-1) sample + median.
        # Quantile (chronos2/moirai2/timesfm*): inverse-CDF sample + median.
        candidates = generate_multivar_candidates_algo3(
            forecaster, backbone, data, config.prediction_steps, n_fine, quantile_levels, rng,
        )
        t2m_d, u10_d, v10_d = candidates["t2m"][0], candidates["u10"][0], candidates["v10"][0]

        # ---- (2) Best-of-N + Physics: chon trong 20 fine ----
        scores = np.empty(n_fine, dtype=np.float64)
        for i in range(n_fine):
            d = physics_scorer.composite_score(
                t2m_d[i], u10_d[i], v10_d[i], history_t2m=data["history_t2m"], lambda_activity=1.0,
            )
            scores[i] = d["composite_score"]
        best_idx = int(np.argmin(scores))
        buckets["best_of_n_physics"].append(compute_multivar_mse(
            candidates, best_idx, data, lat, lon, lat_weight=lat_weight, climatology=climatology,
        ))

        # ---- (3) Cluster+Spatial+Physics: 20 fine (CHUNG) + 5 coarse anchor tu Chronos ----
        ct, cs = engine._build_coarse_anchors_tsfm(data, config.prediction_steps)
        buckets["cluster_spatial_physics"].append(engine.score_shared(
            candidates, ct, cs, data, config.prediction_steps, config.interval_hours,
            lat, lon, lat_weight, climatology,
        ))

    # ---- Tong hop + ghi CSV (1 hang/phuong phap) ----
    print("\n" + "=" * 78)
    budget_note = {
        "greedy": "B=1 (tat dinh)",
        "best_of_n_physics": f"B={n_fine}",
        "cluster_spatial_physics": f"B={n_fine} fine + {n_coarse} coarse anchor (<0.1% compute)",
    }
    for name, wm in buckets.items():
        metrics = aggregate_window_metrics(wm)
        print(f"[{name}] {budget_note[name]}")
        print(f"    MSE da bien: {metrics['multivar_mse_norm_mean']:.6f} "
              f"+/- {metrics['multivar_mse_norm_std']:.6f}")
        if "rmse_denorm_mean" in metrics:
            print(f"    RMSE denorm: {metrics['rmse_denorm_mean']:.6f} | "
                  f"ACC denorm: {metrics['acc_denorm_mean']:.6f}")
        row = {
            "architecture": f"{name}__{backbone}",
            "setting": f"h{config.history_steps}p{config.prediction_steps}",
            "history_steps": config.history_steps,
            "prediction_steps": config.prediction_steps,
            "n_windows": n_windows,
            "n_fine": n_fine if name != "greedy" else 0,
            "n_coarse": n_coarse if name == "cluster_spatial_physics" else 0,
            **metrics,
        }
        append_csv_row(report_path, row)
    print("=" * 78)


def parse_args():
    p = argparse.ArgumentParser(
        description="Runner hop nhat ERA5: 1 lan chay ra Greedy + Best-of-N + Cluster+Spatial (TSFM & free anchor), "
                    "cung pool 20 fine candidate.")
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--backbone", type=str, default="chronos",
                   choices=["chronos", "chronos2", "moirai1", "moirai2", "timesfm", "timesfm3"],
                   help="Foundation model. chronos/moirai1 = sampler that; "
                        "chronos2/moirai2/timesfm/timesfm3 = quantile (Algorithm 3 inverse-CDF).")
    p.add_argument("--model-id", type=str, default=None,
                   help="Override checkpoint. Bo trong -> dung default cua backbone.")
    p.add_argument("--n-clusters", type=int, default=32)
    p.add_argument("--k-neighbors", type=int, default=24)
    p.add_argument("--n-fine", type=int, default=N_BUDGET_TOTAL, help="Pool CHUNG (mac dinh 20).")
    p.add_argument("--n-coarse", type=int, default=5)
    p.add_argument("--k-fine", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--climatology-path", type=str, default=DEFAULT_CLIMATOLOGY_PATH)
    p.add_argument("--report-path", type=str, default=os.path.join(
        os.path.dirname(__file__), "reports", "results_unified_era5.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args.history_steps, args.prediction_steps, args.model_id, args.data_path)
    run_all(
        config, n_windows=args.n_windows, n_clusters=args.n_clusters, k_neighbors=args.k_neighbors,
        n_fine=args.n_fine, n_coarse=args.n_coarse, k_fine=args.k_fine,
        climatology_path=args.climatology_path, seed=args.seed, report_path=args.report_path,
        backbone=args.backbone,
    )
