import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from run import ERA5DataLoader, ChronosProposer  # noqa: E402
from spectral_clustering import spectral_cluster_grid  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import (  # noqa: E402
    build_config, get_window_with_wind_history, aggregate_window_metrics, append_csv_row,
    build_shared_start_indices, N_FINE_DEFAULT, N_COARSE_DEFAULT,
)
from engines.spatial_cluster_multivar import SpatialClusterMultivarEngine, SpatialClusterMultivarConfig  # noqa: E402

ARCHITECTURE_NAME = "cluster_spatial_physics"
W1_DTW = 1.0
W2_FFT = 1.0
W3_SPATIAL = 1.0  # phan khong gian BAT - kien truc day du


class ClusterSpatialPhysicsEvaluator:
    """
    Kien truc day du: nhanh Coarse xay bang Spectral Clustering, verifier
    gom ca Trend(DTW) + Seasonal(FFT) + do hop ly khong gian (spatial
    consistency, w3_spatial=1) tren phan du cuc bo tung node sau Spatial
    Decomp - khac cluster_physics.py (w3_spatial=0) dung de co lap rieng
    dong gop cua phan spatial nay.
    """

    def __init__(self, config, n_clusters=32, k_neighbors=24, n_fine=N_FINE_DEFAULT,
                 n_coarse=N_COARSE_DEFAULT, k_fine=5):
        self.config = config
        self.loader = ERA5DataLoader(config)
        self.forecaster = ChronosProposer(model_id=config.model_id)

        lat = self.loader.ds.latitude.values
        lon = self.loader.ds.longitude.values
        print(f"Spectral Clustering: V={lat.size * lon.size}, C={n_clusters}...")
        self.S, self.labels, self.W = spectral_cluster_grid(lat, lon, n_clusters=n_clusters, k_neighbors=k_neighbors)
        print("Cluster sizes:", self.S.sum(axis=0).astype(int).tolist())

        engine_config = SpatialClusterMultivarConfig(
            n_clusters=n_clusters, n_fine=n_fine, n_coarse=n_coarse, k_fine=k_fine,
            w1_dtw=W1_DTW, w2_fft=W2_FFT, w3_spatial=W3_SPATIAL,
        )
        self.engine = SpatialClusterMultivarEngine(self.forecaster, self.S, self.W, engine_config)

    def run_test_set(self, n_windows: int = 10) -> dict:
        start_indices = build_shared_start_indices(
            self.loader, self.config.history_steps, self.config.prediction_steps, n_windows
        )
        print(f"start_indices ({n_windows} cua so): {start_indices.tolist()}")

        lat, lon = self.config.grid_size
        window_metrics = []
        for start_idx in start_indices:
            data = get_window_with_wind_history(self.loader, self.config, int(start_idx))
            m = self.engine.run(data, self.config.prediction_steps, self.config.interval_hours, lat, lon)
            window_metrics.append(m)

        metrics = aggregate_window_metrics(window_metrics)
        print("\n" + "=" * 70)
        print(f"{ARCHITECTURE_NAME} (Chronos-1, da bien) - {n_windows} cua so")
        print(f"MSE da bien: {metrics['multivar_mse_norm_mean']:.6f} +/- {metrics['multivar_mse_norm_std']:.6f}")
        print("=" * 70)
        return metrics


def parse_args():
    p = argparse.ArgumentParser(description=f"Chay {ARCHITECTURE_NAME} (Chronos-1, da bien) 1 lan, ghi 1 hang vao CSV.")
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--setting", type=str, default="custom")
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--model-id", type=str, default="amazon/chronos-t5-base")
    p.add_argument("--n-clusters", type=int, default=32)
    p.add_argument("--k-neighbors", type=int, default=24)
    p.add_argument("--n-fine", type=int, default=N_FINE_DEFAULT)
    p.add_argument("--n-coarse", type=int, default=N_COARSE_DEFAULT)
    p.add_argument("--k-fine", type=int, default=5)
    p.add_argument("--report-path", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "reports", f"results_{ARCHITECTURE_NAME}.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args.history_steps, args.prediction_steps, args.model_id, args.data_path)

    evaluator = ClusterSpatialPhysicsEvaluator(
        config, n_clusters=args.n_clusters, k_neighbors=args.k_neighbors,
        n_fine=args.n_fine, n_coarse=args.n_coarse, k_fine=args.k_fine,
    )
    metrics = evaluator.run_test_set(n_windows=args.n_windows)

    row = {
        "architecture": ARCHITECTURE_NAME,
        "setting": args.setting,
        "history_steps": args.history_steps,
        "prediction_steps": args.prediction_steps,
        "n_windows": args.n_windows,
        "n_fine": args.n_fine,
        "n_coarse": args.n_coarse,
        **metrics,
    }
    append_csv_row(args.report_path, row)
