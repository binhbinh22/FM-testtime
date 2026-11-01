import argparse
import csv
import os

import numpy as np

from run import ERA5Config, ERA5DataLoader, ChronosProposer
from spectral_clustering import spectral_cluster_grid
from spatial_cluster_search import SpatialClusterSearchConfig, SpatialClusterSearchEngine
from trend_seasonal_eval_utils import build_shared_start_indices


CSV_FIELDS = [
    "history_steps", "prediction_steps", "n_windows",
    "n_clusters", "n_fine", "n_coarse", "k_fine",
    "temperature_fine", "temperature_coarse",
    "w1_dtw", "w2_fft", "w3_spatial",
    "kernel_size_fine", "downsample_factor",
    "mse_norm_mean", "mse_norm_std",
]


class SpatialClusterChronos1Evaluator:
    """
    Ban Spatial-Cluster BoN dung Chronos-1 (Chronos T5 goc, ChronosProposer
    trong run.py) THAY CHO Chronos-2 - khai thac dung diem manh cua
    Chronos-1 la native stochastic trajectory sampling (moi candidate la
    1 lan autoregressive decode THAT, khac han Chronos-2 lay N muc
    quantile trong 1 lan forward).

    Cau hoi khoa hoc dang kiem chung: "Khi Chronos-1 da sinh ra nhieu
    possible futures (N trajectory sample), lieu thong tin spatial/
    physical co the dung o TEST TIME de chon trajectory tot hon khong,
    ma KHONG can retrain Chronos?" - dung CHINH engine
    SpatialClusterSearchEngine (khong sua 1 dong nao), chi doi proposer -
    toan bo verifier (Physics Filter, Spectral Clustering, Spatial
    Decomp, DTW/FFT/spatial-consistency) giu nguyen 100%, vi day la phan
    dang duoc kiem chung, khong phai phan thay doi.

    QUAN TRONG - khac Chronos-2: Chronos-1 KHONG co co che tuong tac
    giua cac series (khong co cross_learning) - moi node van du bao DOC
    LAP, N candidate la N lan sample THAT (ton that N lan decode, khac
    Chronos-2 ma N muc quantile gan nhu mien phi trong 1 lan forward -
    da thao luan va kiem chung truoc do).
    """

    def __init__(self, config: ERA5Config, search_config: SpatialClusterSearchConfig = None):
        self.config = config
        self.loader = ERA5DataLoader(config)
        self.forecaster = ChronosProposer(model_id=config.model_id)
        self.search_config = search_config or SpatialClusterSearchConfig()

        lat = self.loader.ds.latitude.values
        lon = self.loader.ds.longitude.values
        print(f"Spectral Clustering: V={lat.size * lon.size}, C={self.search_config.n_clusters}...")
        self.S, self.labels, self.W = spectral_cluster_grid(
            lat, lon, n_clusters=self.search_config.n_clusters, k_neighbors=self.search_config.k_neighbors
        )
        print("Cluster sizes:", self.S.sum(axis=0).astype(int).tolist())

        self.engine = SpatialClusterSearchEngine(
            proposer=self.forecaster, S=self.S, W=self.W, config=self.search_config
        )

    def run_test_set(self, n_windows: int = 10) -> dict:
        start_indices = build_shared_start_indices(
            self.loader, self.config.history_steps, self.config.prediction_steps, n_windows
        )
        print(f"start_indices ({n_windows} cua so): {start_indices.tolist()}")

        mse_norm_list = []
        for start_idx in start_indices:
            data = self.loader.get_test_window(int(start_idx))
            result = self.engine.run(
                history=data["history_t2m"],
                u_wind_future=data["u10"],
                v_wind_future=data["v10"],
                prediction_steps=self.config.prediction_steps,
                interval_hours=self.config.interval_hours,
                ground_truth=data["gt_t2m"],
            )
            mse_norm_list.append(result["metrics"]["mse_norm"])

        mse_norm_mean = float(np.mean(mse_norm_list))
        mse_norm_std = float(np.std(mse_norm_list))

        print("\n" + "=" * 70)
        print(f"Spatial-Cluster BoN (Chronos-1, sampling) - {n_windows} cua so")
        print(f"MSE (norm) : {mse_norm_mean:.6f} +/- {mse_norm_std:.6f}")
        print("=" * 70)

        return {"mse_norm_mean": mse_norm_mean, "mse_norm_std": mse_norm_std}


def append_csv_row(report_path: str, row: dict):
    file_exists = os.path.isfile(report_path)
    with open(report_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended 1 row -> {report_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Chay Spatial-Cluster BoN (Chronos-1/T5, native sampling) 1 lan, ghi 1 hang vao CSV."
    )
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--data-path", type=str, default=None, help="Mac dinh: dung gia tri co san trong ERA5Config.")
    p.add_argument("--model-id", type=str, default="amazon/chronos-t5-base")

    p.add_argument("--n-clusters", type=int, default=32, help="C - so cum Spectral Clustering.")
    p.add_argument("--k-neighbors", type=int, default=24)
    p.add_argument("--n-fine", type=int, default=15, help="N_A - so trajectory sample nhanh Fine.")
    p.add_argument("--n-coarse", type=int, default=5, help="N_C - so trajectory sample nhanh Coarse.")
    p.add_argument("--k-fine", type=int, default=5, help="K_A - top-K sau Physics Filter.")
    p.add_argument("--temperature-fine", type=float, default=1.0, help="Do da dang sampling nhanh Fine.")
    p.add_argument("--temperature-coarse", type=float, default=1.0, help="Do da dang sampling nhanh Coarse.")
    p.add_argument("--w1", type=float, default=1.0, help="Trong so DTW (trend).")
    p.add_argument("--w2", type=float, default=1.0, help="Trong so FFT-L1 (seasonal).")
    p.add_argument("--w3", type=float, default=1.0, help="Trong so spatial consistency.")
    p.add_argument("--kernel-size-fine", type=int, default=16)
    p.add_argument("--downsample-factor", type=int, default=4)

    p.add_argument("--report-path", type=str, default="report_spatial_cluster_chronos1.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config_kwargs = dict(
        history_steps=args.history_steps,
        prediction_steps=args.prediction_steps,
        model_id=args.model_id,
    )
    if args.data_path is not None:
        config_kwargs["data_path"] = args.data_path
    config = ERA5Config(**config_kwargs)

    search_config = SpatialClusterSearchConfig(
        n_clusters=args.n_clusters,
        k_neighbors=args.k_neighbors,
        downsample_factor=args.downsample_factor,
        n_scenarios_coarse=args.n_coarse,
        n_scenarios_fine=args.n_fine,
        k_physics_fine=args.k_fine,
        temperature_fine=args.temperature_fine,
        temperature_coarse=args.temperature_coarse,
        kernel_size_fine=args.kernel_size_fine,
        w1_dtw=args.w1,
        w2_fft=args.w2,
        w3_spatial=args.w3,
        k_final=1,
    )

    evaluator = SpatialClusterChronos1Evaluator(config, search_config)
    metrics = evaluator.run_test_set(n_windows=args.n_windows)

    row = {
        "history_steps": args.history_steps,
        "prediction_steps": args.prediction_steps,
        "n_windows": args.n_windows,
        "n_clusters": args.n_clusters,
        "n_fine": args.n_fine,
        "n_coarse": args.n_coarse,
        "k_fine": args.k_fine,
        "temperature_fine": args.temperature_fine,
        "temperature_coarse": args.temperature_coarse,
        "w1_dtw": args.w1,
        "w2_fft": args.w2,
        "w3_spatial": args.w3,
        "kernel_size_fine": args.kernel_size_fine,
        "downsample_factor": args.downsample_factor,
        "mse_norm_mean": metrics["mse_norm_mean"],
        "mse_norm_std": metrics["mse_norm_std"],
    }
    append_csv_row(args.report_path, row)
