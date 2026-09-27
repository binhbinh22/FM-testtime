import csv
import numpy as np
import torch

from run import ERA5Config, ERA5DataLoader
from chronos2_proposer import Chronos2Proposer
from spectral_clustering import spectral_cluster_grid
from spatial_cluster_search import SpatialClusterSearchConfig, SpatialClusterSearchEngine
from trend_seasonal_eval_utils import build_shared_start_indices


class SpatialClusterERA5Evaluator:
    """
    Wrapper CHAY XAC NHAN (khong dung de sweep) pipeline Spatial-Cluster
    BoN tren 1 tap du lieu cho truoc, voi 1 bo hyperparameter DA CHON san
    (C, N_A, N_C, K_A, w1, w2, w3 - xem EXPERIMENT_REPORT_fixed_N_budget_
    spatial_cluster.md de biet cach chon). Khong sua doi run.py /
    era5_physics_energy.py.

    Spectral Clustering (S, W) duoc build 1 LAN duy nhat trong __init__ -
    chi phu thuoc cau truc luoi (lat, lon) cua config.data_path, khong
    doi theo cua so du bao.

    QUAN TRONG: dung Chronos2Proposer (chronos2_proposer.py), KHONG dung
    ChronosProposer (Chronos T5 co trong run.py) - yeu cau rieng la chay
    THAT Chronos-2 multivariate (group attention giua cac node qua
    cross_learning=True), khong phai du bao tung node doc lap nhu
    Chronos T5. config.model_id PHAI la "amazon/chronos-2" (hoac bien
    the Chronos-2 khac) - KHONG dung gia tri mac dinh "amazon/chronos-t5-base"
    cua ERA5Config, model do khong tuong thich voi Chronos2Proposer.
    """

    def __init__(self, config: ERA5Config, search_config: SpatialClusterSearchConfig = None):
        if "chronos-2" not in config.model_id and "chronos2" not in config.model_id:
            raise ValueError(
                f"config.model_id='{config.model_id}' khong phai Chronos-2. "
                "SpatialClusterERA5Evaluator yeu cau Chronos2Proposer chay THAT "
                "Chronos-2 (vd model_id='amazon/chronos-2') de co group attention "
                "da node - dat lai ERA5Config(model_id='amazon/chronos-2', ...)."
            )
        self.config = config
        self.loader = ERA5DataLoader(config)
        self.forecaster = Chronos2Proposer(model_id=config.model_id, batch_size=100)
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

    def run_test_set(self, n_windows: int = 10, report_path: str = "report_spatial_cluster.csv"):
        start_indices = build_shared_start_indices(
            self.loader, self.config.history_steps, self.config.prediction_steps, n_windows
        )
        print(f"start_indices ({n_windows} cua so): {start_indices.tolist()}")

        mse_norm_list, mse_denorm_list = [], []
        mae_norm_list, mae_denorm_list = [], []
        spatial_score_means = []

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
            m = result["metrics"]
            mse_norm_list.append(m["mse_norm"])
            mse_denorm_list.append(m["mse_denorm"])
            mae_norm_list.append(m["mae_norm"])
            mae_denorm_list.append(m["mae_denorm"])
            spatial_score_means.append(float(np.mean(result["spatial_scores"])))

        metrics = {
            "mse_norm_mean": float(np.mean(mse_norm_list)),
            "mse_norm_std": float(np.std(mse_norm_list)),
            "mse_denorm_mean": float(np.mean(mse_denorm_list)),
            "mse_denorm_std": float(np.std(mse_denorm_list)),
            "mae_norm_mean": float(np.mean(mae_norm_list)),
            "mae_norm_std": float(np.std(mae_norm_list)),
            "mae_denorm_mean": float(np.mean(mae_denorm_list)),
            "mae_denorm_std": float(np.std(mae_denorm_list)),
            "spatial_score_mean": float(np.mean(spatial_score_means)),
            "spatial_score_std": float(np.std(spatial_score_means)),
        }

        print("\n" + "=" * 70)
        print(f"XAC NHAN 1 LAN tren {self.config.data_path} - {n_windows} cua so")
        print("=" * 70)
        print(f"MSE (denorm)      : {metrics['mse_denorm_mean']:.6f} +/- {metrics['mse_denorm_std']:.6f}")
        print(f"MAE (denorm)      : {metrics['mae_denorm_mean']:.6f} +/- {metrics['mae_denorm_std']:.6f}")
        print(f"MSE (norm)        : {metrics['mse_norm_mean']:.6f} +/- {metrics['mse_norm_std']:.6f}")
        print(f"MAE (norm)        : {metrics['mae_norm_mean']:.6f} +/- {metrics['mae_norm_std']:.6f}")
        print(f"Spatial-Err       : {metrics['spatial_score_mean']:.6f} +/- {metrics['spatial_score_std']:.6f}")
        print("=" * 70)

        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        print(f"Saved report: {report_path}")

        return metrics


if __name__ == "__main__":
    # QUAN TRONG: file nay chi dung de CHAY XAC NHAN 1 LAN DUY NHAT tren
    # tap TEST (era5_test_2017_2018.nc, mac dinh trong ERA5Config), voi
    # hyperparameter DA duoc chon san tu ket qua sweep tren tap VALIDATION
    # (xem EXPERIMENT_REPORT_fixed_N_budget_spatial_cluster.md - can chay
    # sweep that truoc, cac gia tri duoi day CHI la diem khoi dau tu trao
    # doi thiet ke, CHUA duoc tune). KHONG dung file nay de do tham so.
    BEST_C = 32
    BEST_N_A = 15
    BEST_N_C = 5
    BEST_K_A = 5
    BEST_W1 = 1.0
    BEST_W2 = 1.0
    BEST_W3 = 1.0
    KERNEL_SIZE_FINE = 16

    config = ERA5Config(history_steps=512, prediction_steps=64, model_id="amazon/chronos-2")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    search_config = SpatialClusterSearchConfig(
        n_clusters=BEST_C,
        downsample_factor=4,
        n_scenarios_coarse=BEST_N_C,
        n_scenarios_fine=BEST_N_A,
        k_physics_fine=BEST_K_A,
        kernel_size_fine=KERNEL_SIZE_FINE,
        w1_dtw=BEST_W1,
        w2_fft=BEST_W2,
        w3_spatial=BEST_W3,
        k_final=1,
    )

    evaluator = SpatialClusterERA5Evaluator(config, search_config)
    evaluator.run_test_set(n_windows=10, report_path="report_spatial_cluster_FINAL_TEST.csv")
