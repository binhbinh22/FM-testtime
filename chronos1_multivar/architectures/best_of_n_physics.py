import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from run import ERA5DataLoader  # noqa: E402
from era5_physics_energy import ERA5PhysicsEnergyScorer  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import (  # noqa: E402
    build_config, get_window_with_wind_history, generate_multivar_candidates,
    compute_multivar_mse, aggregate_window_metrics, append_csv_row, build_shared_start_indices,
    N_BUDGET_TOTAL, compute_latitude_weights, compute_climatology,
)
from proposers import get_proposer, resolve_model_id, BACKBONE_REGISTRY  # noqa: E402

ARCHITECTURE_NAME = "best_of_n_physics"

DEFAULT_CLIMATOLOGY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "era5_val_2015_2016.nc")


class BestOfNPhysicsMultivarEvaluator:
    """
    Sinh N candidate (sampling that, ca 3 bien), loc bang Physics Filter
    DUNG GIO DA DUOC DU BAO (tung candidate rieng, khong con ground truth
    tuong lai dung chung) - chi Physics Filter, KHONG trend/seasonal/cluster.

    CHI co y nghia voi backbone co sampler THAT (sampler in {autoregressive,
    mixture}): chronos, moirai1. Voi backbone quantile (moirai2/timesfm*),
    N candidate se trung/gan trung nhau -> Best-of-N khong co gi de chon
    (gain ~0); code se CANH BAO neu bi goi voi backbone do.
    """

    def __init__(self, config, n: int = N_BUDGET_TOTAL, backbone: str = "chronos",
                 climatology_path: str = DEFAULT_CLIMATOLOGY_PATH):
        self.config = config
        self.backbone = backbone
        self.n = n

        meta = BACKBONE_REGISTRY.get(backbone, {})
        if not meta.get("supports_bon", False):
            print("\n" + "!" * 70)
            print(f"[CANH BAO] backbone '{backbone}' la sampler='{meta.get('sampler')}' "
                  f"(quantile/tat dinh), KHONG sinh duoc N candidate i.i.d. da dang.")
            print("Best-of-N se chon giua cac ung vien trung/gan trung -> gain ~0.")
            print("Backbone nay CHI nen dung lam single-pass baseline (greedy.py).")
            print("!" * 70 + "\n")

        self.loader = ERA5DataLoader(config)
        self.forecaster = get_proposer(
            backbone, model_id=config.model_id,
            history_steps=config.history_steps, prediction_steps=config.prediction_steps,
        )
        self.lat_weight = compute_latitude_weights(self.loader.ds.latitude.values)
        self.climatology = compute_climatology(climatology_path, config)

    def run_test_set(self, n_windows: int = 10) -> dict:
        start_indices = build_shared_start_indices(
            self.loader, self.config.history_steps, self.config.prediction_steps, n_windows
        )
        print(f"start_indices ({n_windows} cua so): {start_indices.tolist()}")

        lat, lon = self.config.grid_size
        window_metrics = []
        for start_idx in start_indices:
            data = get_window_with_wind_history(self.loader, self.config, int(start_idx))
            candidates = generate_multivar_candidates(
                self.forecaster, data, self.config.prediction_steps, n_samples=self.n, temperature=1.0,
            )
            t2m_denorm = candidates["t2m"][0]
            u10_denorm = candidates["u10"][0]
            v10_denorm = candidates["v10"][0]

            scorer = ERA5PhysicsEnergyScorer(dt=self.config.interval_hours * 3600.0)
            scores = np.empty(self.n, dtype=np.float64)
            for i in range(self.n):
                details = scorer.composite_score(
                    t2m_denorm[i], u10_denorm[i], v10_denorm[i],
                    history_t2m=data["history_t2m"], lambda_activity=1.0,
                )
                scores[i] = details["composite_score"]
            best_idx = int(np.argmin(scores))

            window_metrics.append(compute_multivar_mse(
                candidates, best_idx, data, lat, lon,
                lat_weight=self.lat_weight, climatology=self.climatology,
            ))

        metrics = aggregate_window_metrics(window_metrics)
        print("\n" + "=" * 70)
        print(f"Best-of-N+Physics ({self.backbone}, da bien), N={self.n} - {n_windows} cua so")
        print(f"MSE da bien: {metrics['multivar_mse_norm_mean']:.6f} +/- {metrics['multivar_mse_norm_std']:.6f}")
        if "rmse_denorm_mean" in metrics:
            print(f"RMSE denorm: {metrics['rmse_denorm_mean']:.6f} | ACC denorm: {metrics['acc_denorm_mean']:.6f}")
        print("=" * 70)
        return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Chay Best-of-N+Physics (da backbone sampler-that) 1 lan, ghi 1 hang vao CSV.")
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--setting", type=str, default="custom")
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--backbone", type=str, default="chronos",
                    choices=["chronos", "moirai1", "moirai2", "timesfm", "timesfm3"],
                    help="Backbone. Best-of-N chi co y nghia voi chronos/moirai1 (sampler that).")
    p.add_argument("--model-id", type=str, default=None,
                    help="Override checkpoint. Bo trong -> dung default_model_id cua backbone.")
    p.add_argument("--n", type=int, default=N_BUDGET_TOTAL)
    p.add_argument("--climatology-path", type=str, default=DEFAULT_CLIMATOLOGY_PATH,
                    help="Du lieu dung tinh climatology cho ACC (mac dinh: tap val, tach biet voi data-path dang danh gia).")
    p.add_argument("--report-path", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "reports", "results_best_of_n_physics.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = resolve_model_id(args.backbone, args.model_id)
    config = build_config(args.history_steps, args.prediction_steps, model_id, args.data_path)

    evaluator = BestOfNPhysicsMultivarEvaluator(config, n=args.n, backbone=args.backbone,
                                                 climatology_path=args.climatology_path)
    metrics = evaluator.run_test_set(n_windows=args.n_windows)

    row = {
        "architecture": f"{ARCHITECTURE_NAME}__{args.backbone}",
        "setting": args.setting,
        "history_steps": args.history_steps,
        "prediction_steps": args.prediction_steps,
        "n_windows": args.n_windows,
        "n_fine": args.n,
        "n_coarse": 0,
        **metrics,
    }
    append_csv_row(args.report_path, row)
