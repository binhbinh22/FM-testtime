import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run import ERA5DataLoader, ChronosProposer  # noqa: E402
from era5_physics_energy import ERA5PhysicsEnergyScorer  # noqa: E402
from spectral_clustering import spectral_cluster_grid  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    build_config, get_window_with_wind_history, generate_multivar_candidates,
    build_shared_start_indices, append_csv_row,
    compute_latitude_weights, compute_climatology, _rmse_acc_denorm_for,
    N_BUDGET_TOTAL, N_FINE_DEFAULT, N_COARSE_DEFAULT,
)
from engines.spatial_cluster_multivar import SpatialClusterMultivarEngine, SpatialClusterMultivarConfig  # noqa: E402
from search_strategies import random_search, cem, surrogate_cem, turbo1, best_of_n_select  # noqa: E402


# ============================================================
# "Comparison of Test-Time Search Strategies" (baseline tu DEFT).
# Tat ca dung CHUNG: backbone Chronos (frozen) + physics scorer (energy,
# minimize) + cung budget N (so query scorer). Khac nhau o CACH SEARCH.
#
# Khong gian search cho optimizer (random/cem/turbo/surrogate):
#   OUTPUT-SPACE CAP CUM (bo CDF, theo lua chon da chot):
#   theta in R^{C*K} = he so cua K ham co so thoi gian cho moi trong C cum.
#   decode(theta): offset cap cum -> broadcast ve V node -> cong vao base
#   forecast t2m (median cua backbone). Chi EDIT t2m (bien muc tieu); u10/v10
#   giu o base de lam "ngu canh vat ly" cho physics scorer.
#
# Metric bao cao: tren t2m (bien duoc edit) - RMSE/ACC denorm co trong so vi
# do + MSE norm, dung climatology tu tap val (giong phan con lai cua repo).
# ============================================================

STRATEGIES = ["random_search", "best_of_n_sample", "best_of_n_quantile",
              "cem", "turbo1", "surrogate_cem", "ours_scbon"]


def _temporal_basis(horizon: int, n_basis: int) -> np.ndarray:
    """K ham co so tron tru (DCT-I) tren horizon -> (H, K). Cot 0 = hang so."""
    t = np.arange(horizon)
    B = [np.ones(horizon)]
    for k in range(1, n_basis):
        B.append(np.cos(np.pi * k * (t + 0.5) / horizon))
    return np.stack(B, axis=1).astype(np.float64)  # (H, K)


class SearchStrategyComparison:
    def __init__(self, config, n: int = N_BUDGET_TOTAL, n_clusters: int = 32, k_neighbors: int = 24,
                 n_basis: int = 3, bound_scale: float = 0.3, seed: int = 0,
                 climatology_path: str = None):
        self.config = config
        self.n = n
        self.n_basis = n_basis
        self.bound_scale = bound_scale
        self.seed = seed

        self.loader = ERA5DataLoader(config)
        self.forecaster = ChronosProposer(model_id=config.model_id)

        lat_vals = self.loader.ds.latitude.values
        lon_vals = self.loader.ds.longitude.values
        print(f"Spectral Clustering: V={lat_vals.size * lon_vals.size}, C={n_clusters}...")
        self.S, self.labels, self.W = spectral_cluster_grid(lat_vals, lon_vals, n_clusters=n_clusters, k_neighbors=k_neighbors)
        self.n_clusters = self.S.shape[1]

        self.lat_weight = compute_latitude_weights(lat_vals)
        self.climatology = compute_climatology(climatology_path, config) if climatology_path else None

        # engine cho "Ours (SC-BoN)"
        eng_cfg = SpatialClusterMultivarConfig(
            n_clusters=n_clusters, n_fine=N_FINE_DEFAULT, n_coarse=N_COARSE_DEFAULT, k_fine=5,
            w1_dtw=1.0, w2_fft=1.0, w3_spatial=1.0,
        )
        self.engine = SpatialClusterMultivarEngine(self.forecaster, self.S, self.W, eng_cfg)

    # ---------- tien ich chung ----------
    def _t2m_metrics(self, t2m_field, data):
        """t2m_field: (H, lat, lon) denorm. Tra ve dict RMSE/ACC/MSE-norm cho t2m."""
        gt = data["gt_t2m"]
        out = {}
        if self.climatology is not None:
            rmse, acc = _rmse_acc_denorm_for(
                t2m_field, gt, self.climatology["t2m"], self.lat_weight.reshape(-1, 1))
            out["t2m_rmse_denorm"] = rmse
            out["t2m_acc_denorm"] = acc
        # MSE norm (z-score theo mean/std lich su t2m, per-node)
        mean = data["history_t2m"].mean(axis=0)
        std = data["history_t2m"].std(axis=0)
        std = np.where(std < 1e-5, 1.0, std)
        out["t2m_mse_norm"] = float(np.mean(((t2m_field - gt) / std[np.newaxis]) ** 2))
        return out

    def _build_search_problem(self, data):
        """
        Tra ve (decode, score_fn, dim, bounds) cho optimizer search tren t2m.
        base = median forecast backbone; u10/v10 = base lam ngu canh.
        """
        H = self.config.prediction_steps
        lat, lon = self.config.grid_size
        V = lat * lon

        cands = generate_multivar_candidates(self.forecaster, data, H, n_samples=1, temperature=0.0)
        base_t2m = cands["t2m"][0][0]   # (H, lat, lon)
        base_u10 = cands["u10"][0][0]
        base_v10 = cands["v10"][0][0]

        basis = _temporal_basis(H, self.n_basis)          # (H, K)
        K = self.n_basis
        dim = self.n_clusters * K

        # DUNG std THOI GIAN per-node (dao dong theo thoi gian tai tung diem),
        # KHONG dung std toan cuc - vi std toan cuc bi chenh lech khong gian
        # cuc-xich dao (~25-30 K) chi phoi -> bound qua rong -> optimizer edit
        # xa su that (reward-hack). std thoi gian ~5-10 K la "biet do hop ly".
        hist_std = float(data["history_t2m"].std(axis=0).mean())
        bnd = self.bound_scale * max(hist_std, 1e-3)
        lb = np.full(dim, -bnd); ub = np.full(dim, bnd)

        scorer = ERA5PhysicsEnergyScorer(dt=self.config.interval_hours * 3600.0)

        def decode(theta):
            coef = np.asarray(theta).reshape(self.n_clusters, K)     # (C, K)
            cluster_off = coef @ basis.T                              # (C, H)
            node_off = (self.S @ cluster_off).T                       # (H, V)
            field = base_t2m + node_off.reshape(H, lat, lon)
            return field

        def score_fn(theta):
            field = decode(theta)
            d = scorer.composite_score(field, base_u10, base_v10,
                                       history_t2m=data["history_t2m"], lambda_activity=1.0)
            return d["composite_score"]

        return decode, score_fn, dim, (lb, ub), base_t2m

    _OPTIMIZERS = {"random_search": random_search, "cem": cem,
                   "turbo1": turbo1, "surrogate_cem": surrogate_cem}

    def _run_strategy_on_window(self, strategy, data, ctx):
        """ctx = cache dung chung/window (base search problem) de tranh goi lai backbone."""
        H = self.config.prediction_steps
        lat, lon = self.config.grid_size

        if strategy == "ours_scbon":
            m = self.engine.run(data, H, self.config.interval_hours, lat, lon,
                                lat_weight=self.lat_weight, climatology=self.climatology)
            return self._t2m_metrics(m["final_candidate_t2m"], data)

        if strategy == "best_of_n_sample":
            cands = generate_multivar_candidates(self.forecaster, data, H, n_samples=self.n, temperature=1.0)
            t2m = cands["t2m"][0]; u10 = cands["u10"][0]; v10 = cands["v10"][0]
            scorer = ERA5PhysicsEnergyScorer(dt=self.config.interval_hours * 3600.0)
            scores = [scorer.composite_score(t2m[i], u10[i], v10[i],
                                             history_t2m=data["history_t2m"], lambda_activity=1.0)["composite_score"]
                      for i in range(self.n)]
            return self._t2m_metrics(t2m[best_of_n_select(scores)["best_index"]], data)

        if strategy == "best_of_n_quantile":
            fields, u10m, v10m = self._quantile_candidates(data, self.n)
            scorer = ERA5PhysicsEnergyScorer(dt=self.config.interval_hours * 3600.0)
            scores = [scorer.composite_score(fields[i], u10m, v10m,
                                             history_t2m=data["history_t2m"], lambda_activity=1.0)["composite_score"]
                      for i in range(len(fields))]
            return self._t2m_metrics(fields[best_of_n_select(scores)["best_index"]], data)

        # ---- optimizer-based: dung chung ctx (base problem) ----
        decode, score_fn, dim, bounds, base_t2m = ctx["problem"]
        res = self._OPTIMIZERS[strategy](score_fn, dim, bounds, self.n, np.random.default_rng(self.seed))
        theta = res["best_theta"]
        field = decode(theta) if theta is not None else base_t2m
        return self._t2m_metrics(field, data)

    def _quantile_candidates(self, data, n_levels):
        """
        N trajectory quantile t2m (doi xung quanh 0.5) qua Chronos
        predict_quantiles. Tra ve (list field (H,lat,lon), u10_med, v10_med).
        """
        import torch
        H = self.config.prediction_steps
        lat, lon = self.config.grid_size
        V = lat * lon
        levels = list(np.linspace(0.1, 0.9, n_levels))  # doi xung quanh 0.5
        if 0.5 not in levels:
            levels[len(levels) // 2] = 0.5

        def quantile_field(series_hist):  # series_hist: (T, lat, lon)
            ctx = series_hist.reshape(series_hist.shape[0], V).T.astype(np.float32)
            mean = ctx.mean(1, keepdims=True); std = ctx.std(1, keepdims=True)
            std = np.where(std < 1e-5, 1.0, std)
            normed = (ctx - mean) / std
            qs = []
            bs = self.forecaster.batch_size
            for s in range(0, V, bs):
                b = torch.from_numpy(normed[s:s+bs])
                q, _m = self.forecaster.pipeline.predict_quantiles(
                    b, prediction_length=H, quantile_levels=levels)
                qs.append(q.cpu().numpy())  # (b, H, n_levels)
            q = np.concatenate(qs, axis=0)  # (V, H, n_levels)
            q = q * std[:, :, None] + mean[:, :, None]  # denorm
            # -> list theo level: moi level 1 field (H, lat, lon)
            fields = [q[:, :, li].T.reshape(H, lat, lon) for li in range(len(levels))]
            med = q[:, :, len(levels) // 2].T.reshape(H, lat, lon)
            return fields, med

        t2m_fields, _ = quantile_field(data["history_t2m"])
        _, u10_med = quantile_field(data["history_u10"])
        _, v10_med = quantile_field(data["history_v10"])
        return t2m_fields, u10_med, v10_med

    # ---------- chay toan bo ----------
    def run(self, n_windows, strategies, report_path, setting):
        start_indices = build_shared_start_indices(
            self.loader, self.config.history_steps, self.config.prediction_steps, n_windows)
        print(f"start_indices ({n_windows}): {start_indices.tolist()}")

        needs_problem = any(s in self._OPTIMIZERS for s in strategies)
        agg = {s: [] for s in strategies}
        for start_idx in start_indices:
            data = get_window_with_wind_history(self.loader, self.config, int(start_idx))
            ctx = {}
            if needs_problem:
                ctx["problem"] = self._build_search_problem(data)  # 1 greedy call dung chung cho moi optimizer
            for s in strategies:
                try:
                    agg[s].append(self._run_strategy_on_window(s, data, ctx))
                except Exception as e:  # noqa: BLE001
                    print(f"[LOI {s} @ start={int(start_idx)}] {type(e).__name__}: {e}")

        for s in strategies:
            if not agg[s]:
                print(f"[BO QUA] {s}: khong co window nao chay duoc.")
                continue
            def mean(k):
                vals = [m[k] for m in agg[s] if k in m]
                return float(np.mean(vals)) if vals else ""
            row = {
                "strategy": s, "setting": setting,
                "history_steps": self.config.history_steps,
                "prediction_steps": self.config.prediction_steps,
                "n_windows": n_windows, "budget_N": self.n,
                "t2m_rmse_denorm_mean": mean("t2m_rmse_denorm"),
                "t2m_acc_denorm_mean": mean("t2m_acc_denorm"),
                "t2m_mse_norm_mean": mean("t2m_mse_norm"),
            }
            append_csv_row(report_path, row, fieldnames=list(row.keys()))
            print(f"{s:20s} RMSE={row['t2m_rmse_denorm_mean']}  ACC={row['t2m_acc_denorm_mean']}  MSEnorm={row['t2m_mse_norm_mean']}")


def parse_args():
    p = argparse.ArgumentParser(description="Comparison of Test-Time Search Strategies (DEFT baselines) tren t2m.")
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--setting", type=str, default="h512p64")
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--climatology-path", type=str, default=None)
    p.add_argument("--model-id", type=str, default="amazon/chronos-t5-base")
    p.add_argument("--n", type=int, default=N_BUDGET_TOTAL, help="Budget N = so query scorer, dung chung moi strategy.")
    p.add_argument("--n-clusters", type=int, default=32)
    p.add_argument("--n-basis", type=int, default=3, help="So ham co so thoi gian / cum (dim search = C*n_basis).")
    p.add_argument("--bound-scale", type=float, default=0.3,
                    help="Bound edit optimizer = bound_scale * std-thoi-gian per-node (nho -> tinh chinh nho, fair).")
    p.add_argument("--strategies", type=str, nargs="*", default=STRATEGIES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-path", type=str, default=os.path.join(
        os.path.dirname(__file__), "reports", "results_search_strategies.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args.history_steps, args.prediction_steps, args.model_id, args.data_path)
    comp = SearchStrategyComparison(
        config, n=args.n, n_clusters=args.n_clusters, n_basis=args.n_basis,
        bound_scale=args.bound_scale, seed=args.seed, climatology_path=args.climatology_path,
    )
    comp.run(args.n_windows, args.strategies, args.report_path, args.setting)
    print(f"\nHoan tat. Ket qua: {args.report_path}")
