import json
import os
import subprocess
import sys
import time
import warnings

print("Installing chronos-forecasting...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chronos-forecasting"], check=True)

import numpy as np
import torch
from chronos import ChronosPipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

warnings.filterwarnings("ignore")

"""
FULL BASELINE SUITE v10 tren WEATHER-5K (5,672 tram) - mo rong tu v9
(Zero-shot/Random search/Mean-of-N/Best-of-N+Physics), THEM 2 baseline moi:
TuRBO-1 + Physics va Direct CEM + Physics (ca 2 deu dung L_pw lam ham
diem, budget = N_SAMPLES = 20 query moi tram).

CHAY LAI CA 4 BASELINE CU trong CUNG 1 kernel (khong ghep ket qua tu 2 lan
chay khac nhau) de dam bao CONG BANG: cung 1 lan sinh Chronos candidate,
cung 1 random seed, cung 1 alpha - chi khac chien luoc tim kiem/chon.

TuRBO-1, CEM hoat dong tren CUNG khong gian edit voi Random search: nhieu
them vao zero-shot median (zs_P, zs_V), bien do = sigma that (std cua N
candidate Chronos). Khac Random search (sample 1 lan N diem, chon tot
nhat), TuRBO-1/CEM TOI UU LAP qua budget=20 query (xem turbo1_search.py,
cem_search.py trong datasets/weather5k/ de biet chi tiet + gioi han).

L_pw can CA P (SLP) va V (Wind Speed) CUNG LUC -> TuRBO-1/CEM (nhu
Best-of-N+Physics) CHI ap dung cho cap (SLP, WindSpeed). Temp/Dewpoint
van KHONG CO verifier -> chi Zero-shot/Mean-of-N.
"""

ALL_VARS = ["TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP"]
NO_VERIFIER_VARS = {"Temp": "TMP", "Dewpoint": "DEW"}
PW_VARS = {"WindSpeed": "WND_RATE", "SLP": "SLP"}

HISTORY_STEPS = 48
PREDICTION_STEPS = 24
N_SAMPLES = 20          # budget cho moi baseline (bao gom TuRBO-1, CEM)
START_IDX = 0
MODEL_ID = "amazon/chronos-t5-base"
BATCH_SIZE = 64
SEED = 0
OUT_JSON = "/kaggle/working/weather5k_v10_turbo_cem_results.json"
OUT_CSV_SUMMARY = "/kaggle/working/weather5k_v10_turbo_cem_summary.csv"


# ---------------- TuRBO-1 (tu turbo1_search.py, inline de kernel tu chua) ----------------
def turbo1_minimize(score_fn, dim, budget=20, n_init=5, seed=0,
                     length_init=0.8, length_min=0.05, length_max=1.6,
                     success_tol=3, n_candidates=500):
    rng = np.random.default_rng(seed)
    n_init = min(n_init, budget)

    X = rng.uniform(-1.0, 1.0, size=(n_init, dim))
    y = np.array([score_fn(X[i]) for i in range(n_init)], dtype=np.float64)

    best_idx = int(np.argmin(y))
    center = X[best_idx].copy()
    best_y = float(y[best_idx])

    length = length_init
    fail_tol = max(1, dim)
    n_success, n_fail = 0, 0

    for _ in range(max(0, budget - n_init)):
        kernel = (Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                  + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0)
        gp.fit(X, y)

        lo = np.clip(center - length / 2.0, -1.0, 1.0)
        hi = np.clip(center + length / 2.0, -1.0, 1.0)
        candidates = rng.uniform(lo, hi, size=(n_candidates, dim))

        mu, std = gp.predict(candidates, return_std=True)
        thompson = mu + std * rng.standard_normal(size=mu.shape)
        x_next = candidates[int(np.argmin(thompson))]

        y_next = float(score_fn(x_next))
        X = np.vstack([X, x_next[None, :]])
        y = np.append(y, y_next)

        if y_next < best_y - 1e-12:
            best_y = y_next
            center = x_next.copy()
            n_success += 1
            n_fail = 0
        else:
            n_fail += 1
            n_success = 0

        if n_success >= success_tol:
            length = min(length_max, 2.0 * length)
            n_success = 0
        elif n_fail >= fail_tol:
            length = max(length_min, length / 2.0)
            n_fail = 0

    best_idx = int(np.argmin(y))
    return X[best_idx], float(y[best_idx])


# ---------------- Direct CEM (tu cem_search.py, inline) ----------------
def cem_minimize(score_fn, dim, budget=20, population=5, elite_frac=0.4, seed=0, init_std=0.5):
    rng = np.random.default_rng(seed)
    mean = np.zeros(dim)
    std = np.full(dim, init_std)
    n_elite = max(1, int(np.ceil(population * elite_frac)))
    n_iters = max(1, budget // population)

    all_X, all_y = [], []
    for _ in range(n_iters):
        X = rng.normal(mean, std, size=(population, dim))
        X = np.clip(X, -1.0, 1.0)
        y = np.array([score_fn(X[i]) for i in range(population)], dtype=np.float64)
        all_X.append(X)
        all_y.append(y)
        elite_idx = np.argsort(y)[:n_elite]
        elite = X[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-6

    all_X = np.vstack(all_X)
    all_y = np.concatenate(all_y)
    best_idx = int(np.argmin(all_y))
    return all_X[best_idx], float(all_y[best_idx])


def find_data_file(filename):
    for root, _dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    return None


def load_npz_split(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "data": d["data"], "station_ids": d["station_ids"],
        "latitude": d["latitude"], "longitude": d["longitude"],
        "variables": [str(v) for v in d["variables"]],
    }


def pw_score(P_c, V_c, alpha):
    dP = P_c[1:] - P_c[:-1]
    dV = V_c[1:] - V_c[:-1]
    return float(np.mean((dP - alpha * dV) ** 2))


def generate_chronos_candidates_batch(pipeline, history_batch, prediction_steps, n_samples):
    mean = np.mean(history_batch, axis=1, keepdims=True)
    std = np.std(history_batch, axis=1, keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    normalized = (history_batch - mean) / std
    context = torch.from_numpy(normalized.astype(np.float32))
    with torch.inference_mode():
        forecast = pipeline.predict(
            context, prediction_length=prediction_steps, num_samples=n_samples,
            top_k=50, top_p=1.0, temperature=1.0, limit_prediction_length=False,
        )
    forecast_norm = forecast.cpu().numpy()
    return forecast_norm * std[:, None, :] + mean[:, None, :]


if __name__ == "__main__":
    val_path = find_data_file("val_2022.npz")
    test_path = find_data_file("test_2023.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    val_npz = load_npz_split(val_path)
    test_npz = load_npz_split(test_path)
    n_stations = val_npz["data"].shape[0]
    var_idx = {v: val_npz["variables"].index(v) for v in ALL_VARS}
    rng = np.random.default_rng(SEED)

    print(f"Dang nap Chronos ({MODEL_ID}) tren {device}...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)
    print("Chronos da nap xong.")

    val_P = val_npz["data"][:, :, var_idx["SLP"]]
    val_V = val_npz["data"][:, :, var_idx["WND_RATE"]]
    dP_all = val_P[:, 1:] - val_P[:, :-1]
    dV_all = val_V[:, 1:] - val_V[:, :-1]
    alpha_all = np.sum(dP_all * dV_all, axis=1) / (np.sum(dV_all * dV_all, axis=1) + 1e-8)

    h_start, h_end = START_IDX, START_IDX + HISTORY_STEPS
    g_end = h_end + PREDICTION_STEPS
    dim = 2 * PREDICTION_STEPS  # nua dau: nhieu P (sigma-unit), nua sau: nhieu V

    results = {}
    for label in NO_VERIFIER_VARS:
        results[label] = {"zero_shot": [], "mean_of_n": []}
    for label in PW_VARS:
        results[label] = {
            "zero_shot": [], "random_search": [], "mean_of_n": [], "best_of_n_physics": [],
            "turbo1_physics": [], "cem_physics": [],
        }

    t0 = time.time()
    for batch_start in range(0, n_stations, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_stations)
        b = batch_end - batch_start

        # ---- Temp, Dewpoint: chi Zero-shot / Mean-of-N ----
        for label, var_name in NO_VERIFIER_VARS.items():
            hist_batch = test_npz["data"][batch_start:batch_end, h_start:h_end, var_idx[var_name]]
            future_batch = test_npz["data"][batch_start:batch_end, h_end:g_end, var_idx[var_name]]
            C = generate_chronos_candidates_batch(pipeline, hist_batch, PREDICTION_STEPS, N_SAMPLES)
            for i in range(b):
                gt = future_batch[i]
                zs = np.median(C[i], axis=0)
                mean_n = C[i].mean(axis=0)
                results[label]["zero_shot"].append(float(np.mean((zs - gt) ** 2)))
                results[label]["mean_of_n"].append(float(np.mean((mean_n - gt) ** 2)))

        # ---- SLP, WindSpeed: 6 baseline, dung chung khong gian edit ----
        hist_P_batch = test_npz["data"][batch_start:batch_end, h_start:h_end, var_idx["SLP"]]
        hist_V_batch = test_npz["data"][batch_start:batch_end, h_start:h_end, var_idx["WND_RATE"]]
        future_P_batch = test_npz["data"][batch_start:batch_end, h_end:g_end, var_idx["SLP"]]
        future_V_batch = test_npz["data"][batch_start:batch_end, h_end:g_end, var_idx["WND_RATE"]]

        cand_P = generate_chronos_candidates_batch(pipeline, hist_P_batch, PREDICTION_STEPS, N_SAMPLES)
        cand_V = generate_chronos_candidates_batch(pipeline, hist_V_batch, PREDICTION_STEPS, N_SAMPLES)

        for i in range(b):
            station_index = batch_start + i
            alpha = float(alpha_all[station_index])
            gt_P, gt_V = future_P_batch[i], future_V_batch[i]
            CP, CV = cand_P[i], cand_V[i]

            zs_P, zs_V = np.median(CP, axis=0), np.median(CV, axis=0)
            mean_P, mean_V = CP.mean(axis=0), CV.mean(axis=0)
            sigma_P = float(np.std(CP, axis=0).mean())
            sigma_V = float(np.std(CV, axis=0).mean())

            # Random search (1-shot, N_SAMPLES diem, giu nguyen nhu v9)
            noise_P = zs_P[None, :] + rng.normal(0, sigma_P, size=(N_SAMPLES, PREDICTION_STEPS))
            noise_V = zs_V[None, :] + rng.normal(0, sigma_V, size=(N_SAMPLES, PREDICTION_STEPS))
            noise_scores = np.array([pw_score(noise_P[j], noise_V[j], alpha) for j in range(N_SAMPLES)])
            best_noise_idx = int(np.argmin(noise_scores))

            # Best-of-N + Physics (chon trong chinh N candidate Chronos that)
            real_scores = np.array([pw_score(CP[j], CV[j], alpha) for j in range(N_SAMPLES)])
            best_real_idx = int(np.argmin(real_scores))

            # ---- TuRBO-1 + CEM: toi uu lap tren khong gian edit (don vi sigma) ----
            def score_fn(x, _zs_P=zs_P, _zs_V=zs_V, _sP=sigma_P, _sV=sigma_V, _alpha=alpha):
                P_c = _zs_P + x[:PREDICTION_STEPS] * _sP
                V_c = _zs_V + x[PREDICTION_STEPS:] * _sV
                return pw_score(P_c, V_c, _alpha)

            x_turbo, _ = turbo1_minimize(score_fn, dim=dim, budget=N_SAMPLES, seed=SEED + station_index)
            P_turbo = zs_P + x_turbo[:PREDICTION_STEPS] * sigma_P
            V_turbo = zs_V + x_turbo[PREDICTION_STEPS:] * sigma_V

            x_cem, _ = cem_minimize(score_fn, dim=dim, budget=N_SAMPLES, seed=SEED + station_index)
            P_cem = zs_P + x_cem[:PREDICTION_STEPS] * sigma_P
            V_cem = zs_V + x_cem[PREDICTION_STEPS:] * sigma_V

            results["SLP"]["zero_shot"].append(float(np.mean((zs_P - gt_P) ** 2)))
            results["SLP"]["mean_of_n"].append(float(np.mean((mean_P - gt_P) ** 2)))
            results["SLP"]["random_search"].append(float(np.mean((noise_P[best_noise_idx] - gt_P) ** 2)))
            results["SLP"]["best_of_n_physics"].append(float(np.mean((CP[best_real_idx] - gt_P) ** 2)))
            results["SLP"]["turbo1_physics"].append(float(np.mean((P_turbo - gt_P) ** 2)))
            results["SLP"]["cem_physics"].append(float(np.mean((P_cem - gt_P) ** 2)))

            results["WindSpeed"]["zero_shot"].append(float(np.mean((zs_V - gt_V) ** 2)))
            results["WindSpeed"]["mean_of_n"].append(float(np.mean((mean_V - gt_V) ** 2)))
            results["WindSpeed"]["random_search"].append(float(np.mean((noise_V[best_noise_idx] - gt_V) ** 2)))
            results["WindSpeed"]["best_of_n_physics"].append(float(np.mean((CV[best_real_idx] - gt_V) ** 2)))
            results["WindSpeed"]["turbo1_physics"].append(float(np.mean((V_turbo - gt_V) ** 2)))
            results["WindSpeed"]["cem_physics"].append(float(np.mean((V_cem - gt_V) ** 2)))

        if batch_end % (BATCH_SIZE * 10) == 0 or batch_end == n_stations:
            elapsed = time.time() - t0
            rate = batch_end / elapsed
            eta = (n_stations - batch_end) / rate if rate > 0 else 0
            print(f"  {batch_end}/{n_stations} tram ({elapsed:.1f}s, ETA {eta:.0f}s)")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "config": {
                "history_steps": HISTORY_STEPS, "prediction_steps": PREDICTION_STEPS,
                "n_samples": N_SAMPLES, "model_id": MODEL_ID, "n_stations": n_stations,
                "verifier": "L_pw only (pressure-wind consistency)",
                "turbo1_cem_note": "Cung khong gian edit voi random_search (nhieu quanh zero-shot median, "
                                    "bien do=sigma that cua N candidate Chronos); budget=N_SAMPLES moi tram.",
            },
            "mse_per_station": results,
        }, f)
    print(f"\nDa luu ket qua chi tiet -> {OUT_JSON}")

    print("\n===== TONG KET (trung binh MSE qua 5,672 tram) =====")
    all_labels = list(NO_VERIFIER_VARS) + list(PW_VARS)
    all_baselines = ["zero_shot", "random_search", "mean_of_n", "best_of_n_physics",
                      "turbo1_physics", "cem_physics"]
    rows = []
    for baseline in all_baselines:
        row = {"baseline": baseline, "N": 1 if baseline == "zero_shot" else N_SAMPLES}
        for label in all_labels:
            if baseline in results[label]:
                row[f"{label}_MSE"] = float(np.mean(results[label][baseline]))
            else:
                row[f"{label}_MSE"] = None
        rows.append(row)
        print(f"[{baseline}] " + " ".join(
            f"{label}={row[f'{label}_MSE']:.4f}" if row[f"{label}_MSE"] is not None else f"{label}=N/A"
            for label in all_labels
        ))

    with open(OUT_CSV_SUMMARY, "w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join("" if row[k] is None else str(row[k]) for k in keys) + "\n")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
