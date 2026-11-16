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
UNIFIED RUNNER v12 cho WEATHER-5K - port cau truc run_unified_era5.py sang
WEATHER-5K (tram roi rac), 1 lan sinh candidate -> moi method dung chung.

KHAC v11 (2 diem chinh theo yeu cau):
  1. "zero-shot" = GREEDY TAT DINH (top_k=1, argmax decode) - KHONG phai
     median cua N candidate nua. Giong ERA5 (temperature=0 single forecast).
  2. Them method "Ours" = TREND+SEASONAL+PHYSICS (port 1D cua engine ERA5
     trend_seasonal_multivar.py): chon candidate bang DTW(trend)+FFT(seasonal)
     khop coarse-anchor, cong L_pw cho cap SLP+WindSpeed. (KHONG co spatial
     cross-tram o ban nay - la 1D per-tram, theo lua chon da chot.)

Backbone: Chronos-1 (sampler that). Multi-window (6 cua so khap 2023).

Bang method x bien:
  greedy, mean_of_n, ours      : CA 4 bien (Temp/Dewpoint/WindSpeed/SLP)
  best_of_n/random/turbo/cem   : CHI SLP+WindSpeed (L_pw moi ap dung duoc)
"""

ALL_VARS = ["TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP"]
TS_ALL_VARS = {"Temp": "TMP", "Dewpoint": "DEW", "WindSpeed": "WND_RATE", "SLP": "SLP"}
PW_VARS = {"WindSpeed": "WND_RATE", "SLP": "SLP"}   # cap dung L_pw

HISTORY_STEPS = 48
PREDICTION_STEPS = 24
N_SAMPLES = 20
N_WINDOWS = 6
N_COARSE = 5
K_FINE = 5
DOWNSAMPLE = 4
KERNEL_FINE = 7           # cua so moving-average cho trend (fine)
W_TREND = 1.0
W_SEASONAL = 1.0
W_PW = 1.0
MODEL_ID = "amazon/chronos-t5-base"
BATCH_SIZE = 64
SEED = 0
OUT_JSON = "/kaggle/working/weather5k_v12_unified_results.json"
OUT_CSV = "/kaggle/working/weather5k_v12_unified_summary.csv"


# ============================================================
# Helpers: physics, decompose, DTW, FFT
# ============================================================
def pw_score(P_c, V_c, alpha):
    dP = P_c[1:] - P_c[:-1]
    dV = V_c[1:] - V_c[:-1]
    return float(np.mean((dP - alpha * dV) ** 2))


def moving_avg_decompose(x, k):
    """Trend = trung binh truot (centered, pad bien), Seasonal = x - trend."""
    if k < 2:
        return x.copy(), np.zeros_like(x)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    trend = np.convolve(xp, kernel, mode="valid")[:len(x)]
    return trend, x - trend


def dtw_distance(a, b, band=None):
    """DTW 1D co ban (O(n*m)), co the khac do dai. band = Sakoe-Chiba (None=full)."""
    n, m = len(a), len(b)
    if band is None:
        band = max(n, m)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        for j in range(j0, j1 + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def fft_lowfreq_dist(a, b, n_low=4):
    """Khoang cach pho tan so thap (|FFT| cua n_low he so dau)."""
    fa = np.abs(np.fft.rfft(a))[:n_low]
    fb = np.abs(np.fft.rfft(b))[:n_low]
    L = min(len(fa), len(fb))
    return float(np.mean((fa[:L] - fb[:L]) ** 2))


def ts_score(cand, coarse_list, kernel_fine):
    """
    Diem trend+seasonal cho 1 candidate (cang THAP cang khop coarse-anchor):
    min over coarse: w_trend*DTW(trend) + w_seasonal*FFT(seasonal).
    cand: (pred,). coarse_list: list cac (pred_coarse,) anchor.
    """
    t_c, s_c = moving_avg_decompose(cand, kernel_fine)
    best = 1e18
    for co in coarse_list:
        kc = max(2, kernel_fine // DOWNSAMPLE)
        t_a, s_a = moving_avg_decompose(co, kc)
        e = W_TREND * dtw_distance(t_c, t_a) + W_SEASONAL * fft_lowfreq_dist(s_c, s_a)
        best = min(best, e)
    return best


# ============================================================
# TuRBO-1 + CEM (inline, giong v10/v11)
# ============================================================
def turbo1_minimize(score_fn, dim, budget=20, n_init=5, seed=0,
                    length_init=0.8, length_min=0.05, length_max=1.6, success_tol=3, n_candidates=500):
    rng = np.random.default_rng(seed)
    n_init = min(n_init, budget)
    X = rng.uniform(-1.0, 1.0, size=(n_init, dim))
    y = np.array([score_fn(X[i]) for i in range(n_init)], dtype=np.float64)
    best_idx = int(np.argmin(y)); center = X[best_idx].copy(); best_y = float(y[best_idx])
    length = length_init; fail_tol = max(1, dim); n_s, n_f = 0, 0
    for _ in range(max(0, budget - n_init)):
        k = (Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0)))
        gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0)
        gp.fit(X, y)
        lo = np.clip(center - length / 2, -1, 1); hi = np.clip(center + length / 2, -1, 1)
        cand = rng.uniform(lo, hi, size=(n_candidates, dim))
        mu, sd = gp.predict(cand, return_std=True)
        x_next = cand[int(np.argmin(mu + sd * rng.standard_normal(size=mu.shape)))]
        y_next = float(score_fn(x_next))
        X = np.vstack([X, x_next[None, :]]); y = np.append(y, y_next)
        if y_next < best_y - 1e-12:
            best_y = y_next; center = x_next.copy(); n_s += 1; n_f = 0
        else:
            n_f += 1; n_s = 0
        if n_s >= success_tol:
            length = min(length_max, 2 * length); n_s = 0
        elif n_f >= fail_tol:
            length = max(length_min, length / 2); n_f = 0
    bi = int(np.argmin(y))
    return X[bi]


def cem_minimize(score_fn, dim, budget=20, population=5, elite_frac=0.4, seed=0, init_std=0.5):
    rng = np.random.default_rng(seed)
    mean = np.zeros(dim); std = np.full(dim, init_std)
    n_elite = max(1, int(np.ceil(population * elite_frac))); n_iters = max(1, budget // population)
    allX, allY = [], []
    for _ in range(n_iters):
        X = np.clip(rng.normal(mean, std, size=(population, dim)), -1, 1)
        y = np.array([score_fn(X[i]) for i in range(population)], dtype=np.float64)
        allX.append(X); allY.append(y)
        elite = X[np.argsort(y)[:n_elite]]
        mean = elite.mean(axis=0); std = elite.std(axis=0) + 1e-6
    allX = np.vstack(allX); allY = np.concatenate(allY)
    return allX[int(np.argmin(allY))]


def find_data_file(fn):
    for root, _d, files in os.walk("/kaggle/input"):
        if fn in files:
            return os.path.join(root, fn)
    return None


def load_npz(p):
    d = np.load(p, allow_pickle=True)
    return {"data": d["data"], "variables": [str(v) for v in d["variables"]]}


def gen_batch(pipeline, hist_batch, pred_steps, n_samples, top_k):
    mean = np.mean(hist_batch, axis=1, keepdims=True)
    std = np.std(hist_batch, axis=1, keepdims=True); std = np.where(std < 1e-5, 1.0, std)
    ctx = torch.from_numpy(((hist_batch - mean) / std).astype(np.float32))
    with torch.inference_mode():
        fc = pipeline.predict(ctx, prediction_length=pred_steps, num_samples=n_samples,
                              top_k=top_k, top_p=1.0, temperature=1.0, limit_prediction_length=False)
    return fc.cpu().numpy() * std[:, None, :] + mean[:, None, :]


if __name__ == "__main__":
    val_path = find_data_file("val_2022.npz")
    test_path = find_data_file("test_2023.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    val = load_npz(val_path); test = load_npz(test_path)
    n_stations = val["data"].shape[0]; n_hours = test["data"].shape[1]
    vix = {v: val["variables"].index(v) for v in ALL_VARS}
    rng = np.random.default_rng(SEED)

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)

    # alpha (L_pw) fit tu val-2022
    vP = val["data"][:, :, vix["SLP"]]; vV = val["data"][:, :, vix["WND_RATE"]]
    dP = vP[:, 1:] - vP[:, :-1]; dV = vV[:, 1:] - vV[:, :-1]
    alpha_all = np.sum(dP * dV, axis=1) / (np.sum(dV * dV, axis=1) + 1e-8)

    starts = np.linspace(0, n_hours - HISTORY_STEPS - PREDICTION_STEPS, N_WINDOWS, dtype=int)
    pred_coarse = max(1, PREDICTION_STEPS // DOWNSAMPLE)
    dim = 2 * PREDICTION_STEPS

    METHODS = ["greedy", "mean_of_n", "best_of_n_physics", "random_search",
               "turbo1_physics", "cem_physics", "ours_ts_physics"]
    results = {lab: {m: [] for m in METHODS} for lab in TS_ALL_VARS}

    t0 = time.time()
    done = 0
    total = N_WINDOWS * int(np.ceil(n_stations / BATCH_SIZE))

    for wi, s in enumerate(starts):
        h0, h1, g1 = int(s), int(s) + HISTORY_STEPS, int(s) + HISTORY_STEPS + PREDICTION_STEPS

        for b0 in range(0, n_stations, BATCH_SIZE):
            b1 = min(b0 + BATCH_SIZE, n_stations); b = b1 - b0

            # ---- sinh 1 lan: greedy (top_k=1), pool (N, top_k=50), coarse (downsample) ----
            pool, greedy, coarse = {}, {}, {}
            for lab, vn in TS_ALL_VARS.items():
                hist = test["data"][b0:b1, h0:h1, vix[vn]]
                greedy[lab] = gen_batch(pipeline, hist, PREDICTION_STEPS, 1, top_k=1)[:, 0, :]   # (b, pred)
                pool[lab] = gen_batch(pipeline, hist, PREDICTION_STEPS, N_SAMPLES, top_k=50)       # (b, N, pred)
                hist_ds = hist[:, ::DOWNSAMPLE]
                coarse[lab] = gen_batch(pipeline, hist_ds, pred_coarse, N_COARSE, top_k=50)         # (b, N_C, pred_c)

            gt = {lab: test["data"][b0:b1, h1:g1, vix[vn]] for lab, vn in TS_ALL_VARS.items()}

            for i in range(b):
                si = b0 + i
                alpha = float(alpha_all[si])

                # ===== greedy + mean_of_n: CA 4 bien =====
                for lab in TS_ALL_VARS:
                    results[lab]["greedy"].append(float(np.mean((greedy[lab][i] - gt[lab][i]) ** 2)))
                    results[lab]["mean_of_n"].append(float(np.mean((pool[lab][i].mean(0) - gt[lab][i]) ** 2)))

                # ===== SLP+WindSpeed: best_of_n/random/turbo/cem (L_pw joint) =====
                CP, CV = pool["SLP"][i], pool["WindSpeed"][i]       # (N, pred)
                zs_P, zs_V = np.median(CP, 0), np.median(CV, 0)
                sig_P = float(np.std(CP, 0).mean()); sig_V = float(np.std(CV, 0).mean())

                real = np.array([pw_score(CP[j], CV[j], alpha) for j in range(N_SAMPLES)])
                bi_real = int(np.argmin(real))

                nP = zs_P[None] + rng.normal(0, sig_P, size=(N_SAMPLES, PREDICTION_STEPS))
                nV = zs_V[None] + rng.normal(0, sig_V, size=(N_SAMPLES, PREDICTION_STEPS))
                ns = np.array([pw_score(nP[j], nV[j], alpha) for j in range(N_SAMPLES)])
                bi_noise = int(np.argmin(ns))

                seed_sw = SEED + si * N_WINDOWS + wi

                def score_fn(x, _P=zs_P, _V=zs_V, _sp=sig_P, _sv=sig_V, _a=alpha):
                    return pw_score(_P + x[:PREDICTION_STEPS] * _sp, _V + x[PREDICTION_STEPS:] * _sv, _a)

                xt = turbo1_minimize(score_fn, dim, budget=N_SAMPLES, seed=seed_sw)
                xc = cem_minimize(score_fn, dim, budget=N_SAMPLES, seed=seed_sw)
                P_t, V_t = zs_P + xt[:PREDICTION_STEPS] * sig_P, zs_V + xt[PREDICTION_STEPS:] * sig_V
                P_c, V_c = zs_P + xc[:PREDICTION_STEPS] * sig_P, zs_V + xc[PREDICTION_STEPS:] * sig_V

                for lab, (C, zs, nz, bt, bc) in {
                    "SLP": (CP, zs_P, nP[bi_noise], P_t, P_c),
                    "WindSpeed": (CV, zs_V, nV[bi_noise], V_t, V_c),
                }.items():
                    g = gt[lab][i]
                    results[lab]["best_of_n_physics"].append(float(np.mean((C[bi_real] - g) ** 2)))
                    results[lab]["random_search"].append(float(np.mean((nz - g) ** 2)))
                    results[lab]["turbo1_physics"].append(float(np.mean((bt - g) ** 2)))
                    results[lab]["cem_physics"].append(float(np.mean((bc - g) ** 2)))

                # ===== Ours (trend+seasonal [+physics SLP/WindSpeed]) =====
                # Temp, Dewpoint: chon doc lap theo trend+seasonal
                for lab in ("Temp", "Dewpoint"):
                    coarse_list = [coarse[lab][i][j] for j in range(N_COARSE)]
                    ts = np.array([ts_score(pool[lab][i][j], coarse_list, KERNEL_FINE) for j in range(N_SAMPLES)])
                    bi = int(np.argmin(ts))
                    results[lab]["ours_ts_physics"].append(float(np.mean((pool[lab][i][bi] - gt[lab][i]) ** 2)))

                # SLP + WindSpeed: chon JOINT index theo trend+seasonal(ca 2) + L_pw
                coarse_P = [coarse["SLP"][i][j] for j in range(N_COARSE)]
                coarse_V = [coarse["WindSpeed"][i][j] for j in range(N_COARSE)]
                joint = np.empty(N_SAMPLES)
                for j in range(N_SAMPLES):
                    e = (ts_score(CP[j], coarse_P, KERNEL_FINE) + ts_score(CV[j], coarse_V, KERNEL_FINE)
                         + W_PW * pw_score(CP[j], CV[j], alpha))
                    joint[j] = e
                bj = int(np.argmin(joint))
                results["SLP"]["ours_ts_physics"].append(float(np.mean((CP[bj] - gt["SLP"][i]) ** 2)))
                results["WindSpeed"]["ours_ts_physics"].append(float(np.mean((CV[bj] - gt["WindSpeed"][i]) ** 2)))

            done += 1
            if done % 10 == 0 or (wi == N_WINDOWS - 1 and b1 == n_stations):
                el = time.time() - t0; eta = (total - done) / (done / el) if done else 0
                print(f"  w{wi+1}/{N_WINDOWS} tram {b1}/{n_stations} (batch {done}/{total}, {el:.0f}s, ETA {eta:.0f}s)")

    # ---- tong hop ----
    rows = []
    for m in METHODS:
        row = {"method": m}
        for lab in TS_ALL_VARS:
            vals = results[lab][m]
            row[f"{lab}_MSE"] = float(np.mean(vals)) if vals else None
        rows.append(row)

    with open(OUT_JSON, "w") as f:
        json.dump({"config": {"history": HISTORY_STEPS, "pred": PREDICTION_STEPS, "n_samples": N_SAMPLES,
                              "n_windows": N_WINDOWS, "zero_shot": "greedy top_k=1 (tat dinh)",
                              "ours": "trend+seasonal+physics (1D, no spatial)", "model": MODEL_ID},
                   "mse": {lab: {m: float(np.mean(results[lab][m])) if results[lab][m] else None for m in METHODS}
                           for lab in TS_ALL_VARS}}, f, indent=2)

    print("\n===== v12 UNIFIED (greedy=tat dinh) =====")
    labs = list(TS_ALL_VARS)
    with open(OUT_CSV, "w") as f:
        f.write("method," + ",".join(f"{l}_MSE" for l in labs) + "\n")
        for r in rows:
            print(f"[{r['method']}] " + " ".join(
                f"{l}={r[f'{l}_MSE']:.4f}" if r[f'{l}_MSE'] is not None else f"{l}=N/A" for l in labs))
            f.write(r["method"] + "," + ",".join(
                "" if r[f'{l}_MSE'] is None else f"{r[f'{l}_MSE']:.6f}" for l in labs) + "\n")
    print(f"\nTotal: {time.time()-t0:.0f}s")
