import os
import json
import subprocess
import sys
import time
import warnings

# Cai timesfm tu GIT neu CHUA co (Kaggle/Colab/server A100 deu dung chung kernel nay).
# User xac nhan chay OK tren Colab: cai tu GIT (KHONG phai PyPI) moi co module
# `timesfm3` voi TimesFM3Forecaster:
#   pip install git+https://github.com/google-research/timesfm.git
#   from timesfm3 import TimesFM3Forecaster
#   .predict(context=1d, horizon=H, return_quantiles=True) -> .forecast (H,), .quantiles (H,9)
try:
    import timesfm3  # noqa: F401
    print("timesfm3 da co san.")
except Exception:
    print("Installing timesfm (git) ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/google-research/timesfm.git"], check=True)

import numpy as np

warnings.filterwarnings("ignore")

"""
ABLATION KIEN TRUC (cumulative) cho ERA5 + TimesFM-3.0, physics = HYDROSTATIC.
=============================================================================
5 kien truc cong don (moi buoc them 1 thanh phan verifier), chon JOINT-1-INDEX
tren pool N candidate dung chung qua 3 bien (z500, z1000, t850):

    A1 Greedy (no TTC)        : point forecast TimesFM-3.0 (khong chon)
    A2 Best-of-N + physics    : chon theo hydrostatic residual
    A3 + trend (multiscale)   : + DTW(trend) voi coarse-anchor
    A4 + frequency (FreDF)    : + FFT low-freq(seasonal)
    A5 + spatial (Ours)       : + cluster-anchor deviation (spatial)

PHYSICS (da verify, era5_hydrostatic_verifier.py):
    r = (z500 - z1000) - R_dry * t850 * ln(2),  R_dry = 287 J/(kg*K)
    ghep CA 3 bien, score = mean(r^2). KHONG train gi (hang so vat ly).
    Nguon: dang vi phan ∂φ/∂p=−RT/p (DeepPrim C.5); dang tich phan hypsometric
    la TEXTBOOK (Holton & Hakim 2013), KHONG phai DeepPrim.

CANDIDATE: Gaussian copula correlated-quantile (muot), refer Baron et al.
    arXiv:2510.02224 - KHONG dung Algorithm 3 i.i.d. (rang cua) nua.

SEED: chay 1 seed (SEED=0) de kiem tra nhanh. Khi can bao cao mean+-std, boc
    vong seed quanh phan copula + chon (forecast TimesFM TAT DINH -> giu nguyen,
    chi lap lai phan sampling/selection).
"""

# ---------------- Config ----------------
VARS = ["z500", "z1000", "t850"]                 # 3 bien hydrostatic
DATA_FILE = "era5_test_2017_2018_hydro.nc"       # (time, lon=64, lat=32)

HISTORY_STEPS = 512                               # setting h512p64 (SETTINGS[0])
PREDICTION_STEPS = 64
N_WINDOWS = 3
N_SAMPLES = 20                                    # budget B (pool copula)
N_COARSE = 5
DOWNSAMPLE = 4
KERNEL_FINE = 7

N_CLUSTERS = 32
K_NEIGHBORS = 8
W_PHYS = W_TREND = W_FREQ = W_SPATIAL = 1.0       # trong so z-normed cong don

GRID_STRIDE = 1                                   # tang (vd 2,4) de chay nhanh tren tap con o luoi
BATCH_SIZE = 256                                  # so series/lan goi forecast (chia nho tranh OOM)
R_DRY = 287.0
LN2 = float(np.log(2.0))
MODEL_ID = "google/timesfm-3.0-pytorch"
SEED = 0
# Thu muc output: Kaggle (/kaggle/working), Colab (/content), hoac cwd.
OUT_DIR = next((d for d in ("/kaggle/working", "/content") if os.path.isdir(d)), ".")
OUT_JSON = os.path.join(OUT_DIR, "era5_arch_hydro_results.json")
OUT_CSV = os.path.join(OUT_DIR, "era5_arch_hydro_summary.csv")

ROWS = ["greedy", "bon_physics", "trend", "frequency", "spatial"]
ROW_LABEL = {
    "greedy": "Greedy (no TTC)",
    "bon_physics": "Best-of-N + physics",
    "trend": "+ trend (multiscale)",
    "frequency": "+ frequency (FreDF)",
    "spatial": "+ spatial (Ours)",
}


# ============================================================
# Candidate: Gaussian copula correlated-uniform (shared u qua 3 bien)
# ============================================================
def inverse_cdf(u, q_levels, q_values):
    q_levels = np.asarray(q_levels, float)
    u = np.clip(u, q_levels[0], q_levels[-1])
    idx = np.clip(np.searchsorted(q_levels, u, side="right") - 1, 0, len(q_levels) - 2)
    l0, l1 = q_levels[idx], q_levels[idx + 1]
    v0 = np.take_along_axis(q_values, idx[..., None], -1)[..., 0]
    v1 = np.take_along_axis(q_values, (idx + 1)[..., None], -1)[..., 0]
    w = (u - l0) / (l1 - l0 + 1e-12)
    return v0 + w * (v1 - v0)


def copula_u(n_draw, pred, rng, hist_series=None, rho=None, rho_clip=(0.3, 0.97)):
    """
    [MAC DINH] Correlated-uniform u (n_draw, pred) qua Gaussian copula AR(1),
    THAY cho i.i.d. uniform -> candidate MUOT. refer: Baron et al. arXiv:2510.02224.
    u DUNG CHUNG qua 3 bien (giu kich ban nhat quan, joint-1-index).
    """
    from scipy.stats import norm
    if rho is None:
        hs = None if hist_series is None else np.asarray(hist_series, float)
        if hs is not None and hs.size > 2 and hs.std() > 0:
            x = hs - hs.mean()
            rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        else:
            rho = 0.9
    if not np.isfinite(rho):
        rho = 0.9
    rho = float(np.clip(rho, rho_clip[0], rho_clip[1]))
    k = np.arange(pred)
    Sigma = rho ** np.abs(k[:, None] - k[None, :])
    L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(pred))
    z = (L @ rng.standard_normal((pred, n_draw))).T   # (n_draw, pred)
    return norm.cdf(z)


def copula_pool(q_by_var, q_levels, u):
    """
    q_by_var: dict var -> q_values (pred, n_q). u: (N-1, pred) copula DUNG CHUNG.
    Tra ve dict var -> (N, pred) = (N-1 copula sample voi u chung) + median.
    Index j giong nhau qua 3 bien => 1 kich ban hydrostatic nhat quan.
    """
    out = {}
    pred = u.shape[1]
    for v, qv in q_by_var.items():
        n_q = qv.shape[-1]
        median = inverse_cdf(np.full(pred, 0.5), q_levels, qv)
        qvb = np.broadcast_to(qv[None], (u.shape[0], pred, n_q))
        samples = inverse_cdf(u, q_levels, qvb)
        out[v] = np.vstack([samples, median[None]])  # (N, pred)
    return out


# ============================================================
# Physics (hydrostatic) + trend/seasonal + z-norm
# ============================================================
def hydrostatic_score(z500, z1000, t850):
    """Diem 1 candidate (THAP = thoa vat ly). z500/z1000/t850: (pred,)."""
    r = (z500 - z1000) - R_DRY * t850 * LN2
    return float(np.mean(r ** 2))


def moving_avg_decompose(x, k):
    if k < 2:
        return x.copy(), np.zeros_like(x)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    trend = np.convolve(xp, np.ones(k) / k, mode="valid")[:len(x)]
    return trend, x - trend


def dtw_distance(a, b, band=None):
    n, m = len(a), len(b)
    band = max(n, m) if band is None else band
    INF = 1e18
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def fft_lowfreq_dist(a, b, n_low=4):
    fa = np.abs(np.fft.rfft(a))[:n_low]
    fb = np.abs(np.fft.rfft(b))[:n_low]
    L = min(len(fa), len(fb))
    return float(np.mean((fa[:L] - fb[:L]) ** 2))


def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-12 else np.zeros_like(a)


# ============================================================
# Clustering o luoi (great-circle kNN + Spectral) - cho E_spatial
# ============================================================
EARTH_R = 6371.0088


def haversine_matrix(lat, lon):
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2)
    return EARTH_R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def grid_spectral_clusters(lat_flat, lon_flat, n_clusters, k_neighbors):
    """Spectral clustering tren do thi kNN Gaussian (great-circle). Tra labels, members."""
    from sklearn.cluster import SpectralClustering
    D = haversine_matrix(lat_flat, lon_flat)              # (V, V) km
    V = D.shape[0]
    knn = np.sort(D, axis=1)[:, 1:k_neighbors + 1]
    sigma = float(np.median(knn)) + 1e-9
    Wfull = np.exp(-D ** 2 / (2 * sigma ** 2))
    idx = np.argsort(D, axis=1)[:, 1:k_neighbors + 1]
    mask = np.zeros_like(Wfull)
    mask[np.repeat(np.arange(V), k_neighbors), idx.reshape(-1)] = 1.0
    mask = np.maximum(mask, mask.T)                        # doi xung hoa kNN
    W = Wfull * mask
    labels = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed",
        assign_labels="kmeans", random_state=SEED,
    ).fit_predict(W)
    members = [np.where(labels == c)[0] for c in range(n_clusters)]
    return labels, members


# ============================================================
# TimesFM-3.0 (API moi do user cung cap) - loader + forecast
# ============================================================
def load_timesfm3():
    from timesfm3 import TimesFM3Forecaster
    print("Loading", MODEL_ID, "...")
    return TimesFM3Forecaster.from_pretrained(MODEL_ID)


def tfm_forecast(forecaster, hist_list, horizon):
    """
    hist_list: list cac mang (len,). Tra ve point (B,horizon), quantile (B,horizon,9).
    API: forecaster.predict(context=1d, horizon=, return_quantiles=True)
         -> out.forecast (horizon,), out.quantiles (horizon, 9) [col 0=p10..4=p50..8=p90].
    Thu goi BATCH truoc (nhanh); neu khong ho tro -> loop tung series.
    [VERIFY] predict co nhan batch (list/2D) hay khong tren ban Kaggle.
    """
    # thu batch
    try:
        out = forecaster.predict(context=np.asarray(hist_list, np.float32),
                                 horizon=horizon, return_quantiles=True)
        pts = np.asarray(out.forecast, np.float32)
        qs = np.asarray(out.quantiles, np.float32)
        if pts.ndim == 2 and qs.ndim == 3:                # (B,H), (B,H,9)
            return pts[:, :horizon], qs[:, :horizon, :]
    except Exception as e:
        print("  [batch predict fail -> loop]:", type(e).__name__, str(e)[:90])
    # fallback: loop
    pts, qs = [], []
    for h in hist_list:
        out = forecaster.predict(context=np.asarray(h, np.float32),
                                 horizon=horizon, return_quantiles=True)
        pts.append(np.asarray(out.forecast, np.float32)[:horizon])
        qs.append(np.asarray(out.quantiles, np.float32)[:horizon, :])
    return np.stack(pts), np.stack(qs)


def forecast_all_cells(forecaster, hist_2d, horizon):
    """hist_2d: (n_cells, len). Chia BATCH_SIZE. Tra point (n_cells,horizon), quant (n_cells,horizon,9)."""
    n = hist_2d.shape[0]
    pts, qs = [], []
    for b0 in range(0, n, BATCH_SIZE):
        b1 = min(b0 + BATCH_SIZE, n)
        p, q = tfm_forecast(forecaster, [hist_2d[i] for i in range(b0, b1)], horizon)
        pts.append(p); qs.append(q)
    return np.concatenate(pts, 0), np.concatenate(qs, 0)


def find_file(name):
    # Kaggle (/kaggle/input), Colab (/content, Drive), hoac cwd
    for base in ("/kaggle/input", "/content", "."):
        if os.path.isdir(base):
            for root, _d, files in os.walk(base):
                if name in files:
                    return os.path.join(root, name)
    return name if os.path.exists(name) else None


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import torch
    import xarray as xr
    print("cuda:", torch.cuda.is_available())

    # ---- Data ERA5 hydro: (time, lon, lat) -> flatten (time, n_cells) ----
    path = find_file(DATA_FILE)
    if path is None:
        raise FileNotFoundError(f"Khong thay {DATA_FILE} trong /kaggle/input")
    ds = xr.open_dataset(path)
    lons = ds["longitude"].values            # (64,)
    lats = ds["latitude"].values             # (32,)
    LON, LAT = np.meshgrid(lons, lats, indexing="ij")   # (64,32)
    lon_flat = LON.reshape(-1); lat_flat = LAT.reshape(-1)   # (2048,)

    data = {}
    for v in VARS:
        arr = ds[v].values                   # (time, 64, 32)
        T = arr.shape[0]
        data[v] = arr.reshape(T, -1).astype(np.float32)     # (time, 2048)
    n_time = T
    n_cells_full = data[VARS[0]].shape[1]

    # subsample o luoi (neu GRID_STRIDE>1) - log ro de khong "am tham cat"
    sel = np.arange(0, n_cells_full, GRID_STRIDE)
    if GRID_STRIDE > 1:
        print(f"[GRID_STRIDE={GRID_STRIDE}] dung {sel.size}/{n_cells_full} o luoi")
    lat_sel, lon_sel = lat_flat[sel], lon_flat[sel]
    for v in VARS:
        data[v] = data[v][:, sel]
    n_cells = sel.size
    print(f"ERA5: time={n_time}, cells={n_cells} (grid {lons.size}x{lats.size}), vars={VARS}")

    # ---- Clustering 1 LAN (co dinh qua moi window) ----
    print(f"Spectral clustering: V={n_cells}, C={N_CLUSTERS}, k={K_NEIGHBORS}...")
    labels, cluster_members = grid_spectral_clusters(lat_sel, lon_sel, N_CLUSTERS, K_NEIGHBORS)

    # ---- windows (tat dinh, khong phu thuoc seed) ----
    last_start = n_time - HISTORY_STEPS - PREDICTION_STEPS
    if last_start < 0:
        raise ValueError("Chuoi thoi gian qua ngan cho h512p64")
    starts = np.unique(np.linspace(0, last_start, N_WINDOWS).astype(int))
    pred_coarse = max(1, PREDICTION_STEPS // DOWNSAMPLE)
    rng = np.random.default_rng(SEED)

    forecaster = load_timesfm3()
    results = {v: {r: [] for r in ROWS} for v in VARS}
    t0 = time.time()

    for wi, s in enumerate(starts):
        h0, h1, g1 = int(s), int(s) + HISTORY_STEPS, int(s) + HISTORY_STEPS + PREDICTION_STEPS

        # ---- forecast TAT DINH 1 lan: fine (point greedy + quantile) + coarse ----
        greedy, qf, qc, hist_w, gt = {}, {}, {}, {}, {}
        for v in VARS:
            hist_w[v] = data[v][h0:h1].T                    # (n_cells, H)
            gt[v] = data[v][h1:g1].T                        # (n_cells, pred)
            point, quant = forecast_all_cells(forecaster, hist_w[v], PREDICTION_STEPS)
            greedy[v] = point                               # (n_cells, pred)
            qf[v] = quant                                   # (n_cells, pred, 9)
            hist_ds = hist_w[v][:, ::DOWNSAMPLE]
            _pc, quant_c = forecast_all_cells(forecaster, hist_ds, pred_coarse)
            qc[v] = quant_c                                 # (n_cells, pred_coarse, 9)
        n_q = qf[VARS[0]].shape[-1]
        q_levels = (np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                    if n_q == 9 else np.linspace(0.1, 0.9, n_q))
        print(f"  w{wi+1}/{len(starts)} forecast xong ({time.time()-t0:.0f}s)")

        # ---- sinh pool copula (shared u) + coarse pool / o luoi ----
        pool = {v: np.empty((n_cells, N_SAMPLES, PREDICTION_STEPS), np.float32) for v in VARS}
        coarse = {v: np.empty((n_cells, N_COARSE, pred_coarse), np.float32) for v in VARS}
        for i in range(n_cells):
            u = copula_u(N_SAMPLES - 1, PREDICTION_STEPS, rng, hist_series=hist_w[VARS[0]][i])
            pf = copula_pool({v: qf[v][i] for v in VARS}, q_levels, u)
            uc = copula_u(N_COARSE - 1, pred_coarse, rng,
                          hist_series=hist_w[VARS[0]][i][::DOWNSAMPLE])
            pc = copula_pool({v: qc[v][i] for v in VARS}, q_levels, uc)
            for v in VARS:
                pool[v][i] = pf[v]; coarse[v][i] = pc[v]

        # ---- cluster-anchor cho E_spatial ----
        bg_fore = {v: np.zeros((N_CLUSTERS, PREDICTION_STEPS)) for v in VARS}
        ref_dev = {v: np.zeros(n_cells) for v in VARS}
        for v in VARS:
            for c, mem in enumerate(cluster_members):
                if mem.size == 0:
                    continue
                bg_fore[v][c] = np.median(greedy[v][mem], axis=0)
                bg_hist_c = np.median(hist_w[v][mem], axis=0)                 # (H,)
                ref_dev[v][mem] = np.mean(np.abs(hist_w[v][mem] - bg_hist_c[None]), axis=1)

        # ---- PASS 2: cham diem + chon cumulative / o luoi ----
        for i in range(n_cells):
            c = labels[i]
            Cz5, Cz1, Ct8 = pool["z500"][i], pool["z1000"][i], pool["t850"][i]   # (N, pred)

            e_phys = np.array([hydrostatic_score(Cz5[j], Cz1[j], Ct8[j]) for j in range(N_SAMPLES)])

            e_trend = np.zeros(N_SAMPLES); e_freq = np.zeros(N_SAMPLES)
            for v in VARS:
                anch = [coarse[v][i][a] for a in range(N_COARSE)]
                anch_ts = [moving_avg_decompose(a, max(2, KERNEL_FINE // DOWNSAMPLE)) for a in anch]
                for j in range(N_SAMPLES):
                    tj, sj = moving_avg_decompose(pool[v][i][j], KERNEL_FINE)
                    e_trend[j] += min(dtw_distance(tj, ta) for ta, _ in anch_ts)
                    e_freq[j] += min(fft_lowfreq_dist(sj, sa) for _, sa in anch_ts)

            e_spatial = np.zeros(N_SAMPLES)
            for v in VARS:
                rd = ref_dev[v][i]
                for j in range(N_SAMPLES):
                    dev = float(np.mean(np.abs(pool[v][i][j] - bg_fore[v][c])))
                    e_spatial[j] += abs(dev - rd) / (rd + 1e-6)

            zp, zt, zf, zs = zscore(e_phys), zscore(e_trend), zscore(e_freq), zscore(e_spatial)
            sel_idx = {
                "bon_physics": int(np.argmin(W_PHYS * zp)),
                "trend": int(np.argmin(W_PHYS * zp + W_TREND * zt)),
                "frequency": int(np.argmin(W_PHYS * zp + W_TREND * zt + W_FREQ * zf)),
                "spatial": int(np.argmin(W_PHYS * zp + W_TREND * zt + W_FREQ * zf + W_SPATIAL * zs)),
            }

            for v in VARS:
                g = gt[v][i]
                results[v]["greedy"].append(float(np.mean((greedy[v][i] - g) ** 2)))
                for row, j in sel_idx.items():
                    results[v][row].append(float(np.mean((pool[v][i][j] - g) ** 2)))

        print(f"  w{wi+1}/{len(starts)} scored ({time.time()-t0:.0f}s)")

    # ---- Tong hop + ghi bang ----
    table = {v: {r: float(np.mean(results[v][r])) for r in ROWS} for v in VARS}
    with open(OUT_JSON, "w") as f:
        json.dump({"config": {"model": MODEL_ID, "history": HISTORY_STEPS, "pred": PREDICTION_STEPS,
                              "n_samples": N_SAMPLES, "n_windows": int(len(starts)), "n_cells": int(n_cells),
                              "n_clusters": N_CLUSTERS, "k_neighbors": K_NEIGHBORS, "seed": SEED,
                              "physics": "hydrostatic r=(z500-z1000)-R_dry*t850*ln2, R_dry=287",
                              "candidate": "Gaussian copula (arXiv:2510.02224)",
                              "selection": "joint-1-index/cell, shared-u copula, z-normed cumulative",
                              "spatial": "cluster-anchor per-cell (grid spectral clustering)"},
                   "mse": table}, f, indent=2)

    print("\n===== ARCH ABLATION (ERA5, TimesFM-3.0, hydrostatic) - MSE =====")
    print(f"{'':24s} " + " ".join(f"{v:>10s}" for v in VARS))
    with open(OUT_CSV, "w") as f:
        f.write("architecture," + ",".join(VARS) + "\n")
        for r in ROWS:
            print(f"{ROW_LABEL[r]:24s} " + " ".join(f"{table[v][r]:10.3f}" for v in VARS))
            f.write(ROW_LABEL[r] + "," + ",".join(f"{table[v][r]:.5f}" for v in VARS) + "\n")
    print(f"\nTotal: {time.time()-t0:.0f}s | out: {OUT_JSON}")
