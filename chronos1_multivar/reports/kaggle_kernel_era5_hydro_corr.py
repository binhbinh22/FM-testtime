import os
import subprocess
import sys
import json

print("Installing chronos-forecasting...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chronos-forecasting"], check=True)

import numpy as np
import torch
import xarray as xr
from chronos import ChronosPipeline

"""
CORRELATION CHECK cho verifier Hydrostatic Balance tren ERA5 (5.625 do).

Do truc tiep Pearson correlation(diem hydrostatic residual tren CANDIDATE,
MSE THAT cua candidate) - dung PHUONG PHAP da dung cho L_pw/L_smooth
(WEATHER-5K): KHONG tin cong thuc chi vi no dung tren ground-truth, phai do
xem no co PHAN BIET candidate tot/xau do Chronos sinh ra khong.

Hydrostatic: r = (z500 - z1000) - R_dry * t850 * ln(2),  R_dry=287.
Score candidate = mean_t(r^2) (cang thap cang hop ly vat ly).
Candidate ghep theo CHI SO i: (z500_i, z1000_i, t850_i) - giong cach L_pw
ghep (P_i, V_i). Chronos sinh 3 bien doc lap.

Do correlation voi MSE cua TUNG bien (z500/z1000/t850) + ban DE-BIAS
(tru residual trung binh) de xu ly bias he thong t850 != T_bar.
"""

R_DRY = 287.0
LN2 = float(np.log(2.0))

HISTORY_STEPS = 48
PREDICTION_STEPS = 24
N_SAMPLES = 20
N_WINDOWS = 20
MODEL_ID = "amazon/chronos-t5-base"
BATCH_SIZE = 512
SEED = 0
OUT_JSON = "/kaggle/working/era5_hydro_corr_results.json"


def find_file(name):
    for root, _d, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    return None


def gen_candidates_batch(pipeline, hist_batch, pred_steps, n_samples):
    """hist_batch (n_series, hist) -> (n_series, n_samples, pred) DENORM."""
    mean = np.mean(hist_batch, axis=1, keepdims=True)
    std = np.std(hist_batch, axis=1, keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    ctx = torch.from_numpy(((hist_batch - mean) / std).astype(np.float32))
    with torch.inference_mode():
        fc = pipeline.predict(ctx, prediction_length=pred_steps, num_samples=n_samples,
                              top_k=50, top_p=1.0, temperature=1.0, limit_prediction_length=False)
    return fc.cpu().numpy() * std[:, None, :] + mean[:, None, :]


if __name__ == "__main__":
    path = find_file("era5_test_2017_2018_hydro.nc")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| data:", path)

    ds = xr.open_dataset(path)
    z500 = ds["z500"].values.reshape(ds.sizes["time"], -1)   # (time, n_series)
    z1000 = ds["z1000"].values.reshape(ds.sizes["time"], -1)
    t850 = ds["t850"].values.reshape(ds.sizes["time"], -1)
    n_time, n_series = z500.shape
    print("shape:", z500.shape)

    max_start = n_time - HISTORY_STEPS - PREDICTION_STEPS
    starts = np.linspace(0, max_start, N_WINDOWS, dtype=int)

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)

    hydro_scores, mse_z500, mse_z1000, mse_t850, residual_raw = [], [], [], [], []

    for wi, s in enumerate(starts):
        h0, h1, g1 = int(s), int(s) + HISTORY_STEPS, int(s) + HISTORY_STEPS + PREDICTION_STEPS
        gt = {"z500": z500[h1:g1], "z1000": z1000[h1:g1], "t850": t850[h1:g1]}  # (pred, n_series)
        cand = {}
        for name, arr in (("z500", z500), ("z1000", z1000), ("t850", t850)):
            out = np.empty((n_series, N_SAMPLES, PREDICTION_STEPS), dtype=np.float32)
            for b in range(0, n_series, BATCH_SIZE):
                e = min(b + BATCH_SIZE, n_series)
                out[b:e] = gen_candidates_batch(pipeline, arr[h0:h1].T[b:e], PREDICTION_STEPS, N_SAMPLES)
            cand[name] = out  # (n_series, N_SAMPLES, pred)

        for i in range(N_SAMPLES):
            cz5, cz1, ct8 = cand["z500"][:, i, :], cand["z1000"][:, i, :], cand["t850"][:, i, :]  # (n_series, pred)
            r = (cz5 - cz1) - R_DRY * ct8 * LN2                     # (n_series, pred)
            hydro_scores.append(np.mean(r ** 2, axis=1))           # (n_series,)
            residual_raw.append(np.mean(r, axis=1))
            mse_z500.append(np.mean((cz5 - gt["z500"].T) ** 2, axis=1))
            mse_z1000.append(np.mean((cz1 - gt["z1000"].T) ** 2, axis=1))
            mse_t850.append(np.mean((ct8 - gt["t850"].T) ** 2, axis=1))
        print(f"  window {wi+1}/{N_WINDOWS} done")

    hydro_scores = np.concatenate(hydro_scores)
    residual_raw = np.concatenate(residual_raw)
    mse_z500 = np.concatenate(mse_z500)
    mse_z1000 = np.concatenate(mse_z1000)
    mse_t850 = np.concatenate(mse_t850)
    # de-bias: score theo do lech khoi residual ky vong (uoc tu chinh tap candidate)
    hydro_debiased = (residual_raw - residual_raw.mean()) ** 2

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    results = {
        "n_pairs": int(hydro_scores.size),
        "config": {"history": HISTORY_STEPS, "pred": PREDICTION_STEPS, "n_samples": N_SAMPLES,
                   "n_windows": N_WINDOWS, "model": MODEL_ID},
        "corr_residual2_vs_MSE": {
            "z500": corr(hydro_scores, mse_z500),
            "z1000": corr(hydro_scores, mse_z1000),
            "t850": corr(hydro_scores, mse_t850),
        },
        "corr_DEBIASED_vs_MSE": {
            "z500": corr(hydro_debiased, mse_z500),
            "z1000": corr(hydro_debiased, mse_z1000),
            "t850": corr(hydro_debiased, mse_t850),
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== HYDROSTATIC CORRELATION CHECK =====")
    print(f"n_pairs = {results['n_pairs']}")
    print("corr(residual^2, MSE):  ", results["corr_residual2_vs_MSE"])
    print("corr(debiased,   MSE):  ", results["corr_DEBIASED_vs_MSE"])
