import json
import os
import subprocess
import sys
import time

print("Installing chronos-forecasting...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chronos-forecasting"], check=True)

import numpy as np
import torch
from chronos import ChronosPipeline

"""
KIEM TRA TRUC TIEP: diem L_pw (va L_smooth) cua tung candidate co tuong quan
voi MSE THAT (so voi ground truth) cua chinh candidate do khong?

Voi moi tram, moi candidate trong N=20: tinh ca (a) diem L_pw/L_smooth va
(b) MSE that so voi ground truth -> gop tat ca 5,672*20 = 113,440 cap
(score, MSE) lai, tinh Pearson correlation. Neu score va MSE TUONG QUAN
DUONG manh -> scorer dang chon dung huong (diem tot <-> MSE thap). Neu
tuong quan yeu/khong co -> scorer khong giup ich gi cho viec chon candidate.
"""

ALL_VARS = ["TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP"]
HISTORY_STEPS = 48
PREDICTION_STEPS = 24
N_SAMPLES = 20
START_IDX = 0
MODEL_ID = "amazon/chronos-t5-base"
BATCH_SIZE = 64
OUT_JSON = "/kaggle/working/weather5k_correlation_check_results.json"


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


def second_diff(x):
    return x[2:] - 2 * x[1:-1] + x[:-2]


def pw_score(P_c, V_c, alpha):
    dP = P_c[1:] - P_c[:-1]
    dV = V_c[1:] - V_c[:-1]
    return float(np.mean((dP - alpha * dV) ** 2))


def smooth_score(T_c, hist_var, eps=1e-6):
    d2T = second_diff(T_c)
    return float(np.mean(d2T ** 2)) / (hist_var + eps)


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

    print(f"Dang nap Chronos ({MODEL_ID}) tren {device}...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)
    print("Chronos da nap xong.")

    # alpha cho tat ca tram, fit tu val-2022 (vector hoa)
    val_P = val_npz["data"][:, :, var_idx["SLP"]]
    val_V = val_npz["data"][:, :, var_idx["WND_RATE"]]
    dP_all = val_P[:, 1:] - val_P[:, :-1]
    dV_all = val_V[:, 1:] - val_V[:, :-1]
    alpha_all = np.sum(dP_all * dV_all, axis=1) / (np.sum(dV_all * dV_all, axis=1) + 1e-8)

    h_start, h_end = START_IDX, START_IDX + HISTORY_STEPS
    g_end = h_end + PREDICTION_STEPS
    hist_T = test_npz["data"][:, h_start:h_end, var_idx["TMP"]]
    hist_V = test_npz["data"][:, h_start:h_end, var_idx["WND_RATE"]]
    hist_P = test_npz["data"][:, h_start:h_end, var_idx["SLP"]]
    future_T = test_npz["data"][:, h_end:g_end, var_idx["TMP"]]
    future_V = test_npz["data"][:, h_end:g_end, var_idx["WND_RATE"]]
    future_P = test_npz["data"][:, h_end:g_end, var_idx["SLP"]]

    # Thu thap toan bo cap (score, mse) qua moi (tram, candidate)
    pw_scores_all, mse_P_all, mse_V_all = [], [], []
    smooth_scores_all, mse_T_all = [], []

    t0 = time.time()
    for batch_start in range(0, n_stations, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_stations)
        b = batch_end - batch_start

        cand_T = generate_chronos_candidates_batch(pipeline, hist_T[batch_start:batch_end], PREDICTION_STEPS, N_SAMPLES)
        cand_V = generate_chronos_candidates_batch(pipeline, hist_V[batch_start:batch_end], PREDICTION_STEPS, N_SAMPLES)
        cand_P = generate_chronos_candidates_batch(pipeline, hist_P[batch_start:batch_end], PREDICTION_STEPS, N_SAMPLES)

        for i in range(b):
            station_index = batch_start + i
            alpha = float(alpha_all[station_index])
            hist_var_T = float(np.var(hist_T[station_index]))
            gt_T, gt_V, gt_P = future_T[station_index], future_V[station_index], future_P[station_index]

            for j in range(N_SAMPLES):
                T_c, V_c, P_c = cand_T[i, j], cand_V[i, j], cand_P[i, j]

                s_pw = pw_score(P_c, V_c, alpha)
                s_smooth = smooth_score(T_c, hist_var_T)

                pw_scores_all.append(s_pw)
                mse_P_all.append(float(np.mean((P_c - gt_P) ** 2)))
                mse_V_all.append(float(np.mean((V_c - gt_V) ** 2)))

                smooth_scores_all.append(s_smooth)
                mse_T_all.append(float(np.mean((T_c - gt_T) ** 2)))

        if batch_end % (BATCH_SIZE * 10) == 0 or batch_end == n_stations:
            elapsed = time.time() - t0
            rate = batch_end / elapsed
            eta = (n_stations - batch_end) / rate if rate > 0 else 0
            print(f"  {batch_end}/{n_stations} tram ({elapsed:.1f}s, ETA {eta:.0f}s)")

    pw_scores_all = np.array(pw_scores_all)
    mse_P_all = np.array(mse_P_all)
    mse_V_all = np.array(mse_V_all)
    smooth_scores_all = np.array(smooth_scores_all)
    mse_T_all = np.array(mse_T_all)

    def pearson(a, b):
        a_c, b_c = a - a.mean(), b - b.mean()
        return float(np.sum(a_c * b_c) / (np.sqrt(np.sum(a_c**2) * np.sum(b_c**2)) + 1e-12))

    corr_pw_mseP = pearson(pw_scores_all, mse_P_all)
    corr_pw_mseV = pearson(pw_scores_all, mse_V_all)
    corr_smooth_mseT = pearson(smooth_scores_all, mse_T_all)

    print("\n===== TUONG QUAN (Pearson) GIUA DIEM VERIFIER VA MSE THAT (n=%d cap) =====" % len(pw_scores_all))
    print(f"corr(L_pw score, MSE_SLP)       = {corr_pw_mseP:.4f}")
    print(f"corr(L_pw score, MSE_WindSpeed) = {corr_pw_mseV:.4f}")
    print(f"corr(L_smooth score, MSE_Temp)  = {corr_smooth_mseT:.4f}")
    print("\n(Tuong quan DUONG va MANH la tot: diem verifier thap (tot) phai di cung MSE thap (chinh xac))")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "n_pairs": len(pw_scores_all),
            "corr_pw_vs_mse_slp": corr_pw_mseP,
            "corr_pw_vs_mse_windspeed": corr_pw_mseV,
            "corr_smooth_vs_mse_temp": corr_smooth_mseT,
        }, f, indent=2)
    print(f"\nDa luu -> {OUT_JSON}")
    print(f"Total time: {time.time()-t0:.1f}s")
