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
FULL BASELINE SUITE tren WEATHER-5K (5,672 tram), dung L_pw (pressure-wind
consistency) LAM VERIFIER CHINH - KHONG CON DUNG L_smooth nua (da verify
bang do truc tiep correlation(score, MSE that) tren 113,440 cap candidate:
corr(L_pw, MSE_SLP)=0.310, corr(L_smooth, MSE_Temp)=0.012 - L_pw mang tin
hieu that, L_smooth gan nhu khong).

L_pw can CA P (SLP) va V (Wind Speed) CUNG LUC (khong tach rieng duoc nhu
L_smooth) -> Best-of-N+Physics CHI ap dung duoc cho cap (SLP, WindSpeed),
chon CHUNG 1 candidate index cho ca 2 bien nay. Temp/Dewpoint KHONG CO
verifier nao ap dung duoc trong ban nay - chi bao cao Zero-shot/Mean-of-N
cho 2 bien do (Best-of-N+Physics/Random search = N/A, khong co scorer).
"""

ALL_VARS = ["TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP"]
NO_VERIFIER_VARS = {"Temp": "TMP", "Dewpoint": "DEW"}      # chi Zero-shot/Mean-of-N
PW_VARS = {"WindSpeed": "WND_RATE", "SLP": "SLP"}           # du bang 4 baseline, dung L_pw

HISTORY_STEPS = 48
PREDICTION_STEPS = 24
N_SAMPLES = 20
START_IDX = 0
MODEL_ID = "amazon/chronos-t5-base"
BATCH_SIZE = 64
SEED = 0
OUT_JSON = "/kaggle/working/weather5k_lpw_baselines_results.json"
OUT_CSV_SUMMARY = "/kaggle/working/weather5k_lpw_baselines_summary.csv"


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

    # alpha cho tat ca tram, fit tu val-2022 (vector hoa)
    val_P = val_npz["data"][:, :, var_idx["SLP"]]
    val_V = val_npz["data"][:, :, var_idx["WND_RATE"]]
    dP_all = val_P[:, 1:] - val_P[:, :-1]
    dV_all = val_V[:, 1:] - val_V[:, :-1]
    alpha_all = np.sum(dP_all * dV_all, axis=1) / (np.sum(dV_all * dV_all, axis=1) + 1e-8)

    h_start, h_end = START_IDX, START_IDX + HISTORY_STEPS
    g_end = h_end + PREDICTION_STEPS

    # results["Temp"/"Dewpoint"]["zero_shot"/"mean_of_n"]  (khong co random_search/best_of_n_physics)
    # results["WindSpeed"/"SLP"][4 baseline day du]
    results = {}
    for label in NO_VERIFIER_VARS:
        results[label] = {"zero_shot": [], "mean_of_n": []}
    for label in PW_VARS:
        results[label] = {"zero_shot": [], "random_search": [], "mean_of_n": [], "best_of_n_physics": []}

    t0 = time.time()
    for batch_start in range(0, n_stations, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_stations)
        b = batch_end - batch_start

        # ---- Temp, Dewpoint: chi Zero-shot / Mean-of-N, khong verifier ----
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

        # ---- SLP, WindSpeed: dung chung 1 index candidate, chon theo L_pw ----
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
            CP, CV = cand_P[i], cand_V[i]  # (N_SAMPLES, PREDICTION_STEPS) moi cai

            zs_P, zs_V = np.median(CP, axis=0), np.median(CV, axis=0)
            mean_P, mean_V = CP.mean(axis=0), CV.mean(axis=0)

            sigma_P = float(np.std(CP, axis=0).mean())
            sigma_V = float(np.std(CV, axis=0).mean())
            noise_P = zs_P[None, :] + rng.normal(0, sigma_P, size=(N_SAMPLES, PREDICTION_STEPS))
            noise_V = zs_V[None, :] + rng.normal(0, sigma_V, size=(N_SAMPLES, PREDICTION_STEPS))
            noise_scores = np.array([pw_score(noise_P[j], noise_V[j], alpha) for j in range(N_SAMPLES)])
            best_noise_idx = int(np.argmin(noise_scores))

            real_scores = np.array([pw_score(CP[j], CV[j], alpha) for j in range(N_SAMPLES)])
            best_real_idx = int(np.argmin(real_scores))

            results["SLP"]["zero_shot"].append(float(np.mean((zs_P - gt_P) ** 2)))
            results["SLP"]["mean_of_n"].append(float(np.mean((mean_P - gt_P) ** 2)))
            results["SLP"]["random_search"].append(float(np.mean((noise_P[best_noise_idx] - gt_P) ** 2)))
            results["SLP"]["best_of_n_physics"].append(float(np.mean((CP[best_real_idx] - gt_P) ** 2)))

            results["WindSpeed"]["zero_shot"].append(float(np.mean((zs_V - gt_V) ** 2)))
            results["WindSpeed"]["mean_of_n"].append(float(np.mean((mean_V - gt_V) ** 2)))
            results["WindSpeed"]["random_search"].append(float(np.mean((noise_V[best_noise_idx] - gt_V) ** 2)))
            results["WindSpeed"]["best_of_n_physics"].append(float(np.mean((CV[best_real_idx] - gt_V) ** 2)))

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
                "verifier": "L_pw only (pressure-wind consistency); L_smooth KHONG con dung",
            },
            "mse_per_station": results,
        }, f)
    print(f"\nDa luu ket qua chi tiet -> {OUT_JSON}")

    print("\n===== TONG KET (trung binh MSE qua 5,672 tram) =====")
    all_labels = list(NO_VERIFIER_VARS) + list(PW_VARS)
    all_baselines = ["zero_shot", "random_search", "mean_of_n", "best_of_n_physics"]
    rows = []
    for baseline in all_baselines:
        row = {"baseline": baseline, "N": 1 if baseline == "zero_shot" else N_SAMPLES}
        for label in all_labels:
            if baseline in results[label]:
                row[f"{label}_MSE"] = float(np.mean(results[label][baseline]))
            else:
                row[f"{label}_MSE"] = None  # N/A - khong co verifier cho bien nay
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
