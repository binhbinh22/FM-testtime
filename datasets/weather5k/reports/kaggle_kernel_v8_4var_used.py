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
FULL BASELINE SUITE tren WEATHER-5K (5,672 tram), 4 BIEN DOC LAP (khop dung
Table 2 cua paper PhysicsFormer: Temperature, Dewpoint, Wind Speed, Sea
Level Pressure - KHONG co Wind Direction vi ban than Table 2 goc cung
khong co bien nay).

Moi bien duoc Chronos sinh N=20 candidate RIENG (khong con ghep chung index
voi bien khac, vi L_pw da tat nen khong con ly do vat ly nao buoc phai giu
candidate dong bo giua cac bien). Moi bien TU CHON candidate cua chinh no
theo L_smooth cua chinh no.

Baseline cho moi bien:
  - Zero-shot        : median cua N candidate.
  - Random search     : median + nhieu Gaussian (sigma = std cua N candidate
                        that), cham bang L_smooth, chon diem cao nhat.
  - Mean-of-N         : trung binh cong N candidate that.
  - Best-of-N+Physics : cham N candidate THAT bang L_smooth (chuan hoa theo
                        phuong sai cua chinh cua so lich su 48h), chon diem
                        cao nhat. (= "Ours")
"""

TARGET_VARS = {
    "Temp": "TMP",
    "Dewpoint": "DEW",
    "WindSpeed": "WND_RATE",
    "SLP": "SLP",
}
ALL_VARS = ["TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP"]

HISTORY_STEPS = 48
PREDICTION_STEPS = 24
N_SAMPLES = 20
START_IDX = 0
MODEL_ID = "amazon/chronos-t5-base"
BATCH_SIZE = 64
SEED = 0
OUT_JSON = "/kaggle/working/weather5k_4var_baselines_results.json"
OUT_CSV_SUMMARY = "/kaggle/working/weather5k_4var_baselines_summary.csv"


def find_data_file(filename: str):
    for root, _dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    return None


def load_npz_split(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "data": d["data"],
        "station_ids": d["station_ids"],
        "latitude": d["latitude"],
        "longitude": d["longitude"],
        "variables": [str(v) for v in d["variables"]],
    }


def second_diff(x):
    return x[2:] - 2 * x[1:-1] + x[:-2]


def smoothness_score(X_c, hist_var, eps=1e-6):
    """L_smooth CHUAN HOA theo phuong sai cua chinh cua so lich su (48h), 1 bien duy nhat."""
    d2 = second_diff(X_c)
    return float(np.mean(d2 ** 2)) / (hist_var + eps)


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
    if val_path is None or test_path is None:
        tree = []
        for root, _dirs, files in os.walk("/kaggle/input"):
            tree.append(f"{root}: {files}")
        raise FileNotFoundError("Khong tim thay val_2022.npz/test_2023.npz. Tree:\n" + "\n".join(tree))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    val_npz = load_npz_split(val_path)
    test_npz = load_npz_split(test_path)
    n_stations = val_npz["data"].shape[0]
    print(f"So tram: {n_stations}")

    var_idx = {v: val_npz["variables"].index(v) for v in ALL_VARS}
    rng = np.random.default_rng(SEED)

    print(f"Dang nap Chronos ({MODEL_ID}) tren {device}...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = ChronosPipeline.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)
    print("Chronos da nap xong.")

    h_start, h_end = START_IDX, START_IDX + HISTORY_STEPS
    g_end = h_end + PREDICTION_STEPS

    # results[label_bien][baseline] = list MSE qua tung tram
    results = {label: {k: [] for k in ["zero_shot", "random_search", "mean_of_n", "best_of_n_physics"]}
               for label in TARGET_VARS}

    t0 = time.time()
    for batch_start in range(0, n_stations, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_stations)
        b = batch_end - batch_start

        for label, var_name in TARGET_VARS.items():
            hist_batch = test_npz["data"][batch_start:batch_end, h_start:h_end, var_idx[var_name]]
            future_batch = test_npz["data"][batch_start:batch_end, h_end:g_end, var_idx[var_name]]

            candidates = generate_chronos_candidates_batch(
                pipeline, hist_batch, PREDICTION_STEPS, N_SAMPLES
            )  # (b, N_SAMPLES, PREDICTION_STEPS)

            for i in range(b):
                gt = future_batch[i]
                C = candidates[i]
                hist_var = float(np.var(hist_batch[i]))

                zs = np.median(C, axis=0)
                mse_zs = float(np.mean((zs - gt) ** 2))

                mean_n = C.mean(axis=0)
                mse_mean = float(np.mean((mean_n - gt) ** 2))

                sigma = float(np.std(C, axis=0).mean())
                noise_candidates = zs[None, :] + rng.normal(0, sigma, size=(N_SAMPLES, PREDICTION_STEPS))
                noise_scores = np.array([smoothness_score(noise_candidates[j], hist_var) for j in range(N_SAMPLES)])
                mse_random_search = float(np.mean((noise_candidates[int(np.argmin(noise_scores))] - gt) ** 2))

                real_scores = np.array([smoothness_score(C[j], hist_var) for j in range(N_SAMPLES)])
                mse_bon_physics = float(np.mean((C[int(np.argmin(real_scores))] - gt) ** 2))

                results[label]["zero_shot"].append(mse_zs)
                results[label]["mean_of_n"].append(mse_mean)
                results[label]["random_search"].append(mse_random_search)
                results[label]["best_of_n_physics"].append(mse_bon_physics)

        if batch_end % (BATCH_SIZE * 10) == 0 or batch_end == n_stations:
            elapsed = time.time() - t0
            rate = batch_end / elapsed
            eta = (n_stations - batch_end) / rate if rate > 0 else 0
            print(f"  {batch_end}/{n_stations} tram ({elapsed:.1f}s, {rate:.2f} tram/s, ETA {eta:.0f}s)")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "config": {
                "history_steps": HISTORY_STEPS, "prediction_steps": PREDICTION_STEPS,
                "n_samples": N_SAMPLES, "model_id": MODEL_ID, "n_stations": n_stations,
                "target_vars": TARGET_VARS,
            },
            "mse_per_station": results,
        }, f)
    print(f"\nDa luu ket qua chi tiet -> {OUT_JSON}")

    print("\n===== TONG KET (trung binh MSE qua 5,672 tram, moi bien doc lap) =====")
    rows = []
    for baseline in ["zero_shot", "random_search", "mean_of_n", "best_of_n_physics"]:
        row = {"baseline": baseline, "N": 1 if baseline == "zero_shot" else N_SAMPLES}
        for label in TARGET_VARS:
            arr = np.array(results[label][baseline])
            row[f"{label}_MSE"] = float(arr.mean())
        rows.append(row)
        print(f"[{baseline}] " + " ".join(f"{label}={row[f'{label}_MSE']:.4f}" for label in TARGET_VARS))

    with open(OUT_CSV_SUMMARY, "w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
