import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from loader import load_npz_split, get_station_series, get_npz_window  # noqa: E402
from physics_verifier import StationPhysicsVerifier  # noqa: E402

"""
Pipeline Chronos (Best-of-N) + physics verifier (L_pw + L_smooth) tren
WEATHER-5K, theo dung tinh than test-time computing: Chronos (frozen,
zero-shot) sinh N candidate; physics verifier (KHONG train, fit alpha tu
lich su that) chon ra candidate hop ly nhat.

CANH BAO: script nay can GPU de chay trong thoi gian hop ly (Chronos
autoregressive, N candidate x prediction_steps buoc). KHONG chay local tren
may khong co GPU - dong goi chay tren Kaggle. Script tu in canh bao neu phat
hien dang chay tren CPU.

Du lieu: doc tu cache .npz (val_2022.npz, test_2023.npz) tao boi
build_val_test_cache.py - NHANH hon doc lai CSV moi lan. alpha duoc fit tu
val-2022 CUA CHINH TRAM DANG XET, forecast chay tren test-2023.
"""

VARS_FOR_VERIFIER = ["TMP", "WND_RATE", "SLP"]  # T, V, P can cho L_pw + L_smooth


def generate_chronos_candidates(pipeline, history_1d: np.ndarray, prediction_steps: int,
                                 n_samples: int) -> np.ndarray:
    """
    Sinh N candidate cho 1 chuoi 1 chieu bang Chronos (sampler autoregressive
    THAT, moi candidate la 1 duong doc lap - khac TimesFM quantile-head).
    Tra ve: (n_samples, prediction_steps), da denormalize ve don vi goc.
    """
    mean = float(np.mean(history_1d))
    std = float(np.std(history_1d))
    std = std if std >= 1e-5 else 1.0
    normalized = (history_1d - mean) / std

    context = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
    with torch.inference_mode():
        forecast = pipeline.predict(
            context,
            prediction_length=prediction_steps,
            num_samples=n_samples,
            top_k=50,
            top_p=1.0,
            temperature=1.0,
            limit_prediction_length=False,
        )
    forecast_norm = forecast.cpu().numpy()[0]  # (n_samples, prediction_steps)
    return forecast_norm * std + mean


def parse_args():
    p = argparse.ArgumentParser(
        description="Chronos Best-of-N + physics verifier (L_pw + L_smooth) tren WEATHER-5K."
    )
    p.add_argument("--val-npz", type=str, required=True, help="Duong dan toi val_2022.npz")
    p.add_argument("--test-npz", type=str, required=True, help="Duong dan toi test_2023.npz")
    p.add_argument("--station-index", type=int, default=0,
                   help="Chi so tram trong mang data (0..5671).")
    p.add_argument("--history-steps", type=int, default=48)
    p.add_argument("--prediction-steps", type=int, default=24)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--start-idx", type=int, default=0,
                   help="Vi tri bat dau cua so trong tap TEST (2023).")
    p.add_argument("--model-id", type=str, default="amazon/chronos-t5-base")
    p.add_argument("--lambda-pw", type=float, default=1.0)
    p.add_argument("--lambda-smooth", type=float, default=1.0)
    p.add_argument("--out-json", type=str, default=None,
                   help="Neu co, luu ket qua (score, MSE) ra file JSON nay.")
    return p.parse_args()


if __name__ == "__main__":
    from chronos import ChronosPipeline

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("!" * 70)
        print("[CANH BAO] Khong phat hien GPU. KHONG nen chay script nay tren may local")
        print("(xem memory feedback_no_gpu_workload_local) - hay dong goi chay tren Kaggle GPU T4.")
        print("!" * 70)

    print(f"Dang nap cache: {args.val_npz} , {args.test_npz}")
    val_npz = load_npz_split(args.val_npz)
    test_npz = load_npz_split(args.test_npz)

    station_id = str(val_npz["station_ids"][args.station_index])
    print(f"Tram #{args.station_index}: {station_id} "
          f"(lat={val_npz['latitude'][args.station_index]:.3f}, "
          f"lon={val_npz['longitude'][args.station_index]:.3f})")

    history_P_val = get_station_series(val_npz, args.station_index, "SLP")
    history_V_val = get_station_series(val_npz, args.station_index, "WND_RATE")

    verifier = StationPhysicsVerifier(dt=1.0)
    alpha = verifier.fit_alpha(history_P_val, history_V_val)
    print(f"alpha (pressure-wind sensitivity, fit tu val-2022 cua tram nay): {alpha:.6f}")

    window = get_npz_window(test_npz, args.station_index, args.start_idx,
                             args.history_steps, args.prediction_steps)

    print(f"Dang nap Chronos ({args.model_id}) tren {device}...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = ChronosPipeline.from_pretrained(args.model_id, device_map=device, dtype=dtype)
    print("Chronos da nap xong.")

    candidates = {}
    for var in VARS_FOR_VERIFIER:
        candidates[var] = generate_chronos_candidates(
            pipeline, window["history"][var], args.prediction_steps, args.n_samples
        )
        print(f"  candidates[{var}] shape: {candidates[var].shape}")

    top_idx, details = verifier.select_top_k(
        candidates["TMP"], candidates["WND_RATE"], candidates["SLP"],
        alpha, k=1, lambda_pw=args.lambda_pw, lambda_smooth=args.lambda_smooth,
    )
    best_idx = int(top_idx[0])
    print(f"\nCandidate duoc physics-verifier chon: #{best_idx}")
    print(f"  L_pw={details[best_idx]['L_pw']:.6f}  L_smooth={details[best_idx]['L_smooth']:.6f}  "
          f"composite={details[best_idx]['composite_score']:.6f}")

    # Doi chieu nhanh voi ground truth (TMP) - CHI de tham khao, khong phai
    # metric chinh thuc (chua lat-weight/ACC nhu pipeline ERA5).
    gt_tmp = window["future"]["TMP"]
    mse_per_candidate = np.mean((candidates["TMP"] - gt_tmp[None, :]) ** 2, axis=1)
    mse_best = float(mse_per_candidate[best_idx])
    mse_mean_all = float(mse_per_candidate.mean())
    mse_oracle_best = float(mse_per_candidate.min())
    print(f"\nMSE(TMP) candidate duoc verifier chon : {mse_best:.6f}")
    print(f"MSE(TMP) trung binh ca {args.n_samples} candidate : {mse_mean_all:.6f}")
    print(f"MSE(TMP) oracle (candidate tot nhat thuc su)   : {mse_oracle_best:.6f}")

    if args.out_json:
        import json

        result = {
            "station_id": station_id,
            "station_index": args.station_index,
            "start_idx": args.start_idx,
            "alpha": alpha,
            "best_idx": best_idx,
            "best_candidate_score": details[best_idx],
            "mse_tmp_selected": mse_best,
            "mse_tmp_mean_all": mse_mean_all,
            "mse_tmp_oracle_best": mse_oracle_best,
            "n_samples": args.n_samples,
            "history_steps": args.history_steps,
            "prediction_steps": args.prediction_steps,
            "model_id": args.model_id,
        }
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nDa luu ket qua -> {args.out_json}")
