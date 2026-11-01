import argparse
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from proposers import BACKBONE_REGISTRY, get_proposer, resolve_model_id  # noqa: E402


# ============================================================
# TEST co che "1 lich su -> N tuong lai" cua TUNG backbone, bang DU LIEU GIA
# (synthetic) + luoi nho + model nho -> chay nhanh tren CPU, KHONG can ERA5
# that, KHONG can GPU. Muc dich DUY NHAT: kiem tra N candidate sinh ra co
# THAT SU KHAC NHAU hay khong (sampling that vs quantile/tat dinh), truoc khi
# dem len chay du lieu lon.
#
# KY VONG (da phan tich):
#   chronos  -> N candidate KHAC NHAU (autoregressive sampling)   -> diverse
#   moirai1  -> N candidate KHAC NHAU (mixture distribution)      -> diverse
#   moirai2  -> N candidate TRUNG/GAN TRUNG (quantile)            -> deterministic
#   timesfm  -> N candidate TRUNG NHAU (point/quantile, tat dinh) -> deterministic
#   timesfm3 -> can dien loader (VERIFY_V3 trong proposers.py)
#
# Model dung de test (nho, cho nhanh) - sua neu can:
TEST_MODEL_IDS = {
    "chronos":  "amazon/chronos-t5-tiny",          # ~8M, tai nhanh
    "moirai1":  "Salesforce/moirai-1.1-R-small",
    "moirai2":  "Salesforce/moirai-2.0-R-small",
    "timesfm":  "google/timesfm-2.5-200m-pytorch",  # API moi (verify); doi sang 2.0-500m de test API cu
    "timesfm3": "google/timesfm-3.0-pytorch",
}
# ============================================================


def make_synthetic_history(history_steps, lat, lon, seed=0):
    """
    Lich su gia: trend tuyen tinh nhe + mua vu (chu ky ~ ngay/dem) + nhieu,
    khac nhau theo tung node -> giong tin hieu khi tuong that du de model
    sinh forecast co nghia.
    Shape tra ve: (history_steps, lat, lon)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(history_steps)
    hist = np.zeros((history_steps, lat, lon), dtype=np.float32)
    for i in range(lat):
        for j in range(lon):
            base = 280.0 + 5.0 * (i * lon + j)             # muc nen khac nhau moi node (kieu Kelvin)
            trend = 0.01 * t
            season = 3.0 * np.sin(2 * np.pi * t / 4.0)     # chu ky 4 buoc (vd 1 ngay = 4x6h)
            noise = rng.normal(0, 0.5, size=history_steps)
            hist[:, i, j] = base + trend + season + noise
    return hist


def diversity_report(denorm, node=(0, 0)):
    """
    denorm: (n_samples, prediction_steps, lat, lon)
    Tra ve dict mo ta do da dang giua cac sample tai 1 node.
    """
    n_samples = denorm.shape[0]
    i, j = node
    series = denorm[:, :, i, j]                            # (n_samples, prediction_steps)

    # std giua cac sample tai moi buoc, roi trung binh theo horizon
    per_step_std = series.std(axis=0)                      # (prediction_steps,)
    mean_std = float(per_step_std.mean())

    # kiem tra co 2 sample nao khac nhau khong (tolerance nho)
    all_identical = True
    for k in range(1, n_samples):
        if not np.allclose(series[0], series[k], atol=1e-5):
            all_identical = False
            break

    return {
        "n_samples": n_samples,
        "mean_std_across_samples": mean_std,
        "all_identical": all_identical,
        "series": series,
    }


def test_backbone(backbone, model_id, args):
    meta = BACKBONE_REGISTRY[backbone]
    print("\n" + "=" * 72)
    print(f"BACKBONE: {backbone}  (sampler={meta['sampler']}, supports_bon={meta['supports_bon']})")
    print(f"  model_id: {model_id}")
    print("=" * 72)

    history = make_synthetic_history(args.history_steps, args.lat, args.lon, seed=args.seed)
    print(f"  synthetic history shape: {history.shape}  (history_steps, lat, lon)")

    try:
        proposer = get_proposer(
            backbone, model_id=model_id, device=args.device,
            history_steps=args.history_steps, prediction_steps=args.prediction_steps,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] khong khoi tao duoc proposer: {type(e).__name__}: {e}")
        if args.verbose:
            traceback.print_exc()
        return None

    try:
        denorm, norm, mean, std = proposer.predict(
            history, prediction_steps=args.prediction_steps,
            temperature=args.temperature, n_samples=args.n_samples,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] predict() loi: {type(e).__name__}: {e}")
        if args.verbose:
            traceback.print_exc()
        return None

    print(f"  predict(pred={args.prediction_steps}, n_samples={args.n_samples}, temp={args.temperature}) ->")
    print(f"    denorm shape: {denorm.shape}   norm shape: {norm.shape}")
    print(f"    mean shape:   {mean.shape}   std shape:  {std.shape}")

    rep = diversity_report(denorm, node=(0, 0))
    print(f"  --- Do da dang giua {rep['n_samples']} candidate (tai node (0,0)) ---")
    print(f"    std trung binh giua cac sample (theo horizon): {rep['mean_std_across_samples']:.6f}")
    print(f"    tat ca candidate GIONG HET nhau? {'CO' if rep['all_identical'] else 'KHONG'}")
    for k in range(min(3, rep["n_samples"])):
        head = np.array2string(rep["series"][k][:6], precision=3, floatmode="fixed")
        print(f"      candidate[{k}][:6] = {head}")

    # Verdict + doi chieu voi registry
    is_diverse = (not rep["all_identical"]) and (rep["mean_std_across_samples"] > 1e-4)
    if is_diverse:
        verdict = "SAMPLING THAT (N tuong lai da dang) -> phu hop Best-of-N"
    else:
        verdict = "TAT DINH (quantile/point, N candidate trung nhau) -> chi lam single-pass baseline"
    print(f"  VERDICT: {verdict}")

    match = (is_diverse == meta["supports_bon"])
    print(f"  Khop registry (supports_bon={meta['supports_bon']})? {'OK' if match else 'LECH - CAN XEM LAI'}")

    return {"backbone": backbone, "is_diverse": is_diverse, "meta": meta,
            "mean_std": rep["mean_std_across_samples"], "match": match}


def parse_args():
    p = argparse.ArgumentParser(description="Test co che sinh N candidate cua tung backbone bang du lieu gia (nho, CPU).")
    p.add_argument("--backbones", type=str, nargs="*", default=list(BACKBONE_REGISTRY),
                    help="Backbone can test (mac dinh: het). Vd: --backbones chronos moirai1")
    p.add_argument("--n-samples", type=int, default=8, help="So candidate sinh ra de kiem tra do da dang.")
    p.add_argument("--history-steps", type=int, default=64)
    p.add_argument("--prediction-steps", type=int, default=16)
    p.add_argument("--lat", type=int, default=2, help="So hang luoi gia (nho cho nhanh).")
    p.add_argument("--lon", type=int, default=2, help="So cot luoi gia (nho cho nhanh).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-id", type=str, default=None,
                    help="Override model_id (chi hop ly khi test 1 backbone). Bo trong -> dung TEST_MODEL_IDS.")
    p.add_argument("--verbose", action="store_true", help="In full traceback khi loi.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Grid gia: {args.lat}x{args.lon} = {args.lat * args.lon} series | "
          f"history={args.history_steps} | prediction={args.prediction_steps} | device={args.device}")

    results = []
    for backbone in args.backbones:
        if backbone not in BACKBONE_REGISTRY:
            print(f"\n[BO QUA] backbone khong hop le: {backbone}")
            continue
        if args.model_id and len(args.backbones) == 1:
            model_id = args.model_id
        else:
            model_id = TEST_MODEL_IDS.get(backbone, resolve_model_id(backbone))
        res = test_backbone(backbone, model_id, args)
        if res:
            results.append(res)

    # Bang tong ket
    print("\n" + "#" * 72)
    print("# TONG KET")
    print("#" * 72)
    print(f"{'backbone':12s} {'sampler':16s} {'std giua sample':>16s} {'diverse?':>9s} {'khop registry?':>15s}")
    for r in results:
        print(f"{r['backbone']:12s} {r['meta']['sampler']:16s} "
              f"{r['mean_std']:>16.6f} {('CO' if r['is_diverse'] else 'KHONG'):>9s} "
              f"{('OK' if r['match'] else 'LECH'):>15s}")
    if not results:
        print("(khong backbone nao chay duoc - kiem tra da cai chronos-forecasting/uni2ts/timesfm chua)")
