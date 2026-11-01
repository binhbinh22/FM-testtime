import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "architectures"))

from common import build_config, append_csv_row, N_BUDGET_TOTAL  # noqa: E402
from proposers import BACKBONE_REGISTRY, resolve_model_id  # noqa: E402

from architectures.greedy import GreedyMultivarEvaluator  # noqa: E402
from architectures.best_of_n_physics import BestOfNPhysicsMultivarEvaluator  # noqa: E402


# ============================================================
# Nghien cuu DA-BACKBONE cho Q3 (backbone-agnostic) cua paper.
#
#   - greedy (single-pass baseline): chay voi MOI backbone.
#   - best_of_n_physics (TTC-by-sampling): CHI chay voi backbone co sampler
#     that (supports_bon=True: chronos, moirai1). Backbone quantile bi bo
#     qua o nhanh BoN (in log ro rang).
#
# Ket qua ghi vao 1 CSV tong hop (cot `architecture` da gom ca ten backbone
# dang "<arch>__<backbone>").
#
# QUAN TRONG: file nay danh cho chay THAT tren GPU. Can cai dat:
#   pip install chronos-forecasting uni2ts gluonts timesfm
# va cung cap dung checkpoint cho tung backbone (xem BACKBONE_REGISTRY).
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(description="Chay greedy (moi backbone) + best_of_n (backbone sampler-that), ghi 1 CSV.")
    p.add_argument("--data-path", type=str, required=True, help="Tap TEST (vd era5_test_2017_2018.nc).")
    p.add_argument("--climatology-path", type=str, required=True, help="Tap VAL de tinh climatology cho ACC.")
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--setting", type=str, default="h512p64")
    p.add_argument("--n-windows", type=int, default=37, help="So cua so (37 = non-overlap phu het test h512p64).")
    p.add_argument("--n", type=int, default=N_BUDGET_TOTAL, help="Ngan sach N cho best_of_n_physics.")
    p.add_argument("--backbones", type=str, nargs="*", default=list(BACKBONE_REGISTRY),
                    help="Loc backbone can chay (mac dinh: het). Vd: --backbones chronos moirai1 timesfm")
    p.add_argument("--skip-greedy", action="store_true")
    p.add_argument("--skip-bon", action="store_true")
    p.add_argument("--report-path", type=str, default=os.path.join(
        os.path.dirname(__file__), "reports", "results_backbones.csv"))
    return p.parse_args()


def _row(arch_backbone, args, n_fine, n_coarse, metrics):
    return {
        "architecture": arch_backbone,
        "setting": args.setting,
        "history_steps": args.history_steps,
        "prediction_steps": args.prediction_steps,
        "n_windows": args.n_windows,
        "n_fine": n_fine,
        "n_coarse": n_coarse,
        **metrics,
    }


if __name__ == "__main__":
    args = parse_args()

    for backbone in args.backbones:
        if backbone not in BACKBONE_REGISTRY:
            print(f"[BO QUA] backbone khong hop le: {backbone}")
            continue
        meta = BACKBONE_REGISTRY[backbone]
        model_id = resolve_model_id(backbone, None)
        config = build_config(args.history_steps, args.prediction_steps, model_id, args.data_path)

        # -------- greedy / single-pass baseline (moi backbone) --------
        if not args.skip_greedy:
            print("\n" + "#" * 70)
            print(f"# greedy @ {backbone} ({model_id}) [{meta['sampler']}]")
            print("#" * 70)
            try:
                ev = GreedyMultivarEvaluator(config, backbone=backbone, climatology_path=args.climatology_path)
                metrics = ev.run_test_set(n_windows=args.n_windows)
                append_csv_row(args.report_path, _row(f"greedy__{backbone}", args, 0, 0, metrics))
            except Exception as e:  # noqa: BLE001
                print(f"[LOI greedy @ {backbone}] {type(e).__name__}: {e}")

        # -------- best_of_n_physics: CHI backbone sampler that --------
        if not args.skip_bon:
            if not meta.get("supports_bon", False):
                print(f"[BO QUA BoN] {backbone} la sampler='{meta['sampler']}' (quantile) -> khong chay Best-of-N.")
                continue
            print("\n" + "#" * 70)
            print(f"# best_of_n_physics @ {backbone} (N={args.n})")
            print("#" * 70)
            try:
                ev = BestOfNPhysicsMultivarEvaluator(config, n=args.n, backbone=backbone,
                                                     climatology_path=args.climatology_path)
                metrics = ev.run_test_set(n_windows=args.n_windows)
                append_csv_row(args.report_path, _row(f"best_of_n_physics__{backbone}", args, args.n, 0, metrics))
            except Exception as e:  # noqa: BLE001
                print(f"[LOI BoN @ {backbone}] {type(e).__name__}: {e}")

    print(f"\nHoan tat. Ket qua tong hop: {args.report_path}")
