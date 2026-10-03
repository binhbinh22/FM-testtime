import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "architectures"))

from common import build_config, append_csv_row, SETTINGS, N_FINE_DEFAULT, N_COARSE_DEFAULT, N_BUDGET_TOTAL  # noqa: E402

from architectures.greedy import GreedyMultivarEvaluator  # noqa: E402
from architectures.best_of_n_physics import BestOfNPhysicsMultivarEvaluator  # noqa: E402
from architectures.trend_physics import TrendPhysicsEvaluator  # noqa: E402
from architectures.seasonal_physics import SeasonalPhysicsEvaluator  # noqa: E402
from architectures.trend_seasonal_physics import TrendSeasonalPhysicsEvaluator  # noqa: E402
from architectures.cluster_physics import ClusterPhysicsEvaluator  # noqa: E402
from architectures.cluster_spatial_physics import ClusterSpatialPhysicsEvaluator  # noqa: E402


# ============================================================
# Chay het 7 kien truc x 3 setting (cung ngan sach N=20), ghi 1 CSV tong
# hop - KHONG goi subprocess, goi thang ham run_test_set() cua tung kien
# truc de bat loi ro rang + nhanh hon (model Chronos-1 chi load 1 lan/
# kien truc/setting, khong phai load lai moi lan goi subprocess).
#
# QUAN TRONG: file nay danh cho chay THAT tren server GPU (V=2048 that,
# N=20, 3 setting x 7 kien truc = 21 lan chay day du) - o moi truong
# local KHONG GPU, dung --model-id amazon/chronos-t5-tiny + --data-path
# nho de smoke-test wiring, KHONG dung de lay so that.
# ============================================================

ARCHITECTURES = [
    ("greedy", GreedyMultivarEvaluator, {}),
    ("best_of_n_physics", BestOfNPhysicsMultivarEvaluator, {"n": N_BUDGET_TOTAL}),
    ("trend_physics", TrendPhysicsEvaluator, {"n_fine": N_FINE_DEFAULT, "n_coarse": N_COARSE_DEFAULT}),
    ("seasonal_physics", SeasonalPhysicsEvaluator, {"n_fine": N_FINE_DEFAULT, "n_coarse": N_COARSE_DEFAULT}),
    ("trend_seasonal_physics", TrendSeasonalPhysicsEvaluator, {"n_fine": N_FINE_DEFAULT, "n_coarse": N_COARSE_DEFAULT}),
    ("cluster_physics", ClusterPhysicsEvaluator, {"n_fine": N_FINE_DEFAULT, "n_coarse": N_COARSE_DEFAULT}),
    ("cluster_spatial_physics", ClusterSpatialPhysicsEvaluator, {"n_fine": N_FINE_DEFAULT, "n_coarse": N_COARSE_DEFAULT}),
]


def parse_args():
    p = argparse.ArgumentParser(description="Chay het 7 kien truc x 3 setting, ghi 1 CSV tong hop.")
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--model-id", type=str, default="amazon/chronos-t5-base")
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--report-path", type=str, default=os.path.join(
        os.path.dirname(__file__), "reports", "results_all.csv"))
    p.add_argument("--settings", type=str, nargs="*", default=None,
                    help="Loc setting can chay (vd h512p64) - mac dinh chay het 3 setting.")
    p.add_argument("--architectures", type=str, nargs="*", default=None,
                    help="Loc kien truc can chay - mac dinh chay het 7 kien truc.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    settings = [s for s in SETTINGS if args.settings is None or s[0] in args.settings]
    architectures = [a for a in ARCHITECTURES if args.architectures is None or a[0] in args.architectures]

    for setting_name, history_steps, prediction_steps in settings:
        config = build_config(history_steps, prediction_steps, args.model_id, args.data_path)

        for arch_name, EvaluatorCls, extra_kwargs in architectures:
            print("\n" + "#" * 70)
            print(f"# {arch_name} @ {setting_name}")
            print("#" * 70)

            evaluator = EvaluatorCls(config, **extra_kwargs)
            metrics = evaluator.run_test_set(n_windows=args.n_windows)

            n_fine = extra_kwargs.get("n_fine", extra_kwargs.get("n", 0))
            n_coarse = extra_kwargs.get("n_coarse", 0)

            row = {
                "architecture": arch_name,
                "setting": setting_name,
                "history_steps": history_steps,
                "prediction_steps": prediction_steps,
                "n_windows": args.n_windows,
                "n_fine": n_fine,
                "n_coarse": n_coarse,
                **metrics,
            }
            append_csv_row(args.report_path, row)

    print(f"\nHoan tat. Ket qua tong hop: {args.report_path}")
