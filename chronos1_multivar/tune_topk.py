import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "architectures"))

from common import build_config, append_csv_row, SETTINGS, N_FINE_DEFAULT, N_COARSE_DEFAULT, RESULTS_CSV_FIELDS  # noqa: E402

from architectures.trend_physics import TrendPhysicsEvaluator  # noqa: E402
from architectures.seasonal_physics import SeasonalPhysicsEvaluator  # noqa: E402
from architectures.trend_seasonal_physics import TrendSeasonalPhysicsEvaluator  # noqa: E402
from architectures.cluster_physics import ClusterPhysicsEvaluator  # noqa: E402
from architectures.cluster_spatial_physics import ClusterSpatialPhysicsEvaluator  # noqa: E402


# ============================================================
# Tune K (top-k sau Physics Filter nhanh Fine) trong {3,5,7,9}, giu N=20
# CO DINH (n_fine=15, n_coarse=5, dung quy uoc 3:1 co san trong common.py)
# - chi 5 kien truc co tham so k_fine moi ap dung (Best-of-N+Physics
# luon lay top-1 sau filter, khong co K nen KHONG dua vao day).
#
# Quy trinh (dung tach biet val/test, khong ro ri thong tin tu test):
#   1. Voi moi (kien truc, setting): chay het 4 gia tri K tren VAL,
#      ghi tung diem vao results_k_sweep_val.csv.
#   2. Chon K* = K co MSE val nho nhat.
#   3. Chay DUNG 1 LAN voi K* tren TEST, ghi vao results_k_best_test.csv.
# ============================================================

K_GRID = [3, 5, 7, 9]

ARCHITECTURES_WITH_K = [
    ("trend_physics", TrendPhysicsEvaluator, {}),
    ("seasonal_physics", SeasonalPhysicsEvaluator, {}),
    ("trend_seasonal_physics", TrendSeasonalPhysicsEvaluator, {}),
    ("cluster_physics", ClusterPhysicsEvaluator, {}),
    ("cluster_spatial_physics", ClusterSpatialPhysicsEvaluator, {}),
]

K_SWEEP_CSV_FIELDS = RESULTS_CSV_FIELDS + ["k_fine", "split"]

DEFAULT_CLIMATOLOGY_PATH = os.path.join(os.path.dirname(__file__), "..", "era5_val_2015_2016.nc")


def parse_args():
    p = argparse.ArgumentParser(
        description="Tune K in {3,5,7,9} tren VAL (N=20 co dinh: n_fine=15/n_coarse=5), "
                     "chon K* roi xac nhan 1 lan tren TEST - chi ap dung cho kien truc co tham so k_fine."
    )
    p.add_argument("--val-path", type=str, required=True)
    p.add_argument("--test-path", type=str, required=True)
    p.add_argument("--climatology-path", type=str, default=DEFAULT_CLIMATOLOGY_PATH,
                    help="Du lieu dung tinh climatology cho ACC (mac dinh: tap val, dung chung ca khi dang danh gia test).")
    p.add_argument("--model-id", type=str, default="amazon/chronos-t5-base")
    p.add_argument("--n-windows-val", type=int, default=10)
    p.add_argument("--n-windows-test", type=int, default=10)
    p.add_argument("--n-fine", type=int, default=N_FINE_DEFAULT)
    p.add_argument("--n-coarse", type=int, default=N_COARSE_DEFAULT)
    p.add_argument("--settings", type=str, nargs="*", default=None,
                    help="Loc setting can chay (vd h512p64) - mac dinh chay het 3 setting.")
    p.add_argument("--architectures", type=str, nargs="*", default=None,
                    help="Loc kien truc can chay - mac dinh chay het 5 kien truc co tham so k_fine.")
    p.add_argument("--val-report-path", type=str, default=os.path.join(
        os.path.dirname(__file__), "reports", "results_k_sweep_val.csv"))
    p.add_argument("--test-report-path", type=str, default=os.path.join(
        os.path.dirname(__file__), "reports", "results_k_best_test.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    settings = [s for s in SETTINGS if args.settings is None or s[0] in args.settings]
    architectures = [a for a in ARCHITECTURES_WITH_K if args.architectures is None or a[0] in args.architectures]

    for setting_name, history_steps, prediction_steps in settings:
        val_config = build_config(history_steps, prediction_steps, args.model_id, args.val_path)

        for arch_name, EvaluatorCls, extra_kwargs in architectures:
            print("\n" + "#" * 70)
            print(f"# TUNE K tren VAL: {arch_name} @ {setting_name} (N=20 co dinh: n_fine={args.n_fine}, n_coarse={args.n_coarse})")
            print("#" * 70)

            val_mse = {}
            for k in K_GRID:
                evaluator = EvaluatorCls(
                    val_config, n_fine=args.n_fine, n_coarse=args.n_coarse, k_fine=k,
                    climatology_path=args.climatology_path, **extra_kwargs,
                )
                metrics = evaluator.run_test_set(n_windows=args.n_windows_val)
                val_mse[k] = metrics["multivar_mse_norm_mean"]

                append_csv_row(args.val_report_path, {
                    "architecture": arch_name,
                    "setting": setting_name,
                    "history_steps": history_steps,
                    "prediction_steps": prediction_steps,
                    "n_windows": args.n_windows_val,
                    "n_fine": args.n_fine,
                    "n_coarse": args.n_coarse,
                    "k_fine": k,
                    "split": "val",
                    **metrics,
                }, fieldnames=K_SWEEP_CSV_FIELDS)

            k_best = min(val_mse, key=val_mse.get)
            print(f"K* = {k_best} (MSE val = {val_mse[k_best]:.6f}) -- toan bo K da thu: {val_mse}")

            print("\n" + "#" * 70)
            print(f"# XAC NHAN TREN TEST voi K*={k_best}: {arch_name} @ {setting_name}")
            print("#" * 70)

            test_config = build_config(history_steps, prediction_steps, args.model_id, args.test_path)
            evaluator = EvaluatorCls(
                test_config, n_fine=args.n_fine, n_coarse=args.n_coarse, k_fine=k_best,
                climatology_path=args.climatology_path, **extra_kwargs,
            )
            test_metrics = evaluator.run_test_set(n_windows=args.n_windows_test)

            append_csv_row(args.test_report_path, {
                "architecture": arch_name,
                "setting": setting_name,
                "history_steps": history_steps,
                "prediction_steps": prediction_steps,
                "n_windows": args.n_windows_test,
                "n_fine": args.n_fine,
                "n_coarse": args.n_coarse,
                "k_fine": k_best,
                "split": "test",
                **test_metrics,
            }, fieldnames=K_SWEEP_CSV_FIELDS)

    print(f"\nHoan tat. Sweep VAL: {args.val_report_path} | Xac nhan TEST: {args.test_report_path}")
