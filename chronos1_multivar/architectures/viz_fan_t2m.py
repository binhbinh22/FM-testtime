import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from run import ERA5DataLoader, ChronosProposer  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import build_config, get_window_with_wind_history  # noqa: E402

"""
Visualize 1 lich su t2m -> N tuong lai (fan chart) sinh boi Chronos.
Khong dung physics filter / best-of-N - chi de XEM hinh dang phan ky
cua N sample tu 1 history, truoc khi loc.
"""


def parse_args():
    p = argparse.ArgumentParser(description="Ve fan chart N tuong lai t2m tu 1 lich su (Chronos).")
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--data-path", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "..", "era5_test_2017_2018.nc"))
    p.add_argument("--model-id", type=str, default="amazon/chronos-t5-tiny")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--lat-idx", type=int, default=16)
    p.add_argument("--lon-idx", type=int, default=32)
    p.add_argument("--history-tail", type=int, default=96,
                   help="So buoc lich su ve ra truoc diem du bao (chi de do thi khong qua dai).")
    p.add_argument("--out", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "reports", "fan_chart_t2m.png"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args.history_steps, args.prediction_steps, args.model_id, args.data_path)

    loader = ERA5DataLoader(config)
    data = get_window_with_wind_history(loader, config, args.start_idx)

    forecaster = ChronosProposer(model_id=config.model_id, device="cpu")
    t2m_denorm, _t2m_norm, _mean, _std = forecaster.predict(
        data["history_t2m"], prediction_steps=args.prediction_steps,
        temperature=1.0, n_samples=args.n_samples,
    )

    lat_idx, lon_idx = args.lat_idx, args.lon_idx
    history_1d = data["history_t2m"][-args.history_tail:, lat_idx, lon_idx]
    gt_1d = data["gt_t2m"][:, lat_idx, lon_idx]
    futures_1d = t2m_denorm[:, :, lat_idx, lon_idx]  # (n_samples, prediction_steps)

    history_x = np.arange(-len(history_1d), 0)
    future_x = np.arange(0, args.prediction_steps)

    # Noi lien mach: ghep diem cuoi lich su lam diem dau cho moi duong tuong lai,
    # de matplotlib ve lien 1 net thay vi 2 doan tach roi tai x=0.
    connect_x = np.concatenate([history_x[-1:], future_x])
    last_history = history_1d[-1]

    plt.figure(figsize=(11, 5))
    plt.plot(history_x, history_1d, color="black", linewidth=1.8, label="Lich su (history)")
    for i in range(args.n_samples):
        plt.plot(connect_x, np.concatenate([[last_history], futures_1d[i]]),
                  color="tab:blue", alpha=0.25, linewidth=1.0)
    plt.plot(connect_x, np.concatenate([[last_history], futures_1d.mean(axis=0)]),
              color="tab:blue", linewidth=2.0, label=f"Trung binh {args.n_samples} tuong lai (sample)")
    plt.plot(connect_x, np.concatenate([[last_history], gt_1d]),
              color="tab:red", linewidth=1.8, linestyle="--", label="Ground truth")
    plt.axvline(0, color="gray", linestyle=":", linewidth=1.0)
    plt.xlabel("Buoc thoi gian (0 = thoi diem hien tai)")
    plt.ylabel("t2m (K)")
    plt.title(f"1 lich su -> {args.n_samples} tuong lai (Chronos, grid point lat={lat_idx},lon={lon_idx})")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Da luu hinh -> {args.out}")
