import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from loader import list_station_files, load_station_split, VARS  # noqa: E402

"""
Trich rieng val(2022) + test(2023) cho TOAN BO tram tu ban WEATHER-5K da
giai nen day du (~43GB, con 2014-2021 khong can vi pipeline zero-shot khong
train), nen lai thanh 2 file .npz gon nhe (~1-2GB) de dung lau dai, khong
phai giu ban raw 43GB mai.

Output:
    <out_dir>/val_2022.npz   - arrays shape (n_stations, 8760, 5)
    <out_dir>/test_2023.npz  - arrays shape (n_stations, 8760, 5)
    Moi file con co: station_ids (n_stations,), latitude (n_stations,), longitude (n_stations,)
    Thu tu bien trong chieu cuoi: TMP, DEW, WND_ANGLE, WND_RATE, SLP (dung VARS trong loader.py)
"""


def build_cache(root_path: str, out_dir: str, flag: str):
    files = list_station_files(root_path)
    n = len(files)
    print(f"[{flag}] Tim thay {n} tram, bat dau trich...")

    data = None
    station_ids = []
    latitudes = np.full(n, np.nan, dtype=np.float32)
    longitudes = np.full(n, np.nan, dtype=np.float32)

    t0 = time.time()
    expected_len = None
    for i, csv_path in enumerate(files):
        split = load_station_split(csv_path, flag=flag)
        arr = np.stack([split[v] for v in VARS], axis=-1)  # (time, 5)

        if expected_len is None:
            expected_len = arr.shape[0]
            data = np.full((n, expected_len, len(VARS)), np.nan, dtype=np.float32)

        if arr.shape[0] != expected_len:
            print(f"  [CANH BAO] tram {split['station_id']} co {arr.shape[0]} buoc "
                  f"(ky vong {expected_len}) - cat/dem cho khop.")
            m = min(arr.shape[0], expected_len)
            data[i, :m, :] = arr[:m]
        else:
            data[i] = arr

        station_ids.append(split["station_id"])
        latitudes[i] = split["LATITUDE"] if split["LATITUDE"] is not None else np.nan
        longitudes[i] = split["LONGITUDE"] if split["LONGITUDE"] is not None else np.nan

        if (i + 1) % 500 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            print(f"  [{flag}] {i + 1}/{n} tram ({elapsed:.1f}s)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "val_2022.npz" if flag == "val" else "test_2023.npz")
    np.savez_compressed(
        out_path,
        data=data,
        station_ids=np.array(station_ids),
        latitude=latitudes,
        longitude=longitudes,
        variables=np.array(VARS),
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[{flag}] Da luu -> {out_path} (shape={data.shape}, {size_mb:.1f} MB)")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Trich val-2022 + test-2023 cho toan bo tram WEATHER-5K, nen thanh .npz.")
    p.add_argument("--root-path", type=str, required=True,
                   help="Thu muc WEATHER-5K da giai nen day du (chua global_weather_stations/).")
    p.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "processed"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_cache(args.root_path, args.out_dir, flag="val")
    build_cache(args.root_path, args.out_dir, flag="test")
    print("\nXONG. Co the xoa thu muc raw/ (~43GB) de giai phong dia, chi giu lai processed/.")
