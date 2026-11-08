import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd

"""
Loader nhe cho WEATHER-5K (https://github.com/taohan10200/WEATHER-5K), KHONG
phu thuoc code TSLib goc cua repo do (bo het multiprocessing/shared_memory,
vi pipeline cua minh la zero-shot + verifier, khong train nen khong can toi uu
cho training loop lon).

Cau truc du lieu mong doi sau khi giai nen zip:
    <root_path>/global_weather_stations/*.csv   (5,672 file, moi file 1 tram)

Moi CSV co cac cot (Table 4, paper WEATHER-5K):
    DATE, LONGITUDE, LATITUDE, TMP, DEW, WND_ANGLE, WND_RATE, SLP, MASK, TIME_DIFF
Moi file chua DU 2014-2023 (khong chia san theo nam) - viec cat train/val/test
lam bang LOC THEO NGAY ngay trong loader nay.
"""

VARS = ["TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP"]

SPLIT_RANGES = {
    "train": ("2014-01-01", "2021-12-31 23:00:00"),
    "val": ("2022-01-01", "2022-12-31 23:00:00"),
    "test": ("2023-01-01", "2023-12-31 23:00:00"),
}


def list_station_files(root_path: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(root_path, "global_weather_stations", "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"Khong tim thay CSV nao duoi {root_path}/global_weather_stations/. "
            f"Kiem tra lai da giai nen dung cau truc WEATHER-5K chua."
        )
    return files


def load_station_split(csv_path: str, flag: str) -> Dict[str, np.ndarray]:
    """
    Doc 1 file CSV tram, LOC THEO NGAY dung doan flag ('train'/'val'/'test'),
    tra ve dict cac mang numpy da sap xep tang dan theo thoi gian.
    """
    if flag not in SPLIT_RANGES:
        raise ValueError(f"flag phai la 1 trong {list(SPLIT_RANGES)}, nhan duoc '{flag}'")
    start, end = SPLIT_RANGES[flag]

    df = pd.read_csv(csv_path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")
    df = df.loc[(df["DATE"] >= start) & (df["DATE"] <= end)]

    out: Dict[str, np.ndarray] = {"DATE": df["DATE"].values}
    for col in VARS:
        out[col] = df[col].values.astype(np.float32)
    out["LATITUDE"] = float(df["LATITUDE"].iloc[0]) if "LATITUDE" in df.columns and len(df) else None
    out["LONGITUDE"] = float(df["LONGITUDE"].iloc[0]) if "LONGITUDE" in df.columns and len(df) else None
    out["station_id"] = os.path.basename(csv_path)
    return out


def get_window(station_split: Dict[str, np.ndarray], start_idx: int,
                history_steps: int, prediction_steps: int) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Cat 1 cua so (history, future=ground-truth) tu du lieu 1 tram DA duoc
    load_station_split() loc san theo split (vd 'test' = nam 2023).
    """
    h_end = start_idx + history_steps
    g_end = h_end + prediction_steps

    history, future = {}, {}
    for col in VARS:
        history[col] = station_split[col][start_idx:h_end]
        future[col] = station_split[col][h_end:g_end]
    return {"history": history, "future": future}


# ============================================================
# Doc tu cache .npz (xem build_val_test_cache.py) - NHANH HON NHIEU
# so voi doc lai CSV moi lan, dung cho pipeline Chronos+verifier.
# ============================================================

def load_npz_split(npz_path: str) -> Dict[str, np.ndarray]:
    """
    npz_path: duong dan toi val_2022.npz hoac test_2023.npz
              (tao boi build_val_test_cache.py).
    Tra ve dict: data (n_stations, n_hours, len(VARS)), station_ids,
    latitude, longitude, variables (= VARS, thu tu chieu cuoi cua data).
    """
    d = np.load(npz_path, allow_pickle=True)
    return {
        "data": d["data"],
        "station_ids": d["station_ids"],
        "latitude": d["latitude"],
        "longitude": d["longitude"],
        "variables": [str(v) for v in d["variables"]],
    }


def get_station_series(npz_split: Dict[str, np.ndarray], station_index: int, var: str) -> np.ndarray:
    """Lay chuoi 1 chieu (n_hours,) cua 1 bien, 1 tram, tu du lieu da load_npz_split()."""
    var_idx = npz_split["variables"].index(var)
    return npz_split["data"][station_index, :, var_idx]


def get_npz_window(npz_test_split: Dict[str, np.ndarray], station_index: int, start_idx: int,
                    history_steps: int, prediction_steps: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Giong get_window() nhung doc tu npz_test_split (output cua load_npz_split)."""
    h_end = start_idx + history_steps
    g_end = h_end + prediction_steps

    history, future = {}, {}
    for col in VARS:
        series = get_station_series(npz_test_split, station_index, col)
        history[col] = series[start_idx:h_end]
        future[col] = series[h_end:g_end]
    return {"history": history, "future": future}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Dung thu: python loader.py <root_path_cua_WEATHER-5K_da_giai_nen>")
        sys.exit(0)

    root = sys.argv[1]
    files = list_station_files(root)
    print(f"Tim thay {len(files)} tram.")

    test_data = load_station_split(files[0], flag="test")
    print(f"Tram {test_data['station_id']}: {len(test_data['DATE'])} buoc trong tap test (2023), "
          f"lat={test_data['LATITUDE']}, lon={test_data['LONGITUDE']}")

    w = get_window(test_data, start_idx=0, history_steps=48, prediction_steps=24)
    print("history TMP (5 gia tri dau):", w["history"]["TMP"][:5])
    print("future  TMP (5 gia tri dau):", w["future"]["TMP"][:5])
