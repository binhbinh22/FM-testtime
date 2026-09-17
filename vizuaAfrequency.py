import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import dataclass
from typing import Tuple, Dict
from chronos import ChronosPipeline

# ============================================================
# CONFIG & DATA LOADER (GIỮ NGUYÊN)
# ============================================================
@dataclass
class ERA5Config:
    data_path: str = "/home/user18/binhnkt/era5_test_2017_2018.nc"
    history_steps: int = 512
    prediction_steps: int = 128
    inspect_lat: int = 16
    inspect_lon: int = 32
    seed: int = 42

class ERA5DataLoader:
    def __init__(self, config: ERA5Config):
        self.config = config
        self.ds = xr.open_dataset(config.data_path).transpose("time", "latitude", "longitude")

    def get_test_window(self, start_idx: int) -> Dict[str, np.ndarray]:
        h_end = start_idx + self.config.history_steps
        p_end = h_end + self.config.prediction_steps
        window = self.ds.isel(time=slice(start_idx, p_end))
        return {"history_t2m": window["2m_temperature"].values[:self.config.history_steps]}

# ============================================================
# HÀM VIZUALIZATION MỚI (FFT)
# ============================================================
def visualize_raw_and_fft(series_A, config):
    lat, lon = config.inspect_lat, config.inspect_lon
    
    # Ép kiểu sang numpy array ngay tại đây để tránh lỗi với numpy functions
    if isinstance(series_A, torch.Tensor):
        series_A = series_A.detach().cpu().numpy()
        
    point_A = series_A[:, lat, lon]
    time_steps = np.arange(len(point_A))

    # FFT Calculation
    point_A_centered = point_A - np.mean(point_A)
    fft_vals = np.fft.rfft(point_A_centered)
    amplitude = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(len(point_A))

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    
    # 1. Raw Time Domain
    axes[0].plot(time_steps, point_A, color="black", linewidth=1.5)
    axes[0].set_title(f"Chuỗi A gốc (Time Domain) tại điểm ({lat}, {lon})")
    axes[0].set_ylabel("Nhiệt độ (K)")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Frequency Domain
    axes[1].plot(freqs, amplitude, color="blue", linewidth=2)
    axes[1].set_title("Biên độ phổ (Frequency Domain) - Phân tích FFT")
    axes[1].set_xlabel("Tần số (Normalized Frequency)")
    axes[1].set_ylabel("Biên độ (Amplitude)")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("raw_vs_fft.png", dpi=200)
    plt.show()
    print("Đã lưu biểu đồ: raw_vs_fft.png")

# ============================================================
# MAIN THỰC THI
# ============================================================
if __name__ == "__main__":
    config = ERA5Config()
    loader = ERA5DataLoader(config)
    
    # Lấy dữ liệu A
    data = loader.get_test_window(start_idx=0)
    series_A = torch.tensor(data["history_t2m"], dtype=torch.float32)
    
    # Vẽ
    visualize_raw_and_fft(series_A, config)