import numpy as np
from typing import Tuple


def centered_moving_average_decompose(x: np.ndarray, kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phan ra Trend / Seasonal bang Centered Moving Average (CMA), dung quy
    uoc cua Autoformer / FEDformer / TimeMixer (moving-average trend
    extraction). Trien khai thuan numpy (khong phu thuoc torch) de dung
    chung duoc cho ca nhanh fine va nhanh coarse cua pipeline multiscale.

        Trend(t)    = CMA(x, kernel_size)(t)
        Seasonal(t) = x(t) - Trend(t)

    Padding: lap lai gia tri bien (tuong duong "replicate" padding trong
    ban torch cua vizua_A_trend_seasonal.py) de giu nguyen do dai chuoi
    sau khi lay trung binh truot centered:
        front_pad = kernel_size // 2
        end_pad   = kernel_size - front_pad - 1

    Input:
        x: shape (T, ...) - truc 0 LUON la truc thoi gian.
        kernel_size: do rong cua so trung binh truot (>= 1).

    Return:
        trend, seasonal: cung shape voi x (float64).
    """
    if kernel_size < 1:
        raise ValueError(f"kernel_size phai >= 1, nhan {kernel_size}")

    x = np.asarray(x, dtype=np.float64)

    if kernel_size == 1:
        return x.copy(), np.zeros_like(x)

    front_pad = kernel_size // 2
    end_pad = kernel_size - front_pad - 1

    front = np.repeat(x[:1], front_pad, axis=0) if front_pad > 0 else x[:0]
    end = np.repeat(x[-1:], end_pad, axis=0) if end_pad > 0 else x[:0]
    x_padded = np.concatenate([front, x, end], axis=0)

    # Cumulative-sum trick: trung binh truot O(T) thay vi O(T * kernel_size).
    cumsum = np.cumsum(x_padded, axis=0)
    cumsum = np.concatenate([np.zeros_like(cumsum[:1]), cumsum], axis=0)
    window_sum = cumsum[kernel_size:] - cumsum[:-kernel_size]
    trend = window_sum / kernel_size

    seasonal = x - trend
    return trend, seasonal


def kernel_size_for_scale(kernel_size_fine: int, factor: int) -> int:
    """
    Quy doi kernel_size tu scale fine sang scale coarse theo dung ti le
    downsample, nhat quan voi vizua_A_trend_seasonal.py
    (kernel_size_C = kernel_size_A // factor), dam bao ca 2 nhanh cung
    trich xuat "cung mot khoang thoi gian vat ly" lam Trend (vi du: cung
    la trung binh truot 1 ngay, chi khac so buoc roi rac).
    """
    if factor < 1:
        raise ValueError(f"factor phai >= 1, nhan {factor}")
    return max(2, kernel_size_fine // factor)


if __name__ == "__main__":
    np.random.seed(0)

    T, ny, nx = 24, 4, 4
    t = np.arange(T)
    # Trend tuyen tinh + seasonal hinh sin (chu ky 4 buoc) + nhieu nho
    trend_true = 0.05 * t
    seasonal_true = np.sin(2 * np.pi * t / 4.0)
    x = (trend_true + seasonal_true)[:, None, None] * np.ones((1, ny, nx))
    x += 0.01 * np.random.randn(T, ny, nx)

    trend, seasonal = centered_moving_average_decompose(x, kernel_size=4)

    print("=" * 60)
    print("Kiem tra centered_moving_average_decompose")
    print("=" * 60)
    print("x.shape      :", x.shape)
    print("trend.shape  :", trend.shape)
    print("seasonal.shape:", seasonal.shape)
    assert trend.shape == x.shape
    assert seasonal.shape == x.shape
    print("Reconstruct max abs error:", np.max(np.abs(trend + seasonal - x)))

    kc = kernel_size_for_scale(4, factor=2)
    print(f"kernel_size_for_scale(4, factor=2) = {kc}")
    assert kc == 2
    print("OK")
