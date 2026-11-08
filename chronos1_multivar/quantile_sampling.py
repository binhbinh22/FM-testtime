import numpy as np

"""
Algorithm 3 (Best-of-N Sample, DEFT arXiv:2607.19659) - LOI inverse-CDF
sampling cho cac model QUANTILE-HEAD (Chronos-2 / Moirai-2 / TimesFM-2.x/3).

Model quantile chi xuat gia tri tai 1 so muc quantile co dinh (vd 0.1..0.9)
o moi buoc thoi gian -> KHONG co mau i.i.d. da dang. Algorithm 3 tao do da
dang bang cach, voi moi candidate i va moi buoc t:
    u_{i,t} ~ U(0,1)                      (doc lap tung buoc)
    y_i[t]  = F^{-1}(u_{i,t} | t)         (noi suy inverse-CDF tu luoi quantile)
roi LUON them median (quantile 0.5) vao pool. Tong pool = N candidate
(N-1 sample + 1 median).

LUU Y (da canh bao voi user): sampling doc lap tung buoc tao duong RANG CUA
theo thoi gian; verifier dua tren sai phan (L_pw dung dP, dV) se nhay cam voi
dieu nay. Day la tinh chat cua Algorithm 3, khong phai bug.

File nay THUAN numpy, KHONG phu thuoc model/GPU -> test duoc doc lap.
"""


def inverse_cdf_interp(u: np.ndarray, q_levels: np.ndarray, q_values: np.ndarray) -> np.ndarray:
    """
    Noi suy tuyen tinh inverse-CDF (semantics giong np.interp, clamp o 2 duoi).

    u        : (...,)        so ngau nhien trong [0,1]
    q_levels : (n_q,)        cac muc quantile, TANG DAN trong (0,1) (vd 0.1..0.9)
    q_values : (..., n_q)    gia tri quantile, khop shape dau voi u, KHONG GIAM
                             theo truc cuoi (vi quantile tang theo level)

    return   : (...,)        gia tri inverse-CDF
    """
    q_levels = np.asarray(q_levels, dtype=np.float64)
    u_c = np.clip(u, q_levels[0], q_levels[-1])                 # clamp duoi -> lay quantile bien
    idx = np.searchsorted(q_levels, u_c, side="right") - 1      # bin trai
    idx = np.clip(idx, 0, len(q_levels) - 2)
    l0 = q_levels[idx]
    l1 = q_levels[idx + 1]
    v0 = np.take_along_axis(q_values, idx[..., None], axis=-1)[..., 0]
    v1 = np.take_along_axis(q_values, (idx + 1)[..., None], axis=-1)[..., 0]
    w = (u_c - l0) / (l1 - l0 + 1e-12)
    return v0 + w * (v1 - v0)


def algo3_candidates_from_quantiles(q_values: np.ndarray, q_levels, n_candidates: int,
                                    rng: np.random.Generator, median_level: float = 0.5) -> np.ndarray:
    """
    q_values : (n_series, pred, n_q)  gia tri quantile (don vi THAT/denorm)
    q_levels : (n_q,)                 muc quantile tuong ung cot cuoi
    n_candidates: tong pool N (= N-1 inverse-CDF sample + 1 median)

    return   : (n_candidates, pred, n_series)
    """
    q_values = np.asarray(q_values, dtype=np.float64)
    q_levels = np.asarray(q_levels, dtype=np.float64)
    n_series, pred, n_q = q_values.shape
    n_draw = max(0, n_candidates - 1)

    # median = inverse-CDF tai median_level (dong 11 Algorithm 3: luon them)
    median = inverse_cdf_interp(
        np.full((n_series, pred), float(median_level)), q_levels, q_values
    )  # (n_series, pred)

    if n_draw == 0:
        return np.transpose(median[None], (0, 2, 1))  # (1, pred, n_series)

    # N-1 inverse-CDF sample, u doc lap tung (candidate, series, buoc)
    u = rng.uniform(0.0, 1.0, size=(n_draw, n_series, pred))
    q_b = np.broadcast_to(q_values[None], (n_draw, n_series, pred, n_q))
    samples = inverse_cdf_interp(u, q_levels, q_b)              # (n_draw, n_series, pred)

    all_cand = np.concatenate([samples, median[None]], axis=0)  # (n_candidates, n_series, pred)
    return np.transpose(all_cand, (0, 2, 1))                    # (n_candidates, pred, n_series)


if __name__ == "__main__":
    # Test: dung quantile cua N(0,1) -> inverse-CDF sample phai co phan phoi ~ N(0,1)
    from scipy.stats import norm  # chi dung trong test

    rng = np.random.default_rng(0)
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # 1 series, 1 buoc, quantile cua N(0,1)
    qv = norm.ppf(levels).reshape(1, 1, len(levels))
    cand = algo3_candidates_from_quantiles(qv, levels, n_candidates=20000, rng=rng)  # (20000,1,1)
    vals = cand[:-1, 0, 0]  # bo median
    print("mean ~0 :", round(float(vals.mean()), 3), "| std ~0.57 (clamp 0.1-0.9):", round(float(vals.std()), 3))
    print("median candidate (cuoi) =", round(float(cand[-1, 0, 0]), 4), "(phai ~0)")
