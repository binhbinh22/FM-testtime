import numpy as np
from typing import Optional


class PINTHarmonicOscillatorScorer:
    """
    Physics-informed regularizer theo tinh than PINT (Physics-Informed
    Neural Time-series) / PINN (Raissi et al., 2019): nhung mot phuong
    trinh vat ly da biet vao qua trinh xep hang / lua chon nghiem, thay vi
    hoc hoan toan tu du lieu. Day la mot ky thuat physics-informed KHAC
    (va doc lap) voi continuity equation da dung trong
    era5_physics_energy.py - dung de bo sung soft regularization rieng
    cho thanh phan chu ky (seasonal).

    PHUONG TRINH VAT LY (harmonic oscillator - climate seasonal cycle):

        u''(t) + omega^2 * u(t) = 0,          omega = 2*pi / T_period

    Voi T_period = 365 ngay (chu ky quay quanh Mat Troi cua Trai Dat),
    nghiem giai tich la sin/cos - dung dang dao dong tuan hoan cua nhiet
    do theo mua. Day la domain knowledge duoc nhung THANG vao scorer,
    khong can hoc.

    LUU Y QUAN TRONG:
    - Phuong trinh nay khong co hang so dich (offset) o ve phai, nen chi
      dung dung khi du lieu dau vao ĐA duoc chuan hoa (zero-mean) hoac it
      nhat dao dong quanh 0. Vi vay scorer nay duoc thiet ke de ap len
      thanh phan SEASONAL (S = raw - trend, thuong da dao dong quanh 0)
      thay vi ap truc tiep len chuoi nhiet do tho (co trend + muc nen
      khong tuan hoan).
    - dt_days PHAI cung don vi (ngay) voi period_days de omega * dt co y
      nghia vat ly dung. Voi du lieu ERA5 lay moi interval_hours gio:
          dt_days = interval_hours / 24.0
    - Day la SOFT constraint dung de xep hang (residual thap hon = khop
      voi dao dong mua tot hon), KHONG phai constraint cung.
    """

    def __init__(self, dt_days: float, period_days: float = 365.0):
        if dt_days <= 0:
            raise ValueError(f"dt_days phai > 0, nhan {dt_days}")
        if period_days <= 0:
            raise ValueError(f"period_days phai > 0, nhan {period_days}")
        self.dt_days = dt_days
        self.period_days = period_days
        self.omega = 2.0 * np.pi / period_days

    def second_derivative(self, u: np.ndarray) -> np.ndarray:
        """
        u''(t) qua sai phan trung tam bac 2:
            u''[i] = (u[i+1] - 2*u[i] + u[i-1]) / dt_days^2
        Input:  u shape (T, ...), truc 0 la thoi gian.
        Return: shape (T-2, ...) (bo 2 diem bien vi khong du 3 diem lan can).
        """
        u = np.asarray(u, dtype=np.float64)
        if u.shape[0] < 3:
            raise ValueError(
                "Can it nhat 3 buoc thoi gian de tinh dao ham bac 2 "
                f"(nhan T={u.shape[0]})."
            )
        return (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (self.dt_days ** 2)

    def residual_field(self, u: np.ndarray) -> np.ndarray:
        """
        Residual field day du cua u''(t) + omega^2 * u(t) = 0.
        Input: u shape (T, ...). Return: shape (T-2, ...)
        """
        d2u = self.second_derivative(u)
        u_align = np.asarray(u, dtype=np.float64)[1:-1]
        return d2u + (self.omega ** 2) * u_align

    def harmonic_residual_score(self, u: np.ndarray) -> float:
        """K_harmonic(u) dang scalar - MSE cua residual field."""
        residual = self.residual_field(u)
        return float(np.mean(residual ** 2))

    def score_batch(self, U_scenarios: np.ndarray) -> np.ndarray:
        """
        Tinh K_harmonic cho tung candidate trong batch.
        Input:  U_scenarios shape (n_scenarios, T, ...)
        Return: scores shape (n_scenarios,)
        """
        n = U_scenarios.shape[0]
        scores = np.empty(n, dtype=np.float64)
        for i in range(n):
            scores[i] = self.harmonic_residual_score(U_scenarios[i])
        return scores


if __name__ == "__main__":
    # Chuoi nam co chu ky dung 365 ngay -> residual phai gan 0.
    dt_days = 6.0 / 24.0  # ERA5 interval 6h
    period_days = 365.0
    n_steps = 400
    t_days = np.arange(n_steps) * dt_days

    omega = 2.0 * np.pi / period_days
    u_periodic = np.sin(omega * t_days)
    u_noise = np.random.RandomState(0).randn(n_steps)

    scorer = PINTHarmonicOscillatorScorer(dt_days=dt_days, period_days=period_days)

    k_periodic = scorer.harmonic_residual_score(u_periodic)
    k_noise = scorer.harmonic_residual_score(u_noise)

    print("=" * 60)
    print("PINT Harmonic Oscillator Scorer - sanity check")
    print("=" * 60)
    print(f"K_harmonic(sin dung chu ky 365 ngay) = {k_periodic:.6f}")
    print(f"K_harmonic(nhieu trang)              = {k_noise:.6f}")
    assert k_periodic < k_noise, "Chuoi khop dung chu ky phai co residual thap hon nhieu trang"
    print("OK - chuoi khop dung chu ky mua co residual thap hon han nhieu ngau nhien.")
