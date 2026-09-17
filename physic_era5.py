import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HARMONIC OSCILLATOR FITTER
# u(t) = A*cos(w*t) + B*sin(w*t),  w = 2*pi / 365 (T = 1 nam)
# ============================================================

class HarmonicOscillatorForecaster:
    """
    Fit nghiem giai tich cua phuong trinh u'' + w^2 u = 0
    tu du lieu lich su, roi suy ra tuong lai.

    Khong can huan luyen mang neural nao - chi la least-squares
    tuyen tinh theo 2 tham so A, B.
    """

    def __init__(self, period_days: float = 365.0, interval_hours: float = 6.0):
        self.period_days = period_days
        self.omega = 2 * np.pi / period_days          # w = 2*pi / T
        self.interval_days = interval_hours / 24.0     # quy doi 1 buoc sang ngay
        self.A = None
        self.B = None

    def _step_to_days(self, step_indices: np.ndarray) -> np.ndarray:
        """Quy doi index buoc (0,1,2,...) sang thoi gian tinh theo ngay."""
        return step_indices * self.interval_days

    def fit(self, history: np.ndarray, t0_step: int = 0):
        """
        history: mang 1D, gia tri quan sat lich su (da chuan hoa, mean~0)
        t0_step: index buoc bat dau cua history (mac dinh = 0)

        Giai he: u_i = A*cos(w*t_i) + B*sin(w*t_i)  bang least-squares
        """
        n = len(history)
        step_indices = np.arange(t0_step, t0_step + n)
        t_days = self._step_to_days(step_indices)

        # Ma tran thiet ke Phi, kich thuoc (N, 2)
        Phi = np.stack([
            np.cos(self.omega * t_days),
            np.sin(self.omega * t_days)
        ], axis=1)

        # Giai least-squares: [A, B] = argmin || Phi @ [A,B] - u ||^2
        coeffs, residuals, rank, sv = np.linalg.lstsq(Phi, history, rcond=None)
        self.A, self.B = coeffs[0], coeffs[1]

        return self.A, self.B

    def predict(self, n_future_steps: int, t0_step: int) -> np.ndarray:
        """
        Suy ra tuong lai bang chinh cong thuc nghiem giai tich,
        khong can lap lai fit, khong can du lieu moi.

        t0_step: index buoc dau tien cua doan tuong lai can du bao
                 (vi du neu history dai 512 buoc bat dau tu step 0,
                  thi tuong lai bat dau tu t0_step = 512)
        """
        if self.A is None or self.B is None:
            raise RuntimeError("Phai goi fit() truoc khi predict().")

        step_indices = np.arange(t0_step, t0_step + n_future_steps)
        t_days = self._step_to_days(step_indices)

        u_future = self.A * np.cos(self.omega * t_days) + self.B * np.sin(self.omega * t_days)
        return u_future

    def physics_residual(self, u: np.ndarray, t0_step: int) -> np.ndarray:
        """
        Tinh residual R(t) = u''(t) + w^2 * u(t) bang sai phan huu han (finite difference),
        dung de cham diem verifier cho 1 candidate bat ky (khong nhat thiet
        phai la candidate do chinh minh sinh ra tu fit()).

        u: mang 1D gia tri candidate can kiem tra
        """
        dt = self.interval_days
        # dao ham bac 2 xap xi bang sai phan trung tam
        u_tt = np.zeros_like(u)
        u_tt[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dt ** 2)
        u_tt[0] = u_tt[1]     # bien: lap gia tri lan can de tranh loi tai 2 dau mut
        u_tt[-1] = u_tt[-2]

        step_indices = np.arange(t0_step, t0_step + len(u))
        t_days = self._step_to_days(step_indices)

        residual = u_tt + (self.omega ** 2) * u
        return residual


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    interval_hours = 6.0
    history_steps = 512
    future_steps = 128

    # ---- Sinh du lieu gia lap: dao dong nam that + nhieu + 1 xu huong nho ----
    true_omega = 2 * np.pi / 365
    interval_days = interval_hours / 24.0
    t_all = np.arange(history_steps + future_steps) * interval_days

    true_A, true_B = 5.0, 2.0          # bien do "that" gia lap
    seasonal_true = true_A * np.cos(true_omega * t_all) + true_B * np.sin(true_omega * t_all)
    trend_true = 0.002 * t_all          # xu huong am dan len rat nho
    noise = np.random.normal(0, 0.5, size=len(t_all))

    full_series = seasonal_true + trend_true + noise

    history = full_series[:history_steps]
    future_gt = full_series[history_steps:]

    # ---- Fit tren lich su, du bao tuong lai ----
    forecaster = HarmonicOscillatorForecaster(period_days=365, interval_hours=interval_hours)
    A_hat, B_hat = forecaster.fit(history, t0_step=0)
    print(f"A_hat = {A_hat:.4f} (true A = {true_A})")
    print(f"B_hat = {B_hat:.4f} (true B = {true_B})")

    future_pred = forecaster.predict(n_future_steps=future_steps, t0_step=history_steps)

    # ---- Tinh physics residual tren chinh doan du bao ----
    residual = forecaster.physics_residual(future_pred, t0_step=history_steps)
    print(f"Physics residual trung binh |R(t)| tren doan du bao: {np.mean(np.abs(residual)):.6f}")

    # ---- Ve hinh ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    t_hist_days = np.arange(history_steps) * interval_days
    t_fut_days = np.arange(history_steps, history_steps + future_steps) * interval_days

    axes[0].plot(t_hist_days, history, color="black", label="Lich su (history)")
    axes[0].plot(t_fut_days, future_gt, color="green", linestyle="--", label="Thuc te tuong lai (ground truth)")
    axes[0].plot(t_fut_days, future_pred, color="red", label="Du bao tu phuong trinh tuan hoan")
    axes[0].axvline(t_hist_days[-1], color="gray", linestyle=":")
    axes[0].set_title("Du bao bang nghiem giai tich cua u'' + w^2 u = 0")
    axes[0].set_xlabel("Thoi gian (ngay)")
    axes[0].set_ylabel("Gia tri (da chuan hoa)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_fut_days, residual, color="purple")
    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].set_title("Physics residual R(t) = u''(t) + w^2 u(t) tren doan du bao")
    axes[1].set_xlabel("Thoi gian (ngay)")
    axes[1].set_ylabel("Residual")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("harmonic_oscillator_forecast_demo.png", dpi=200)
    plt.show()
    print("\nDa luu hinh: harmonic_oscillator_forecast_demo.png")