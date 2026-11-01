import numpy as np
from typing import Dict, List, Optional, Tuple


class PressureWindSmoothnessScorer:
    """
    L_pw (pressure-wind consistency) va L_smooth (energy-aware smoothness),
    theo dung Eq. 11-12 trong PhysicsFormer (arXiv:2406.14399).

    KHONG can train mang neural nao:
      - L_smooth khong co tham so, tinh thang tu candidate.
      - alpha_i trong L_pw (he so lien he bien doi ap suat voi bien doi
        gio, xap xi quan he dia chuyen phong) duoc fit bang least-squares
        DANG DONG (closed-form, 1 cong thuc dai so), KHONG phai gradient
        descent - fit MOT LAN tu lich su thuc, roi dung lai cho moi candidate.

    LUU Y DU LIEU: pipeline ERA5 hien tai (t2m/u10/v10) KHONG co bien ap
    suat (mean_sea_level_pressure). L_pw chi tinh duoc khi P_scenario duoc
    cung cap (can them bien nay vao data loader). L_smooth dung duoc ngay
    voi du lieu hien co.
    """

    def __init__(self, dt: float = 1.0):
        self.dt = dt  # khoang thoi gian giua 2 buoc (gio), dung chuan hoa sai phan bac 1

    # ---------- wind speed (scalar) tu 2 thanh phan u, v ----------
    @staticmethod
    def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.sqrt(u ** 2 + v ** 2)

    # ---------- L_pw: pressure-wind consistency ----------
    @staticmethod
    def fit_alpha(history_P: np.ndarray, history_V: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Fit alpha TUNG DIEM LUOI bang OLS dang dong (khong train lap):
            alpha_i = argmin_a sum_t (dP_t - a * dV_t)^2 = <dP, dV> / <dV, dV>

        history_P, history_V: shape (time_steps, ny, nx) - DU LIEU LICH SU
        THAT (khong phai candidate), dung MOT LAN de uoc luong he so ty le
        giua bien doi ap suat va bien doi toc do gio tai moi diem luoi.

        Return: alpha, shape (ny, nx)
        """
        dP = history_P[1:] - history_P[:-1]
        dV = history_V[1:] - history_V[:-1]
        numerator = np.sum(dP * dV, axis=0)
        denominator = np.sum(dV * dV, axis=0) + eps
        return numerator / denominator

    def pressure_wind_consistency_score(
        self, P_scenario: np.ndarray, V_scenario: np.ndarray, alpha: np.ndarray,
    ) -> float:
        """
        L_pw cho MOT candidate, dang scalar (MSE cua residual).

        P_scenario, V_scenario: shape (time_steps, ny, nx) - ap suat va
        toc do gio (scalar, xem wind_speed()) CUA CANDIDATE dang xet.
        alpha: shape (ny, nx), lay tu fit_alpha() tren lich su.
        """
        dP = (P_scenario[1:] - P_scenario[:-1]) / self.dt
        dV = (V_scenario[1:] - V_scenario[:-1]) / self.dt
        residual = dP - alpha[None, :, :] * dV
        return float(np.mean(residual ** 2))

    # ---------- L_smooth: energy-aware smoothness ----------
    @staticmethod
    def _second_diff(x: np.ndarray) -> np.ndarray:
        """Delta_t^2 x[t] = x[t+1] - 2*x[t] + x[t-1]."""
        return x[2:] - 2 * x[1:-1] + x[:-2]

    def energy_smoothness_score(self, T_scenario: np.ndarray, V_scenario: np.ndarray) -> float:
        """
        L_smooth cho MOT candidate - KHONG tham so, khong can lich su,
        khong can fit/train gi ca.
        """
        d2T = self._second_diff(T_scenario)
        d2V = self._second_diff(V_scenario)
        return float(np.mean(d2T ** 2) + np.mean(d2V ** 2))

    # ---------- Gop lai thanh verifier hoan chinh ----------
    def composite_score(
        self,
        T_scenario: np.ndarray,
        V_scenario: np.ndarray,
        alpha: Optional[np.ndarray] = None,
        P_scenario: Optional[np.ndarray] = None,
        lambda_pw: float = 1.0,
        lambda_smooth: float = 1.0,
    ) -> Dict[str, float]:
        """
        score = lambda_pw * L_pw + lambda_smooth * L_smooth

        Neu khong co P_scenario/alpha (chua co bien ap suat trong pipeline),
        L_pw = 0.0 va chi L_smooth duoc tinh.
        """
        details: Dict[str, float] = {}

        if P_scenario is not None and alpha is not None:
            details["L_pw"] = self.pressure_wind_consistency_score(P_scenario, V_scenario, alpha)
        else:
            details["L_pw"] = 0.0

        details["L_smooth"] = self.energy_smoothness_score(T_scenario, V_scenario)
        details["composite_score"] = lambda_pw * details["L_pw"] + lambda_smooth * details["L_smooth"]
        return details

    def select_top_k(
        self,
        T_scenarios: np.ndarray,
        V_scenarios: np.ndarray,
        alpha: Optional[np.ndarray] = None,
        P_scenarios: Optional[np.ndarray] = None,
        k: int = 3,
        lambda_pw: float = 1.0,
        lambda_smooth: float = 1.0,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """
        T_scenarios, V_scenarios: shape (n_scenarios, time_steps, ny, nx)
        P_scenarios (optional): shape (n_scenarios, time_steps, ny, nx)
        """
        n = T_scenarios.shape[0]
        all_details = []
        for i in range(n):
            P_i = P_scenarios[i] if P_scenarios is not None else None
            all_details.append(self.composite_score(
                T_scenarios[i], V_scenarios[i], alpha=alpha, P_scenario=P_i,
                lambda_pw=lambda_pw, lambda_smooth=lambda_smooth,
            ))
        scores = np.array([d["composite_score"] for d in all_details])
        ranking = np.argsort(scores)  # index 0 = score thap nhat = tot nhat
        return ranking[:k], all_details


if __name__ == "__main__":
    ny, nx = 32, 64
    predict_steps = 24
    n_scenarios = 10

    scorer = PressureWindSmoothnessScorer(dt=1.0)

    # --- Truong hop CHUA co pressure (dung dung du lieu hien co: t2m/u10/v10) ---
    T_scenarios = np.random.rand(n_scenarios, predict_steps, ny, nx)
    u_scenarios = np.random.rand(n_scenarios, predict_steps, ny, nx)
    v_scenarios = np.random.rand(n_scenarios, predict_steps, ny, nx)
    V_scenarios = scorer.wind_speed(u_scenarios, v_scenarios)

    top3_idx, details = scorer.select_top_k(T_scenarios, V_scenarios, k=3)
    print("Chi L_smooth (chua co pressure) - top3 index:", top3_idx)
    print("L_smooth cua 10 candidate:", np.round([d["L_smooth"] for d in details], 6))

    # --- Truong hop CO them pressure (sau khi them bien vao data loader) ---
    history_P = np.random.rand(200, ny, nx) * 1000 + 100000  # gia lap SLP lich su (Pa)
    history_u = np.random.rand(200, ny, nx)
    history_v = np.random.rand(200, ny, nx)
    history_V = scorer.wind_speed(history_u, history_v)
    alpha = scorer.fit_alpha(history_P, history_V)  # fit 1 lan, khong train lap

    P_scenarios = np.random.rand(n_scenarios, predict_steps, ny, nx) * 1000 + 100000

    top3_idx_pw, details_pw = scorer.select_top_k(
        T_scenarios, V_scenarios, alpha=alpha, P_scenarios=P_scenarios, k=3,
    )
    print("\nCo ca L_pw - top3 index:", top3_idx_pw)
    print("composite_score cua 10 candidate:",
          np.round([d["composite_score"] for d in details_pw], 6))
