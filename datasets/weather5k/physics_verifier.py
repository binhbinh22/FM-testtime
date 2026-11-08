import numpy as np
from typing import Dict, List, Tuple

"""
Physics verifier cho WEATHER-5K (du lieu TUNG TRAM, chuoi 1 chieu theo thoi
gian), theo dung Eq. 11-12 trong PhysicsFormer (arXiv:2406.14399) - chinh la
paper nguon cua WEATHER-5K, nen cong thuc nay duoc thiet ke/validate TREN
DUNG dataset nay (khac voi ban ERA5 truoc do phai suy dien).

KHAC voi era5_physics_pw_smooth.py (du lieu luoi (time, lat, lon)): o day moi
bien chi la 1 chuoi 1 chieu (time,) vi WEATHER-5K la diem tram roi rac, khong
phai luoi. File nay viet moi hoan toan, KHONG import/sua gi tu code ERA5.

==============================================================================
TRANG THAI (2026-10-31) - DA VERIFY BANG DO TRUC TIEP, KHONG CON SUY DOAN:
==============================================================================
Do truc tiep Pearson correlation(diem verifier, MSE THAT) tren 113,440 cap
(candidate, tram) tu Chronos-1 that:
    corr(L_pw score, MSE_SLP)        = 0.310  (tuong quan DUONG, trung binh)
    corr(L_pw score, MSE_WindSpeed)  = 0.032  (gan nhu khong)
    corr(L_smooth score, MSE_Temp)   = 0.012  (gan nhu khong - VO DUNG)

=> `L_pw` LA VERIFIER CHINH tu gio, mang tin hieu du bao that (it nhat cho
SLP). `L_smooth` KHONG CON DUOC SU DUNG trong pipeline chinh thuc nua - du
gia dinh vat ly cua no "dung" hon ve mat thong ke tho (xem ghi chu cu trong
energy_smoothness_score), no khong giup phan biet candidate tot/xau, dan den
Best-of-N+Physics(L_smooth) THUA ca Zero-shot/Mean-of-N tren toan bo 4 bien
test full-scale (xem docs/EXPERIMENT_REPORT_weather5k_chronos_ttc_physics.md).
Cac ham lien quan L_smooth (energy_smoothness_score, activity_penalty,
energy_smoothness_score_with_activity) VAN GIU LAI trong file nay chi de
tham khao/lich su, KHONG dung trong runner/kernel moi.

GIOI HAN cua L_pw can luu y: chi ap dung duoc cho CAP (SLP, WindSpeed) cung
luc (cong thuc can ca P va V). Temp, Dewpoint HIEN KHONG CO verifier nao ap
dung duoc - Best-of-N+Physics cho 2 bien nay dang la N/A, chi bao cao
Zero-shot/Mean-of-N.

KHONG can train neural network nao:
  - alpha (pressure-wind sensitivity): fit 1 LAN bang OLS dang dong (dai so,
    khong phai gradient descent) tu LICH SU THAT cua tram (vd val-2022), roi
    dung lai de cham diem cac candidate Chronos sinh ra o test-2023.

Luu y: WEATHER-5K da co san WND_RATE (do lon toc do gio) truc tiep, KHONG
can doi tu u/v nhu ERA5.
"""


class StationPhysicsVerifier:

    def __init__(self, dt: float = 1.0):
        self.dt = dt  # khoang thoi gian giua 2 buoc (gio), dung chuan hoa sai phan bac 1

    # ---------- L_pw: pressure-wind consistency ----------
    @staticmethod
    def fit_alpha(history_P: np.ndarray, history_V: np.ndarray, eps: float = 1e-8) -> float:
        """
        Fit alpha bang OLS dang dong (khong train lap):
            alpha = argmin_a sum_t (dP_t - a*dV_t)^2 = <dP, dV> / <dV, dV>

        history_P, history_V: chuoi LICH SU THAT cua 1 tram (vd nam val-2022),
        dung MOT LAN de uoc luong he so ty le giua bien doi ap suat va bien
        doi toc do gio tai tram do.
        """
        dP = history_P[1:] - history_P[:-1]
        dV = history_V[1:] - history_V[:-1]
        denom = float(np.sum(dV * dV)) + eps
        return float(np.sum(dP * dV) / denom)

    def pressure_wind_consistency_score(self, P_candidate: np.ndarray, V_candidate: np.ndarray,
                                         alpha: float) -> float:
        """L_pw cho MOT candidate (scalar, MSE cua residual)."""
        dP = (P_candidate[1:] - P_candidate[:-1]) / self.dt
        dV = (V_candidate[1:] - V_candidate[:-1]) / self.dt
        residual = dP - alpha * dV
        return float(np.mean(residual ** 2))

    # ---------- L_smooth: energy-aware smoothness ----------
    @staticmethod
    def _second_diff(x: np.ndarray) -> np.ndarray:
        """Delta_t^2 x[t] = x[t+1] - 2*x[t] + x[t-1]."""
        return x[2:] - 2 * x[1:-1] + x[:-2]

    def energy_smoothness_score(self, T_candidate: np.ndarray, V_candidate: np.ndarray) -> float:
        """L_smooth cho MOT candidate - khong tham so, khong can lich su."""
        d2T = self._second_diff(T_candidate)
        d2V = self._second_diff(V_candidate)
        return float(np.mean(d2T ** 2) + np.mean(d2V ** 2))

    @staticmethod
    def activity_penalty(T_candidate: np.ndarray, history_T: np.ndarray, eps: float = 1e-6) -> float:
        """
        Phat candidate "phang bat thuong" so voi do bien thien THAT trong lich
        su - chong nghiem tam thuong cho L_smooth (candidate hang so/gan hang
        so luon thoa L_smooth~0 mot cach gia tao). Giong dung cach lam trong
        era5_physics_energy.py (composite_score). Khong phat neu candidate
        bien thien BANG hoac NHIEU hon lich su (chi phat khi qua phang).

        CHUAN HOA theo historical_var -> ve thang [0,1], tranh lech thang do
        so voi L_smooth (da gap loi nay 1 lan: activity_penalty tho theo don
        vi do^2 lon hon L_smooth hang tram-nghin lan, lan at hoan toan diem so).
        """
        candidate_var = float(np.var(T_candidate))
        historical_var = float(np.var(history_T))
        return (max(0.0, historical_var - candidate_var) / (historical_var + eps)) ** 2

    def energy_smoothness_score_with_activity(self, T_candidate: np.ndarray, V_candidate: np.ndarray,
                                               history_T: np.ndarray, lambda_activity: float = 0.0,
                                               eps: float = 1e-6) -> float:
        """
        L_smooth (CHUAN HOA theo historical_var) + activity_penalty tuy chon
        (mac dinh TAT, lambda_activity=0.0).

        DA THU BAT activity_penalty 2 LAN (ca ban tho va ban da chuan hoa
        thang do + chuan hoa cua so thoi gian) - CA 2 LAN DEU LAM MSE TE HON
        HAN so voi L_smooth thuan (21.8-23.4 vs 16.8 tren full 5,672 tram
        WEATHER-5K). Nguyen nhan: cong thuc nay phat BAT KY candidate nao co
        phuong sai thap hon tham chieu, khong chi candidate suy bien phang
        tuyet doi - vi du lieu that von da "tron" (xem phan tich tuong quan:
        tron hon nhieu trang 220 lan), penalty nay thuong xuyen phat nham ca
        candidate tron-nhung-DUNG, day verifier chon candidate on ao hon,
        kem chinh xac hon. KHONG dung lambda_activity>0 cho den khi thiet ke
        lai cong thuc (vd chi phat truong hop GAN NHU HANG SO tuyet doi,
        khong phai so sanh ty le voi muc tham chieu).
        """
        historical_var = float(np.var(history_T))
        smooth = self.energy_smoothness_score(T_candidate, V_candidate) / (historical_var + eps)
        activity = self.activity_penalty(T_candidate, history_T, eps)
        return smooth + lambda_activity * activity

    # ---------- Gop lai thanh verifier hoan chinh ----------
    def composite_score(
        self,
        T_candidate: np.ndarray,
        V_candidate: np.ndarray,
        P_candidate: np.ndarray,
        alpha: float,
        lambda_pw: float = 1.0,
        lambda_smooth: float = 1.0,
    ) -> Dict[str, float]:
        """score = lambda_pw * L_pw + lambda_smooth * L_smooth (cang THAP cang tot)."""
        L_pw = self.pressure_wind_consistency_score(P_candidate, V_candidate, alpha)
        L_smooth = self.energy_smoothness_score(T_candidate, V_candidate)
        return {
            "L_pw": L_pw,
            "L_smooth": L_smooth,
            "composite_score": lambda_pw * L_pw + lambda_smooth * L_smooth,
        }

    def select_top_k(
        self,
        T_candidates: np.ndarray,
        V_candidates: np.ndarray,
        P_candidates: np.ndarray,
        alpha: float,
        k: int = 1,
        lambda_pw: float = 1.0,
        lambda_smooth: float = 1.0,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """
        T_candidates, V_candidates, P_candidates: shape (n_candidates, prediction_steps)
        Tra ve: (top_k_indices sap xep tot->kem, chi tiet diem cua TAT CA candidate)
        """
        n = T_candidates.shape[0]
        if k > n:
            raise ValueError(f"k={k} lon hon so candidate hien co (n={n})")

        all_details = [
            self.composite_score(T_candidates[i], V_candidates[i], P_candidates[i],
                                  alpha, lambda_pw, lambda_smooth)
            for i in range(n)
        ]
        scores = np.array([d["composite_score"] for d in all_details])
        ranking = np.argsort(scores)  # index 0 = score thap nhat = tot nhat
        return ranking[:k], all_details


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, n_candidates = 24, 20

    # Gia lap lich su (val-2022) de fit alpha
    history_P = 1013.0 + rng.normal(0, 1.0, size=200).cumsum() * 0.1
    history_V = 3.0 + rng.normal(0, 0.5, size=200)

    verifier = StationPhysicsVerifier(dt=1.0)
    alpha = verifier.fit_alpha(history_P, history_V)
    print("alpha fit tu history (gia lap):", round(alpha, 6))

    # Gia lap N candidate tu 1 model forecast (vd Chronos) cho 3 bien T, V, P
    T_candidates = 300.0 + rng.normal(0, 1.5, size=(n_candidates, H)).cumsum(axis=1) * 0.1
    V_candidates = 3.0 + rng.normal(0, 1.0, size=(n_candidates, H))
    P_candidates = 1013.0 + rng.normal(0, 1.0, size=(n_candidates, H)).cumsum(axis=1) * 0.1

    top_idx, details = verifier.select_top_k(T_candidates, V_candidates, P_candidates, alpha, k=3)
    print("Top-3 candidate (index, tu tot nhat):", top_idx.tolist())
    print("composite_score cua ca 20 candidate:",
          np.round([d["composite_score"] for d in details], 4))
