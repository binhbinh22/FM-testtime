import numpy as np
from typing import Tuple, Dict, Optional, List


class ERA5PhysicsEnergyScorer:
    """
    Cài đặt bám sát Equation 52 (PINFDiT, ICLR 2026) cho ràng buộc vật lý t2m
    trên ERA5 / WeatherBench2.

    Phương trình liên tục (continuity equation) áp dụng cho trường nhiệt độ T
    bị vận chuyển bởi trường gió v = (u, v):

        dT/dt = -(v . grad(T)) - T * (div v)
              = Transport Term + Compression Term

    LƯU Ý QUAN TRỌNG (theo đúng paper):
    Nhiệt độ 2m KHÔNG phải đại lượng bảo toàn tuyệt đối do có số hạng
    diabatic (bức xạ mặt trời, surface heat flux, chu kỳ ngày đêm...).
    Vì vậy phương trình này chỉ được dùng làm SOFT REGULARIZATION,
    không phải constraint cứng (residual = 0 KHÔNG đồng nghĩa "đúng").

    Năng lượng tổng dùng để xếp hạng / hướng dẫn inference:
        E(x) = K(x; F) + alpha * log p_theta(x)

    Trong đó:
        K(x; F)       : physics residual (phần này class xử lý)
        log p_theta(x): learned prior (diffusion model) - xử lý phần diabatic,
                         KHÔNG nằm trong class này, phải truyền từ bên ngoài.
    """

    def __init__(self, dx: float = 1.0, dy: float = 1.0, dt: float = 1.0, alpha: float = 1.0):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.alpha = alpha  # trọng số cân bằng giữa physics residual và learned prior

    # ---------- K(x; F): physics residual thuần túy ----------
    def pde_residual_field(self, T_scenario: np.ndarray, u_wind: np.ndarray, v_wind: np.ndarray) -> np.ndarray:
        """
        Trả về residual field đầy đủ (không rút gọn thành scalar), để có thể
        dùng làm map trực quan hoặc tổng hợp linh hoạt về sau.

        T_scenario, u_wind, v_wind: shape (time_steps, ny, nx)
        Return shape: (time_steps - 1, ny, nx)
        """
        dT_dt = (T_scenario[1:] - T_scenario[:-1]) / self.dt

        T_align = T_scenario[:-1]
        u_align = u_wind[:-1]
        v_align = v_wind[:-1]

        # Sai phân trung tâm (np.gradient) cho spatial derivatives
        dT_dy, dT_dx = np.gradient(T_align, self.dy, self.dx, axis=(1, 2))
        du_dy, du_dx = np.gradient(u_align, self.dy, self.dx, axis=(1, 2))
        dv_dy, dv_dx = np.gradient(v_align, self.dy, self.dx, axis=(1, 2))

        transport_term = -(u_align * dT_dx + v_align * dT_dy)
        compression_term = -T_align * (du_dx + dv_dy)

        residual = dT_dt - transport_term - compression_term
        return residual

    def pde_residual_score(self, T_scenario: np.ndarray, u_wind: np.ndarray, v_wind: np.ndarray) -> float:
        """K(x; F) dạng scalar - MSE của residual field."""
        residual = self.pde_residual_field(T_scenario, u_wind, v_wind)
        return float(np.mean(residual ** 2))

    def normalized_residual_score(
        self, T_scenario: np.ndarray, u_wind: np.ndarray, v_wind: np.ndarray, eps: float = 1e-6
    ) -> float:
        """
        (Đã bỏ - xem composite_score bên dưới)
        Lý do: chia K cho phương sai của CHÍNH kịch bản không sửa được lỗi,
        vì kịch bản hằng số tuyệt đối có cả tử và mẫu đều -> 0, tỷ lệ vẫn 0.
        Cần so sánh với độ biến thiên THAM CHIẾU (lịch sử thật) thay vì tự
        chuẩn hóa nội tại. Xem `composite_score`.
        """
        residual = self.pde_residual_field(T_scenario, u_wind, v_wind)
        K = np.mean(residual ** 2)
        temporal_variance = np.var(T_scenario, axis=0).mean()
        return float(K / (temporal_variance + eps))

    def composite_score(
        self,
        T_scenario: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        history_t2m: np.ndarray,
        lambda_activity: float = 1.0,
    ) -> Dict[str, float]:
        """
        Điểm tổng hợp CHỐNG "trivial solution bias":

            score = K_physics_residual + lambda_activity * activity_penalty

        activity_penalty phạt các kịch bản "phẳng bất thường" (biến thiên
        thấp hơn hẳn so với biến thiên thật trong lịch sử), để verifier
        không còn bị dụ chọn nghiệm hằng số chỉ vì nó thỏa PDE một cách
        tầm thường (residual ~ 0).

        history_t2m: chuỗi lịch sử thật, shape (history_steps, ny, nx),
                     dùng làm tham chiếu "mức biến thiên hợp lý".
        """
        K = self.pde_residual_score(T_scenario, u_wind, v_wind)

        candidate_variance = np.var(T_scenario, axis=0).mean()
        historical_variance = np.var(history_t2m, axis=0).mean()

        # Phạt nếu kịch bản "phẳng" hơn hẳn lịch sử; không phạt nếu biến
        # thiên bằng hoặc nhiều hơn lịch sử (không có gì "bất thường phẳng").
        activity_penalty = max(0.0, historical_variance - candidate_variance) ** 2

        score = K + lambda_activity * activity_penalty
        return {
            "K_physics_residual": K,
            "candidate_variance": float(candidate_variance),
            "historical_variance": float(historical_variance),
            "activity_penalty": float(activity_penalty),
            "composite_score": float(score),
        }

    def select_top_k_composite(
        self,
        T_scenarios: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        history_t2m: np.ndarray,
        k: int = 3,
        lambda_activity: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        """
        Bản select_top_k đã vá lỗi trivial-solution, dùng composite_score
        thay vì pde_residual_score thuần túy.
        """
        n = T_scenarios.shape[0]
        all_details = [
            self.composite_score(T_scenarios[i], u_wind, v_wind, history_t2m, lambda_activity)
            for i in range(n)
        ]
        scores = np.array([d["composite_score"] for d in all_details])
        ranking = np.argsort(scores)
        top_k_indices = ranking[:k]
        top_k_scenarios = T_scenarios[top_k_indices]
        return top_k_indices, top_k_scenarios, all_details

    # ---------- E(x) = K(x;F) + alpha * log p_theta(x) ----------
    def energy_score(
        self,
        T_scenario: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        log_p_theta: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Tính năng lượng tổng hợp theo đúng Eq. trong paper.

        log_p_theta: log-likelihood từ diffusion model (learned prior) cho
                     scenario này. Nếu không cung cấp, chỉ trả về K(x;F)
                     và cảnh báo rằng phần diabatic chưa được mô hình hóa.
        """
        K = self.pde_residual_score(T_scenario, u_wind, v_wind)

        if log_p_theta is None:
            return {"K_physics_residual": K, "log_p_theta": None, "energy_E": K}

        # Energy càng THẤP càng tốt trong các bài PINFDiT-style (giống năng lượng
        # trong energy-based model / diffusion guidance). Nếu log_p_theta là
        # log-likelihood (càng cao càng "hợp lý"), ta trừ đi để giữ chiều "thấp = tốt".
        E = K - self.alpha * log_p_theta
        return {"K_physics_residual": K, "log_p_theta": log_p_theta, "energy_E": E}

    # ---------- Xếp hạng nhiều kịch bản ----------
    def rank_scenarios(
        self,
        T_scenarios: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        log_p_theta_list: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        T_scenarios: shape (n_scenarios, time_steps, ny, nx)
        log_p_theta_list: shape (n_scenarios,), log-likelihood từ diffusion prior
                           cho mỗi kịch bản (nếu có). Nếu None -> chỉ dùng K(x;F),
                           tương đương "pure advection validator" (KHÔNG khuyến nghị
                           dùng làm tiêu chí duy nhất theo đúng tinh thần paper).
        """
        n = T_scenarios.shape[0]
        K_scores = np.array([
            self.pde_residual_score(T_scenarios[i], u_wind, v_wind) for i in range(n)
        ])

        if log_p_theta_list is None:
            E_scores = K_scores
        else:
            E_scores = K_scores - self.alpha * log_p_theta_list

        ranking = np.argsort(E_scores)  # index 0 = năng lượng thấp nhất = tốt nhất
        details = {
            "K_physics_residual": K_scores,
            "log_p_theta": log_p_theta_list,
            "energy_E": E_scores,
        }
        return ranking, details

    # ---------- Chọn thẳng top-k kịch bản tốt nhất ----------
    def select_top_k(
        self,
        T_scenarios: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        k: int = 3,
        log_p_theta_list: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Dùng physics làm verifier để chọn thẳng ra k kịch bản tốt nhất
        trong tổng số N kịch bản (ví dụ N=10 -> chọn k=3).

        Return:
            top_k_indices  : index của k kịch bản tốt nhất trong mảng gốc,
                              đã sắp xếp từ tốt nhất -> kém nhất trong top-k
            top_k_scenarios: array shape (k, time_steps, ny, nx)
            details         : toàn bộ điểm số của tất cả N kịch bản (để đối chiếu)
        """
        n = T_scenarios.shape[0]
        if k > n:
            raise ValueError(f"k={k} lớn hơn số kịch bản hiện có (n={n})")

        ranking, details = self.rank_scenarios(T_scenarios, u_wind, v_wind, log_p_theta_list)

        top_k_indices = ranking[:k]
        top_k_scenarios = T_scenarios[top_k_indices]

        return top_k_indices, top_k_scenarios, details


if __name__ == "__main__":
    ny, nx = 32, 64
    predict_steps = 24
    n_scenarios = 10

    # Giả lập: dữ liệu gió tương lai đã biết (u10, v10) và 10 kịch bản t2m từ mô hình AI
    u10_wind = np.random.rand(predict_steps, ny, nx)
    v10_wind = np.random.rand(predict_steps, ny, nx)
    T_scenarios = np.random.rand(n_scenarios, predict_steps, ny, nx)  # (10, 24, 32, 64)

    scorer = ERA5PhysicsEnergyScorer(dx=1.0, dy=1.0, dt=1.0, alpha=1.0)

    # ĐÚNG YÊU CẦU: 10 kịch bản -> dùng physics làm verifier -> chọn ra 3 tốt nhất
    top3_idx, top3_scenarios, details = scorer.select_top_k(
        T_scenarios, u10_wind, v10_wind, k=3
    )

    print("Index của 3 kịch bản tốt nhất (trong 10 kịch bản gốc):", top3_idx)
    print("Physics residual (K) của toàn bộ 10 kịch bản:", np.round(details["K_physics_residual"], 6))
    print("Shape của 3 kịch bản được chọn:", top3_scenarios.shape)  # (3, 24, 32, 64)


def compute_real_grid_spacing(lat_array: np.ndarray, resolution_deg: float = 5.625) -> Dict[str, np.ndarray]:
    """
    Tính dx, dy thật (mét) cho lưới lat/lon toàn cầu.
    dy: cố định theo vĩ độ (kinh tuyến cách đều nhau)
    dx: thay đổi theo vĩ độ do kinh tuyến hội tụ về cực -> dx = R*cos(lat)*dlon
    """
    R_EARTH = 6_371_000.0  # bán kính Trái Đất (m)
    dlat_rad = np.deg2rad(resolution_deg)
    dlon_rad = np.deg2rad(resolution_deg)

    dy = R_EARTH * dlat_rad  # hằng số, không đổi theo lat
    dx = R_EARTH * np.cos(np.deg2rad(lat_array)) * dlon_rad  # array theo từng hàng vĩ độ

    return {"dx_per_lat_row": dx, "dy": dy}