import numpy as np
from typing import Optional


class ERA5DTWEnergyScorer:
    """
    Tính E_temporal(Y_fine, anchor) bằng Dynamic Time Warping (DTW), dùng
    làm 1 trong 3 thành phần của verifier tổng hợp:

        E(Y_fine) = w1 * E_temporal(Y_fine, anchor)
                  + w2 * E_LF(Y_fine, Y_coarse)
                  + w3 * E_physics(Y_fine)

    CÔNG THỨC DTW:
    Cho 2 chuỗi 1D A = (a_1, ..., a_n), B = (b_1, ..., b_m):

        d(i, j)  = |a_i - b_j|                          (local cost)
        D(0, 0)  = 0
        D(i, j)  = d(i, j) + min( D(i-1, j),             # insertion
                                   D(i, j-1),             # deletion
                                   D(i-1, j-1) )          # match

        DTW(A, B) = D(n, m)

    Chuẩn hóa theo tổng độ dài đường đi để so sánh công bằng giữa các
    candidate/anchor có độ dài khác nhau:

        DTW_norm(A, B) = D(n, m) / (n + m)

    Ý NGHĨA TRONG PIPELINE (GIẢ ĐỊNH CẦN XÁC NHẬN):
    `anchor` mặc định đang được coi là đoạn LỊCH SỬ THẬT gần nhất
    (history_t2m) - DTW khi đó đo mức độ "trôi chảy" (temporal
    consistency) giữa quỹ đạo dự báo Y_fine và hình dạng/xu hướng quan
    sát được ngay trước đó trong lịch sử thật. Đây là một lựa chọn hợp
    lý vì DTW không đòi hỏi 2 chuỗi cùng độ dài, nên anchor = toàn bộ
    history_t2m dùng được trực tiếp mà không cần cắt/pad.
    Nếu ý đồ của bạn là anchor khác (ví dụ ground truth, hoặc một
    candidate coarse đã upsample), chỉ cần đổi giá trị truyền vào
    dtw_field_score / score_batch - code không phụ thuộc vào nguồn gốc
    của anchor.

    Ràng buộc Sakoe-Chiba band (self.window) giới hạn |i-j| <= window,
    giúp giảm chi phí tính toán (O(n*window) thay vì O(n*m)) và tránh
    warping quá mức (candidate bị "kéo dãn/co lại" bất thường để khớp
    anchor một cách giả tạo).
    """

    def __init__(self, window: Optional[int] = None, distance: str = "abs"):
        """
        window: độ rộng Sakoe-Chiba band (số bước cho phép lệch |i-j|).
                None = không giới hạn (full DTW, chi phí O(n*m)).
        distance: "abs" -> d(i,j) = |a_i - b_j|
                  "sq"  -> d(i,j) = (a_i - b_j)^2
        """
        if distance not in ("abs", "sq"):
            raise ValueError(f"distance phải là 'abs' hoặc 'sq', nhận: {distance}")
        self.window = window
        self.distance = distance

    # ---------- local cost ----------
    def _local_cost(self, ai: float, bj: float) -> float:
        diff = ai - bj
        return abs(diff) if self.distance == "abs" else diff * diff

    # ---------- DTW cho 1 chuỗi 1D ----------
    def _dtw_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        DTW thuần túy giữa 2 chuỗi 1D, cài bằng quy hoạch động (numpy
        thuần, không phụ thuộc thư viện ngoài như fastdtw/dtaidistance
        để dễ kiểm soát và không thêm dependency).

        Độ phức tạp: O(n*m), hoặc O(n*window) nếu có Sakoe-Chiba band.

        Trả về DTW_norm(a, b) = D(n, m) / (n + m).
        """
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            raise ValueError("Chuỗi rỗng không thể tính DTW.")

        D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        D[0, 0] = 0.0

        band = max(n, m) if self.window is None else self.window

        for i in range(1, n + 1):
            j_lo = max(1, i - band)
            j_hi = min(m, i + band)
            for j in range(j_lo, j_hi + 1):
                cost = self._local_cost(a[i - 1], b[j - 1])
                D[i, j] = cost + min(
                    D[i - 1, j],
                    D[i, j - 1],
                    D[i - 1, j - 1],
                )

        if np.isinf(D[n, m]):
            raise RuntimeError(
                "DTW band quá hẹp so với chênh lệch độ dài (n, m) giữa "
                "Y_fine và anchor; hãy tăng window hoặc để None."
            )

        return float(D[n, m] / (n + m))

    # ---------- E_temporal trên toàn trường không gian ----------
    def dtw_field_score(self, Y_fine: np.ndarray, anchor: np.ndarray) -> float:
        """
        E_temporal(Y_fine, anchor) trên toàn bộ trường không gian.

        Input:
            Y_fine: shape (T_fine, ny, nx)
            anchor: shape (T_anchor, ny, nx) - T_anchor có thể khác T_fine
                    (DTW xử lý được 2 độ dài khác nhau)
        Return:
            float - trung bình DTW_norm trên toàn bộ điểm lưới (ny*nx)

        CẢNH BÁO ĐỘ PHỨC TẠP: O(ny * nx * T_fine * T_anchor) (hoặc nhân
        thêm window/max(T_fine,T_anchor) nếu có band). Với grid lớn
        (vd 32x64 = 2048 điểm) và T~24-64, hàm này có thể chậm - nên
        dùng self.window để giới hạn band, hoặc subsample điểm lưới nếu
        grid quá lớn.
        """
        if Y_fine.shape[1:] != anchor.shape[1:]:
            raise ValueError(
                f"Y_fine và anchor phải cùng (ny, nx): "
                f"{Y_fine.shape[1:]} vs {anchor.shape[1:]}"
            )

        ny, nx = Y_fine.shape[1], Y_fine.shape[2]
        scores = np.empty((ny, nx), dtype=np.float64)

        for lat in range(ny):
            for lon in range(nx):
                a = Y_fine[:, lat, lon]
                b = anchor[:, lat, lon]
                scores[lat, lon] = self._dtw_1d(a, b)

        return float(scores.mean())

    # ---------- Tính cho nhiều candidate cùng lúc ----------
    def score_batch(self, T_scenarios: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """
        Tính E_temporal cho từng candidate trong batch, so với MỘT anchor
        dùng chung.

        Input:
            T_scenarios: shape (n_scenarios, T_fine, ny, nx)
            anchor: shape (T_anchor, ny, nx)
        Return:
            scores: shape (n_scenarios,)
        """
        n = T_scenarios.shape[0]
        scores = np.empty(n, dtype=np.float64)
        for i in range(n):
            scores[i] = self.dtw_field_score(T_scenarios[i], anchor)
        return scores


if __name__ == "__main__":
    np.random.seed(0)

    n_scenarios = 5
    T_fine, ny, nx = 24, 32, 64
    T_anchor = 48  # anchor (history) có thể khác độ dài Y_fine

    T_scenarios = np.random.rand(n_scenarios, T_fine, ny, nx)
    anchor = np.random.rand(T_anchor, ny, nx)

    # LƯU Ý: khi anchor (history) dài hơn nhiều so với Y_fine (forecast),
    # window PHẢI >= |T_fine - T_anchor| để band còn "chạm" được tới
    # D[n, m], nếu không code sẽ raise RuntimeError (đây là hành vi ĐÚNG,
    # không phải bug - band quá hẹp so với chênh lệch độ dài là lỗi cấu
    # hình cần được báo, không nên tính ra một số sai một cách im lặng).
    # Với T_fine=24, T_anchor=48 (chênh lệch 24), cần window >= 24.
    scorer = ERA5DTWEnergyScorer(window=24, distance="abs")

    scores = scorer.score_batch(T_scenarios, anchor)

    print("=" * 60)
    print("E_temporal (DTW) cho từng candidate")
    print("=" * 60)
    for i, s in enumerate(scores):
        print(f"  Candidate {i:02d}: E_temporal = {s:.6f}")