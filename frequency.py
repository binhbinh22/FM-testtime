import numpy as np

from era5_multiscale import downsample_avgpool


class ERA5FrequencyEnergyScorer:
    """
    Tính E_LF(Y_fine, Y_coarse) - khoảng cách miền tần số giữa dự báo fine
    và dự báo coarse, dùng làm 1 trong 3 thành phần của verifier tổng hợp:

        E(Y_fine) = w1 * E_temporal(Y_fine, anchor)
                  + w2 * E_LF(Y_fine, Y_coarse)
                  + w3 * E_physics(Y_fine)

    NGUYÊN TẮC BẮT BUỘC (đã thống nhất): KHÔNG so trực tiếp FFT của Y_fine
    (độ dài T_fine) với FFT của Y_coarse (độ dài T_coarse) vì 2 phổ có
    bước tần số (frequency resolution) khác nhau -> chỉ số k không tương
    ứng cùng một tần số vật lý giữa 2 phổ.

    CÁCH ĐÚNG: downsample Y_fine về ĐÚNG scale của Y_coarse TRƯỚC KHI FFT,
    dùng CHUNG hàm downsample_avgpool (era5_multiscale.py, cùng factor và
    cùng quy ước cắt biên đã dùng để tạo X_coarse từ lịch sử X):

        Y_fine_ds = downsample_avgpool(Y_fine, factor)   # cùng độ dài Y_coarse
        E_LF(Y_fine, Y_coarse) = FreDF_L1( FFT(Y_fine_ds), FFT(Y_coarse) )

    Sau bước downsample, Y_fine_ds và Y_coarse CÙNG độ dài thời gian nên
    FFT của cả hai có cùng bước tần số -> so sánh trực tiếp toàn bộ phổ
    (không cần cắt M tần số thấp một cách tùy tiện như bản nháp trước).

    CÔNG THỨC FreDF (Learning to Forecast in the Frequency Domain):
        X[k] = FFT(x)[k]
        L_FreDF(x_pred, x_true) = (1/n) * sum_k |X_pred[k] - X_true[k]|
        (module số phức, hoặc tách Re/Im tùy self.mode)
    """

    def __init__(self, factor: int = 2, mode: str = "complex_abs"):
        """
        factor: hệ số downsample dùng để đưa Y_fine về scale của Y_coarse.
                BẮT BUỘC trùng với factor đã dùng để tạo X_coarse từ X
                ban đầu (mặc định = 2, theo TimeMixer 2-scale hiện tại).
        mode: "complex_abs" -> |X_pred[k] - X_true[k]| trên số phức
              "real_imag"   -> |Re diff| + |Im diff|
        """
        if mode not in ("complex_abs", "real_imag"):
            raise ValueError(
                f"mode phải là 'complex_abs' hoặc 'real_imag', nhận: {mode}"
            )
        self.factor = factor
        self.mode = mode

    def _fft_1d(self, x: np.ndarray) -> np.ndarray:
        return np.fft.fft(x, axis=0)

    def _l1_complex(self, Xp: np.ndarray, Xt: np.ndarray) -> float:
        if self.mode == "complex_abs":
            diff = np.abs(Xp - Xt)
        else:
            diff = np.abs(Xp.real - Xt.real) + np.abs(Xp.imag - Xt.imag)
        return float(np.mean(diff))

    def freq_l1_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        FreDF L1 giữa 2 chuỗi 1D CÙNG ĐỘ DÀI (đã downsample về cùng scale).
        Cố tình KHÔNG tự động zero-pad khi độ dài khác nhau, vì làm vậy sẽ
        làm sai lệch frequency resolution một cách âm thầm (xem docstring
        class) - lỗi sai sẽ được raise tường minh thay vì tính ra một con
        số trông hợp lý nhưng sai về ý nghĩa vật lý.
        """
        if len(a) != len(b):
            raise ValueError(
                f"freq_l1_1d yêu cầu 2 chuỗi CÙNG độ dài (đã downsample về "
                f"cùng scale). Nhận len(a)={len(a)}, len(b)={len(b)}. Hãy "
                "downsample Y_fine bằng downsample_avgpool trước khi gọi."
            )
        Xa = self._fft_1d(a)
        Xb = self._fft_1d(b)
        return self._l1_complex(Xa, Xb)

    def low_freq_field_score(self, Y_fine: np.ndarray, Y_coarse: np.ndarray) -> float:
        """
        E_LF(Y_fine, Y_coarse) trên toàn bộ trường không gian.

        Input:
            Y_fine:   shape (T_fine, ny, nx)   - candidate ở scale fine
            Y_coarse: shape (T_coarse, ny, nx) - candidate ở scale coarse
                      (T_coarse phải == T_fine // self.factor, tức Y_coarse
                      được Chronos sinh trên X_coarse với prediction_steps
                      đã chia factor tương ứng - đúng như trong file thí
                      nghiệm multiscale ban đầu)

        Bước 1: downsample Y_fine -> Y_fine_ds (cùng độ dài Y_coarse)
        Bước 2: FFT cả 2, tính FreDF L1 trên từng điểm lưới
        Bước 3: lấy trung bình không gian
        """
        if Y_fine.shape[1:] != Y_coarse.shape[1:]:
            raise ValueError(
                f"Y_fine và Y_coarse phải cùng (ny, nx): "
                f"{Y_fine.shape[1:]} vs {Y_coarse.shape[1:]}"
            )

        Y_fine_ds = downsample_avgpool(Y_fine, factor=self.factor)

        if Y_fine_ds.shape[0] != Y_coarse.shape[0]:
            raise ValueError(
                f"Sau downsample, Y_fine_ds có T={Y_fine_ds.shape[0]} "
                f"nhưng Y_coarse có T={Y_coarse.shape[0]}. Kiểm tra lại "
                "factor và prediction_steps của nhánh coarse (phải = "
                "prediction_steps fine // factor)."
            )

        ny, nx = Y_fine_ds.shape[1], Y_fine_ds.shape[2]
        scores = np.empty((ny, nx), dtype=np.float64)

        for lat in range(ny):
            for lon in range(nx):
                scores[lat, lon] = self.freq_l1_1d(
                    Y_fine_ds[:, lat, lon], Y_coarse[:, lat, lon]
                )

        return float(scores.mean())

    def score_batch(
        self,
        T_scenarios_fine: np.ndarray,
        Y_coarse_ref: np.ndarray,
    ) -> np.ndarray:
        """
        Tính E_LF cho từng candidate fine trong batch, so với MỘT candidate
        coarse tham chiếu cố định (Y_coarse_ref).

        LƯU Ý (điểm cần bạn xác nhận thêm): theo sơ đồ kiến trúc, nhánh
        coarse tạo ra K_C candidate tốt nhất (sau khi lọc bằng Physics),
        không phải chỉ 1 candidate. Hàm này hiện giả định bạn đã chọn ra
        MỘT candidate đại diện (ví dụ K_C_best[0], hoặc trung bình K_C_best)
        trước khi truyền vào đây. Nếu bạn muốn E_LF là khoảng cách nhỏ nhất
        (hoặc trung bình) tới CẢ K_C candidate, cần một hàm tổng hợp riêng
        gọi score_batch K_C lần rồi rút gọn (min/mean theo trục K) - có
        thể bổ sung sau khi thống nhất cách tổng hợp.

        Input:
            T_scenarios_fine: shape (n_scenarios, T_fine, ny, nx)
            Y_coarse_ref: shape (T_coarse, ny, nx)
        Return:
            scores: shape (n_scenarios,)
        """
        n = T_scenarios_fine.shape[0]
        scores = np.empty(n, dtype=np.float64)
        for i in range(n):
            scores[i] = self.low_freq_field_score(T_scenarios_fine[i], Y_coarse_ref)
        return scores


if __name__ == "__main__":
    np.random.seed(0)

    n_scenarios = 5
    T_fine, ny, nx = 24, 32, 64
    factor = 2
    T_coarse = T_fine // factor

    T_scenarios_fine = np.random.rand(n_scenarios, T_fine, ny, nx)
    Y_coarse_ref = np.random.rand(T_coarse, ny, nx)

    scorer = ERA5FrequencyEnergyScorer(factor=factor, mode="complex_abs")

    scores = scorer.score_batch(T_scenarios_fine, Y_coarse_ref)

    print("=" * 60)
    print("E_LF (Frequency L1, sau downsample về cùng scale)")
    print("=" * 60)
    for i, s in enumerate(scores):
        print(f"  Candidate {i:02d}: E_LF = {s:.6f}")