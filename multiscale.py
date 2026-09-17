import numpy as np


def downsample_avgpool(data: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downsample chuỗi thời gian bằng Average Pooling, theo đúng tinh thần
    TimeMixer (ICLR 2024, Sec 3.1 - Multiscale Mixing Architecture):

        x_m = AvgPool(x, kernel = 2^m)

    áp dụng TRỰC TIẾP lên chuỗi gốc x (không downsample đệ quy nhiều lần),
    để tránh sai số cộng dồn qua các lần pooling liên tiếp.

    QUY ƯỚC CẮT BIÊN (PHẢI dùng nhất quán ở MỌI nơi gọi hàm này):
    Nếu độ dài thời gian không chia hết cho `factor`, CẮT BỎ PHẦN ĐẦU
    chuỗi, GIỮ LẠI PHẦN CUỐI (đoạn gần thời điểm hiện tại nhất). Lý do:
    trong forecasting, phần cuối chuỗi (gần "now") quan trọng hơn phần
    đầu. Nếu quy ước này bị phá vỡ ở một nơi gọi hàm (ví dụ tạo X_coarse
    từ lịch sử) nhưng không nhất quán ở nơi khác (ví dụ tạo Y_fine_ds từ
    candidate dự đoán), 2 chuỗi sau downsample sẽ bị LỆCH PHA thời gian,
    khiến E_LF (frequency-domain distance) tính sai mà không có lỗi runtime
    nào báo cho biết.

    Input:
        data: shape (time_steps, ...) - trục 0 LUÔN là trục thời gian
        factor: hệ số downsample. factor=2 tương ứng kernel=2^1 trong
                công thức TimeMixer (bài toán hiện tại dùng đúng 2 scale:
                m=0 là fine gốc, m=1 là coarse với factor=2).

    Return:
        data_ds: shape (time_steps // factor, ...)
    """
    if factor < 1:
        raise ValueError(f"factor phải >= 1, nhận factor={factor}")
    if factor == 1:
        return data.copy()

    t = data.shape[0]
    new_t = t // factor
    if new_t == 0:
        raise ValueError(
            f"Độ dài thời gian ({t}) nhỏ hơn factor ({factor}), "
            "không đủ để downsample."
        )

    # Giữ lại đoạn cuối chuỗi (gần hiện tại nhất), cắt bỏ phần dư ở đầu
    trailing = data[-new_t * factor:]

    new_shape = (new_t, factor) + data.shape[1:]
    data_ds = trailing.reshape(new_shape).mean(axis=1)

    return data_ds


if __name__ == "__main__":
    # Test nhanh: chuỗi (10, 32, 64), factor=2 -> (5, 32, 64)
    np.random.seed(0)
    x = np.random.rand(10, 32, 64)
    x_ds = downsample_avgpool(x, factor=2)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {x_ds.shape}")
    assert x_ds.shape == (5, 32, 64)

    # Test trường hợp không chia hết: (11, 32, 64), factor=2 -> cắt bỏ 1 bước đầu -> (5, 32, 64)
    x_odd = np.random.rand(11, 32, 64)
    x_odd_ds = downsample_avgpool(x_odd, factor=2)
    print(f"Odd input shape : {x_odd.shape} -> Output shape: {x_odd_ds.shape}")
    assert x_odd_ds.shape == (5, 32, 64)

    print("OK - downsample_avgpool hoạt động đúng.")