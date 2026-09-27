import numpy as np

from spatial_decompose import (
    aggregate_to_clusters,
    spatial_decompose,
    history_reference_Q,
    spatial_consistency_score,
)


def compute_spatial_err(
    final_candidate: np.ndarray,
    history: np.ndarray,
    S: np.ndarray,
    W: np.ndarray,
) -> float:
    """
    Tinh Spatial-Err HAU KIEM (post-hoc) tren 1 candidate da duoc CHON SAN
    boi bat ky kien truc nao (Greedy, Trend+FFT+Physics, Spatial-Cluster
    BoN...), dung CHUNG 1 dinh nghia (spatial_decompose.py) de so sanh
    cong bang giua cac kien truc - ke ca kien truc KHONG dung S/W trong
    qua trinh chon candidate (Greedy, Trend+FFT+Physics).

    Y nghia: cho biet candidate cuoi cung "hop ly ve khong gian" toi dau
    so voi lich su that, BAT KE kien truc do co toi uu hoa truc tiep cho
    dieu nay hay khong - day la cach DUY NHAT de chung minh Spatial-Cluster
    BoN thuc su cai thien tuong tac khong gian, thay vi chi tu bao cao
    diem so cua chinh no.

    final_candidate: (T_pred, ny, nx)
    history:          (T_hist, ny, nx) - dung de tinh history_reference_Q
    S: (V, C) ma tran gan cum (spectral_clustering.py)
    W: (V, V) do thi kNN (era5_graph.py) - PHAI la CUNG 1 do thi/cum dung
       cho moi kien truc trong 1 bang so sanh, neu khong Spatial-Err se
       khong con y nghia so sanh cheo.
    """
    T_hist, ny, nx = history.shape
    history_flat = history.reshape(T_hist, ny * nx)
    history_global_broadcast = aggregate_to_clusters(history_flat, S) @ S.T
    history_local = history_flat - history_global_broadcast
    hist_Q = history_reference_Q(history_local, W)

    T_pred = final_candidate.shape[0]
    candidate_flat = final_candidate.reshape(T_pred, ny * nx)
    _, _, Y_local = spatial_decompose(candidate_flat, S)

    return spatial_consistency_score(Y_local, W, hist_Q)
