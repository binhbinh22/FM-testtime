import numpy as np
from typing import Tuple

from spectral_clustering import cluster_pooling_matrix


def aggregate_to_clusters(X_flat: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    X_flat: (T, V) -> X_c: (T, C), trung binh trong tung cum theo ma tran
    gan S (V, C). Dung CHUNG ham nay cho ca lich su (muc 2.1/2.2, gop X
    thanh C sieu nut) lan cho candidate can gop len cap cum (muc 2.4,
    Y_global).
    """
    if X_flat.ndim != 2:
        raise ValueError(f"X_flat phai la (T, V), nhan shape {X_flat.shape}")
    if X_flat.shape[1] != S.shape[0]:
        raise ValueError(
            f"X_flat co V={X_flat.shape[1]} nhung S co V={S.shape[0]} - khong khop."
        )
    P = cluster_pooling_matrix(S)  # (C, V)
    return X_flat @ P.T  # (T, C)


def spatial_decompose(Y_fine_flat: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Buoc 2.4 - Cross-scale Spatial Decomp: tach 1 candidate fine (T, V)
    thanh:
        Y_global_cluster:   (T, C) - tin hieu nen TRUNG BINH cua tung cum
        Y_global_broadcast: (T, V) - Y_global_cluster chieu nguoc lai
                                      tung node trong cum (S @ Y_global_cluster.T).T
        Y_local:            (T, V) = Y_fine_flat - Y_global_broadcast
                                      (phan du cuc bo tung node)

    Dung dung mo ta muc 2.4 bao cao: "Tin hieu nen cua Cum (Y_global)" va
    "Phan du cuc bo tung node (Y_local)".
    """
    Y_global_cluster = aggregate_to_clusters(Y_fine_flat, S)  # (T, C)
    Y_global_broadcast = Y_global_cluster @ S.T  # (T, V)
    Y_local = Y_fine_flat - Y_global_broadcast
    return Y_global_cluster, Y_global_broadcast, Y_local


def laplacian_smoothness(z_flat: np.ndarray, W: np.ndarray) -> float:
    """
    Do min Laplacian TAI 1 buoc thoi gian (z_flat shape (V,)):

        Q(z) = sum_ij w_ij (z_i - z_j)^2  /  sum_i d_i z_i^2

    Cong thuc theo muc 2.3 ARCHITECTURE_1_MCTS_SPATIOTEMPORAL.md (gan
    variogram score, Scheuerer & Hamill 2015, Monthly Weather Review).

    Co chu dich KHONG cong epsilon vao mau so: neu z gan hang so (mau ->
    0), Q khong xac dinh ve mat toan hoc (0/0) - tra ve NaN de goi code
    phia tren phai xu ly tuong minh (bo qua buoc thoi gian do), thay vi
    am tham chia epsilon ra 1 con so trong hop ly nhung vo nghia (cung
    triet ly voi canh bao trivial-solution trong era5_physics_energy.py).
    """
    d = W.sum(axis=1)
    denom = float(np.sum(d * z_flat**2))
    if denom < 1e-12:
        return float("nan")
    diff = z_flat[:, None] - z_flat[None, :]
    numer = float(np.sum(W * diff**2))
    return numer / denom


def history_reference_Q(history_local_flat: np.ndarray, W: np.ndarray) -> float:
    """
    Q trung binh tinh tren phan du cuc bo (Y_local) cua LICH SU THAT, dung
    lam tham chieu "muc do min hop ly" - giong cach era5_physics_energy.py
    dung historical_variance lam tham chieu chong trivial-solution.
    """
    T = history_local_flat.shape[0]
    qs = np.array([laplacian_smoothness(history_local_flat[t], W) for t in range(T)])
    qs = qs[~np.isnan(qs)]
    if qs.size == 0:
        raise ValueError(
            "history_reference_Q: toan bo buoc thoi gian deu suy bien "
            "(z gan hang so o moi buoc) - khong tinh duoc tham chieu."
        )
    return float(np.mean(qs))


def spatial_consistency_score(Y_local_flat: np.ndarray, W: np.ndarray, history_Q: float) -> float:
    """
    E_spatial cho 1 candidate: trung binh |Q(z_t) - history_Q| qua cac
    buoc thoi gian t cua Y_local (T, V), bo qua nhung t bi NaN.

    Co chu dich KHONG dung tuong quan thoi gian giua cac node de cham
    phan nay (da quyet dinh trong ARCHITECTURE_1_MCTS_SPATIOTEMPORAL.md
    muc 2.3) - vi neu cac candidate dung chung 1 duong muc quantile,
    tuong quan thoi gian se bi ap dat gia tao boi chinh cach sinh, khong
    phan anh tuong tac vat ly that.
    """
    T = Y_local_flat.shape[0]
    qs = np.array([laplacian_smoothness(Y_local_flat[t], W) for t in range(T)])
    qs = qs[~np.isnan(qs)]
    if qs.size == 0:
        return float("nan")
    return float(np.mean(np.abs(qs - history_Q)))


if __name__ == "__main__":
    from era5_graph import flatten_latlon, build_knn_graph
    from spectral_clustering import spectral_cluster_grid

    np.random.seed(0)

    lat = np.linspace(-90, 90, 8)
    lon = np.linspace(0, 354.375, 16)
    V = lat.size * lon.size
    C = 4
    T = 6

    S, labels, W = spectral_cluster_grid(lat, lon, n_clusters=C, k_neighbors=6)

    # Candidate gia lap: nen phang theo cum (hang so trong 1 cum, khac
    # nhau giua cum) + nhieu nho tung node -> Y_local phai xap xi nhieu do.
    cluster_base = np.random.randn(T, C) * 5.0
    Y_global_true = cluster_base @ S.T  # (T, V)
    noise = 0.1 * np.random.randn(T, V)
    Y_fine = Y_global_true + noise

    Y_global_cluster, Y_global_broadcast, Y_local = spatial_decompose(Y_fine, S)

    print("Y_global_cluster.shape:", Y_global_cluster.shape, "(ky vong (T,C)=", (T, C), ")")
    print("Y_local.shape:", Y_local.shape, "(ky vong (T,V)=", (T, V), ")")

    # Y_global_cluster = cluster_base + trung binh nhieu trong tung cum
    # (khong tuyet doi bang cluster_base vi nhieu khong trung binh ve
    # dung 0 tren mau huu han) - chi kiem tra no GAN cluster_base, dung
    # sai so ty le voi bien do nhieu.
    assert np.allclose(Y_global_cluster, cluster_base, atol=0.15), (
        "Y_global_cluster phai gan cluster_base (sai lech chi den tu trung binh nhieu huu han)"
    )
    recon_err = np.max(np.abs(Y_global_broadcast + Y_local - Y_fine))
    print("Reconstruct max abs error (Y_global_broadcast + Y_local - Y_fine):", recon_err)
    assert recon_err < 1e-10, "Phan ra phai bao toan: global + local = fine"

    # Y_local xap xi noise ban dau (vi cluster_base da bi tru het qua gop dung).
    local_vs_noise = np.max(np.abs(Y_local - (noise - noise.mean(axis=1, keepdims=True) * 0)))
    print("Y_local gan noise goc (sai so nho, do S khong doi xung tuyet doi giua node):", local_vs_noise)

    hist_local = 0.1 * np.random.randn(20, V)  # gia lap phan du lich su
    hist_Q = history_reference_Q(hist_local, W)
    print("\nhistory_reference_Q:", hist_Q)
    assert not np.isnan(hist_Q)

    score_noisy = spatial_consistency_score(Y_local, W, hist_Q)
    print("spatial_consistency_score (candidate nhieu nho, gan lich su):", score_noisy)

    # Candidate "phang bat thuong" (moi node bang nhau tuyet doi trong 1
    # cum va giua cac cum cung gan bang nhau) -> Q(z) -> NaN (0/0) o hau
    # het buoc thoi gian vi z gan hang so tuyet doi -> score phai la NaN
    # hoac rat khac hist_Q, KHONG duoc ngam dinh = 0 (tot gia tao).
    Y_flat_trivial = np.ones((T, V)) * 3.0
    _, _, Y_local_trivial = spatial_decompose(Y_flat_trivial, S)
    score_trivial = spatial_consistency_score(Y_local_trivial, W, hist_Q)
    print("spatial_consistency_score (candidate hang so tuyet doi -> NaN ky vong):", score_trivial)
    assert np.isnan(score_trivial), "Candidate hang so tuyet doi phai cho NaN, khong duoc la 0"

    print("\nOK")
