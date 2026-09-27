import numpy as np
from typing import Optional, Tuple

from sklearn.cluster import SpectralClustering

from era5_graph import flatten_latlon, build_knn_graph


def spectral_cluster_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    n_clusters: int,
    k_neighbors: int = 8,
    sigma_km: Optional[float] = None,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Buoc 2.1 cua kien truc: chay Spectral Clustering 1 LAN tren cau truc
    do thi (lat, lon), sinh ma tran gan S (V x C).

    S[i, c] = 1 neu node i thuoc cum c, nguoc lai 0 (one-hot cung, KHONG
    phai soft assignment - dung dung nghia "ma tran gan S" trong bao cao).

    Clustering CHI dua tren cau truc khong gian cua luoi (khoang cach
    great-circle qua era5_graph.py), KHONG dua tren gia tri t2m/gio - day
    la lua chon co chu dich: cum dia ly on dinh qua moi cua so du bao,
    tinh 1 lan roi dung lai, khong phai tinh lai moi lan chay pipeline.

    TODO (mo, chua chot - xem ARCHITECTURE_1_MCTS_SPATIOTEMPORAL.md muc
    2.3): ban day du de xuat loc bang heat kernel K=exp(-tau*L) TRUOC khi
    gop cum de tranh aliasing, va vung dem (halo) o bien cum. Ban nay
    CHUA lam 2 dieu do - chi cluster thuan tren affinity kNN Gaussian.

    Return:
        S: (V, C) one-hot cluster assignment
        labels: (V,) nhan cum moi node (0..C-1)
        W: (V, V) do thi kNN da dung de cluster - TAI SU DUNG cho
           spatial consistency score sau nay (khong build lai do thi
           2 lan, dam bao cung 1 cau truc khong gian xuyen suot pipeline)
    """
    lat_flat, lon_flat = flatten_latlon(lat, lon)
    W = build_knn_graph(lat_flat, lon_flat, k=k_neighbors, sigma_km=sigma_km)

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    )
    labels = clustering.fit_predict(W)

    V = W.shape[0]
    S = np.zeros((V, n_clusters), dtype=np.float64)
    S[np.arange(V), labels] = 1.0

    cluster_sizes = S.sum(axis=0)
    empty_clusters = np.where(cluster_sizes == 0)[0]
    if empty_clusters.size > 0:
        raise RuntimeError(
            f"Spectral Clustering sinh ra cum rong: {empty_clusters.tolist()} "
            f"(n_clusters={n_clusters} co the qua lon so voi cau truc do thi "
            f"hien co k_neighbors={k_neighbors} - thu giam n_clusters hoac "
            "tang k_neighbors)."
        )

    return S, labels, W


def cluster_pooling_matrix(S: np.ndarray) -> np.ndarray:
    """
    P = D_S^{-1} S^T  (C x V) - phep trung binh TRONG TUNG CUM, dung de
    gop X (T,V) -> X_c (T,C) = X @ P^T. Dung ky hieu voi muc 2.3
    ARCHITECTURE_1_MCTS_SPATIOTEMPORAL.md (X_c = P . X).
    """
    cluster_size = S.sum(axis=0)  # (C,)
    if np.any(cluster_size == 0):
        raise ValueError("cluster_pooling_matrix: co cum rong trong S.")
    P = (S / cluster_size[None, :]).T  # (C, V)
    return P


if __name__ == "__main__":
    np.random.seed(0)

    # Luoi nho gia lap (khong phai luoi that) chi de kiem tra wiring.
    lat = np.linspace(-90, 90, 8)
    lon = np.linspace(0, 354.375, 16)
    n_clusters = 4

    S, labels, W = spectral_cluster_grid(lat, lon, n_clusters=n_clusters, k_neighbors=6)

    V = lat.size * lon.size
    print("V =", V, "| S.shape =", S.shape, "| C =", n_clusters)
    assert S.shape == (V, n_clusters)
    assert np.allclose(S.sum(axis=1), 1.0), "moi node phai thuoc dung 1 cum"

    cluster_sizes = S.sum(axis=0)
    print("Kich thuoc tung cum:", cluster_sizes.astype(int).tolist())

    P = cluster_pooling_matrix(S)
    print("P.shape:", P.shape)

    # X_c = X @ P^T phai la trung binh dung trong tung cum.
    X = np.random.randn(5, V)  # (T=5, V)
    X_c = X @ P.T  # (T, C)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        expected = X[:, idx].mean(axis=1)
        assert np.allclose(X_c[:, c], expected), f"cum {c} trung binh sai"
    print("Kiem tra trung binh cum: OK")

    print("\nGuard khi n_clusters qua lon so voi V (test): ", end="")
    try:
        spectral_cluster_grid(lat, lon, n_clusters=200, k_neighbors=6)
        print("KHONG raise - loi!")
    except (RuntimeError, ValueError):
        print("raise dung nhu ky vong (n_clusters > V hoac cum rong)")
    print("OK")
