import numpy as np
from typing import Optional, Tuple

EARTH_RADIUS_KM = 6371.0


def flatten_latlon(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grid lat (ny,), lon (nx,) -> lat_flat, lon_flat moi cai (V,) voi
    V = ny*nx. Thu tu phang PHAI khop voi thu tu dung de flatten du lieu
    (history.reshape(T, ny*nx) voi indexing "ij": lat ngoai, lon trong) -
    dung cung quy uoc voi ChronosProposer.predict trong run.py (reshape
    (history_steps, lat, lon) -> (lat*lon, history_steps)).
    """
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")  # (ny, nx)
    return lat_grid.reshape(-1), lon_grid.reshape(-1)


def haversine_distance_matrix(lat_flat: np.ndarray, lon_flat: np.ndarray) -> np.ndarray:
    """
    Khoang cach great-circle (km) giua MOI cap trong V diem luoi.

    Dung haversine thay vi Euclid tho tren (lat,lon) vi luoi ERA5
    (equiangular_with_poles, da xac nhan tren du lieu that bang
    inspect_era5.py) co 2 diem cuc (lat=-90/+90) noi MOI kinh do hoi tu ve
    CUNG 1 vi tri vat ly. Euclid tren (lat,lon) tho se coi 64 diem o cuc
    la cach xa nhau tuy lon, trong khi thuc te chung trung nhau - anh
    huong truc tiep den chat luong Spectral Clustering va do min Laplacian
    o cac cum gan cuc.

    Return: (V, V) km, doi xung, duong cheo = 0.
    """
    lat_r = np.radians(lat_flat)
    lon_r = np.radians(lon_flat)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def build_knn_graph(
    lat_flat: np.ndarray,
    lon_flat: np.ndarray,
    k: int = 8,
    sigma_km: Optional[float] = None,
) -> np.ndarray:
    """
    Do thi k-nearest-neighbor tren khoang cach great-circle, trong so
    Gaussian kernel: w_ij = exp(-d_ij^2 / (2*sigma^2)) neu j la 1 trong k
    lang gieng gan nhat cua i. Symmetrize bang max(W, W^T) (ke neu 1
    trong 2 chieu coi la kNN cua nhau - quy uoc chuan cho kNN graph
    khong huong).

    sigma mac dinh = trung vi khoang cach toi lang gieng thu k (self-
    tuning kieu Zelnik-Manor & Perona 2004) neu khong truyen vao, de
    trong so tu dong thich nghi voi mat do luoi (khong dong deu o cac vi
    vi do cua luoi equiangular).

    Return: W (V, V), khong am, duong cheo = 0.
    """
    if k < 1:
        raise ValueError(f"k phai >= 1, nhan {k}")

    dist = haversine_distance_matrix(lat_flat, lon_flat)
    V = dist.shape[0]
    if k >= V:
        raise ValueError(f"k={k} phai < so node V={V}")

    order = np.argsort(dist, axis=1)
    knn_idx = order[:, 1 : k + 1]  # bo chinh no (cot 0, dist=0)

    if sigma_km is None:
        kth_dist = np.take_along_axis(dist, knn_idx[:, -1:], axis=1)
        sigma_km = float(np.median(kth_dist))
        sigma_km = max(sigma_km, 1e-6)

    W = np.zeros((V, V), dtype=np.float64)
    rows = np.repeat(np.arange(V), k)
    cols = knn_idx.reshape(-1)
    w = np.exp(-(dist[rows, cols] ** 2) / (2.0 * sigma_km**2))
    W[rows, cols] = w
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


def graph_laplacian(W: np.ndarray, normalized: bool = False) -> np.ndarray:
    """L = D - W (combinatorial), hoac normalized L = D^-1/2 (D-W) D^-1/2."""
    d = W.sum(axis=1)
    L = np.diag(d) - W
    if not normalized:
        return L
    d_safe = np.where(d < 1e-12, 1.0, d)
    d_inv_sqrt = 1.0 / np.sqrt(d_safe)
    return np.diag(d_inv_sqrt) @ L @ np.diag(d_inv_sqrt)


if __name__ == "__main__":
    lat = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])

    lat_flat, lon_flat = flatten_latlon(lat, lon)
    print("V =", lat_flat.size, "(ky vong", lat.size * lon.size, ")")

    dist = haversine_distance_matrix(lat_flat, lon_flat)
    print("dist shape:", dist.shape)
    print("dist max (km, ky vong ~ nua chu vi Trai Dat ~20015km):", dist.max())

    # Kiem tra tinh chat cuc: moi diem o lat=-90 phai cach nhau 0km du lon khac nhau
    pole_mask = np.isclose(lat_flat, -90.0)
    pole_idx = np.where(pole_mask)[0]
    print("\nSo diem o cuc nam (lat=-90):", pole_idx.size, "(ky vong", lon.size, ")")
    pole_dist = dist[np.ix_(pole_idx, pole_idx)]
    print("Khoang cach giua cac diem cuc (phai ~0 het):", pole_dist.max())
    assert pole_dist.max() < 1e-6, "Cac diem cung 1 cuc phai trung vi tri (dist=0)"

    W = build_knn_graph(lat_flat, lon_flat, k=3)
    print("\nW shape:", W.shape, "| symmetric:", np.allclose(W, W.T))
    print("W nnz mean degree:", (W > 0).sum(axis=1).mean())

    L = graph_laplacian(W)
    eigvals = np.linalg.eigvalsh(L)
    print("\nLaplacian eigenvalues nho nhat (ky vong >=0, it nhat 1 gia tri ~0):")
    print(np.round(eigvals[:5], 6))
    assert eigvals.min() > -1e-8, "Laplacian phai positive semi-definite"
    print("OK")
