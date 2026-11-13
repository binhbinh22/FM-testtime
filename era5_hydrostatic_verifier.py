import numpy as np

"""
Verifier Hydrostatic Balance cho ERA5 (phuong trinh can bang thuy tinh,
nguon: DeepPrim ICLR 2026 App. C.5; Holton & Hakim 2013; Vallis 2017).

Dang tich phan (hypsometric) giua 2 muc ap suat 1000 va 500 hPa:
    phi(500) - phi(1000) = R_dry * T_bar * ln(1000/500)
voi T_bar = nhiet do TRUNG BINH cua lop 1000-500 hPa, xap xi bang t850
(850 hPa nam gan giua lop). R_dry = 287 J/(kg*K).

Residual (cang nho cang thoa vat ly):
    r = (z500 - z1000) - R_dry * t850 * ln(2)

KHONG train gi: cong thuc dong, hang so vat ly. Dung de CHAM candidate
(z500, z1000, t850) do FMTS sinh ra - candidate nao vi pham can bang thuy
tinh nhieu (|r| lon) thi it hop ly ve vat ly.

LUU Y PHUONG PHAP (bai hoc L_smooth): cong thuc DUNG tren ground-truth
KHONG dam bao no PHAN BIET duoc candidate tot/xau. Phai do truc tiep
corr(|r| tren candidate, MSE that cua candidate) trbefore khi tin dung lam
selector (xem kernel correlation-check).
"""

R_DRY = 287.0          # J/(kg*K)
LN2 = float(np.log(2.0))


def hydrostatic_residual(z500: np.ndarray, z1000: np.ndarray, t850: np.ndarray) -> np.ndarray:
    """
    Residual can bang thuy tinh, element-wise (giu nguyen shape dau vao).
    z500, z1000: geopotential (phi, m^2/s^2); t850: nhiet do (K).
    """
    return (z500 - z1000) - R_DRY * t850 * LN2


def hydrostatic_score(z500: np.ndarray, z1000: np.ndarray, t850: np.ndarray) -> float:
    """
    Diem cho 1 candidate (scalar, cang THAP cang tot) = MSE cua residual.
    z500/z1000/t850: 1 truong du bao, shape bat ky (vd (pred, lat, lon)).
    """
    r = hydrostatic_residual(z500, z1000, t850)
    return float(np.mean(r ** 2))


if __name__ == "__main__":
    import xarray as xr

    # Kiem tra cong thuc tren GROUND-TRUTH that: residual phai NHO & on dinh
    ds = xr.open_dataset("era5_test_2017_2018_hydro.nc")
    z500 = ds["z500"].values
    z1000 = ds["z1000"].values
    t850 = ds["t850"].values
    print("shape:", z500.shape, "| z500 ~", z500.mean(), "| z1000 ~", z1000.mean(), "| t850 ~", t850.mean())

    thickness = z500 - z1000                      # do day lop 1000-500 (phi)
    rhs = R_DRY * t850 * LN2                       # ve phai hypsometric
    r = thickness - rhs

    print(f"\nThickness (z500-z1000) trung binh : {thickness.mean():10.1f} m^2/s^2")
    print(f"R*t850*ln2             trung binh : {rhs.mean():10.1f} m^2/s^2")
    print(f"Residual r             trung binh : {r.mean():10.1f}  (bias he thong do t850 != T_bar)")
    print(f"Residual r             std        : {r.std():10.1f}")
    print(f"|r|/thickness          trung binh : {np.abs(r).mean()/thickness.mean()*100:6.2f}%")
    print("\n=> Neu |r| nho hon NHIEU so voi thickness (vai %), cong thuc implement DUNG.")
    print("   (Bias he thong la do t850 xap xi T_bar, khong phai loi code.)")
