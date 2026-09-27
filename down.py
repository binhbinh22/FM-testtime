import xarray as xr
import gcsfs

# Re-export: frequency.py va multiscale_trend_seasonal_search.py dang lam
# `from down import downsample_avgpool`, nhung ham nay thuc su dinh nghia
# o multiscale.py - khong co trong down.py ban goc (import se ImportError).
# Re-export tai day de khong phai sua lai 2 file kia.
from multiscale import downsample_avgpool  # noqa: F401


def download_era5_slice_to_local(
    local_file_path: str = "era5_test_2017_2018.nc",
    start_date: str = "2017-01-01",
    end_date: str = "2018-12-31",
):
    """
    Tai 1 slice thoi gian tuy chon tu kho ERA5/WeatherBench2 (1959-2022) ve
    local. Dung ham nay de tao ca tap TEST (2017-2018, mac dinh) lan tap
    VALIDATION (mot giai doan KHAC, khong giao voi test) dung de sweep
    hyperparameter (N, K, w1, w2, ...) ma khong lam ro ri thong tin tu tap
    test - chi chon (N, K) tot nhat tren val, roi moi chay 1 lan tren test
    de bao cao ket qua cuoi cung.
    """
    print("1. Kết nối GCS ẩn danh...")
    fs = gcsfs.GCSFileSystem(token='anon')
    zarr_path = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
    ds = xr.open_zarr(fs.get_mapper(zarr_path), consolidated=True)

    print(f"2. Cắt lát dữ liệu ({start_date} -> {end_date})...")
    ds_slice = ds.sel(time=slice(start_date, end_date))

    print("3. Trích xuất 3 biến vật lý cần thiết để tối ưu dung lượng...")
    ds_subset = ds_slice[['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']]

    print(f"4. Đang tải và lưu xuống ổ cứng tại: {local_file_path}")
    print("Quá trình này có thể mất vài phút tùy tốc độ mạng...")

    # Lệnh to_netcdf sẽ thực hiện việc kéo dữ liệu từ Cloud và ghi thẳng xuống ổ cứng
    ds_subset.to_netcdf(local_file_path, engine='netcdf4')

    print("Đã lưu thành công! Bạn có thể ngắt kết nối mạng ở các lần chạy sau.")


def download_test_set_to_local(local_file_path: str = "era5_test_2017_2018.nc"):
    """Giu nguyen cho tuong thich nguoc: tai dung tap TEST 2017-2018."""
    download_era5_slice_to_local(local_file_path, start_date="2017-01-01", end_date="2018-12-31")


if __name__ == "__main__":
    download_test_set_to_local("era5_test_2017_2018.nc")