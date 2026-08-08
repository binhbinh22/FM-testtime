import xarray as xr
import gcsfs

def download_test_set_to_local(local_file_path: str = "era5_test_2017_2018.nc"):
    print("1. Kết nối GCS ẩn danh...")
    fs = gcsfs.GCSFileSystem(token='anon')
    zarr_path = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
    ds = xr.open_zarr(fs.get_mapper(zarr_path), consolidated=True)

    print("2. Cắt lát tập Testing (2017-2018)...")
    ds_test = ds.sel(time=slice("2017-01-01", "2018-12-31"))

    print("3. Trích xuất 3 biến vật lý cần thiết để tối ưu dung lượng...")
    ds_subset = ds_test[['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']]

    print(f"4. Đang tải và lưu xuống ổ cứng tại: {local_file_path}")
    print("Quá trình này có thể mất vài phút tùy tốc độ mạng...")
    
    # Lệnh to_netcdf sẽ thực hiện việc kéo dữ liệu từ Cloud và ghi thẳng xuống ổ cứng
    ds_subset.to_netcdf(local_file_path, engine='netcdf4')
    
    print("Đã lưu thành công! Bạn có thể ngắt kết nối mạng ở các lần chạy sau.")

if __name__ == "__main__":
    download_test_set_to_local("era5_test_2017_2018.nc")