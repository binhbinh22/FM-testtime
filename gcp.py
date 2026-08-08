import xarray as xr
import gcsfs

def load_era5_zarr_public(gcs_path: str) -> xr.Dataset:
    """
    Hàm nạp dữ liệu ERA5 Zarr từ public bucket sử dụng quyền ẩn danh.
    """
    # Khởi tạo FileSystem ẩn danh để bỏ qua lỗi xác thực gcloud
    fs = gcsfs.GCSFileSystem(token='anon')
    
    # Tạo mapper để xarray hiểu cách đọc qua gcsfs
    mapper = fs.get_mapper(gcs_path)
    
    try:
        # Consolidated=True giúp đọc metadata nhanh hơn rất nhiều
        dataset = xr.open_zarr(mapper, consolidated=True)
        return dataset
    except KeyError:
        dataset = xr.open_zarr(mapper, consolidated=False)
        return dataset

if __name__ == "__main__":
    # ĐƯỜNG DẪN CHUẨN (Sử dụng gs:// thay vì https://)
    # Trỏ đúng vào thư mục độ phân giải 64x32 theo thiết lập của bài báo
    zarr_dir = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
    
    print("Đang kết nối tới Google Cloud Storage...")
    ds = load_era5_zarr_public(zarr_dir)
    
    # In ra thông tin tổng quan
    print("Kết nối thành công! Cấu trúc Dataset:\n", ds)
    
    # Trích xuất riêng biến nhiệt độ 2m (t2m)
    if '2m_temperature' in ds.data_vars:
        t2m_data = ds['2m_temperature']
        print(f"\nShape của mảng 2m_temperature: {t2m_data.shape}")
    else:
        print("\nKhông tìm thấy biến '2m_temperature' trong tập dữ liệu.")