# Multiscale Test-Time Search for ERA5 Forecasting

## Mô tả tổng quan

Pipeline mới đã được bổ sung dưới dạng module riêng, không sửa đổi các file gốc [run.py](run.py) và [era5_physics_energy.py](era5_physics_energy.py).

## Những gì đã được xử lý

### 1) Module downsampling theo phong cách TimeMixer
File: [multiscale_search.py](multiscale_search.py)

- Triển khai lớp `MultiscaleTimeMixerDownsampler`.
- Dùng average pooling theo trục thời gian để tạo 3 tầng tỷ lệ:
  - Scale 0: chuỗi gốc / finest (A)
  - Scale 1: downsample 1 lần (B)
  - Scale 2: downsample 2 lần (C)
- Mục đích là tách tín hiệu theo mức độ thô/chi tiết để search ở các tầng khác nhau.

### 2) 3-stage multiscale search
File: [multiscale_search.py](multiscale_search.py)

Pipeline gồm 3 bước:

1. Coarse-scale search (Scale C)
   - Dùng Chronos proposer sinh nhiều kịch bản ở scale thô nhất.
   - Chạy physics verifier để chọn anchor tốt nhất.

2. Intermediate-scale search (Scale B)
   - Dùng history ở scale trung gian.
   - Sinh các candidate ở scale B.
   - Dùng DTW consistency check so với anchors ở scale C để lọc candidate.
   - Sau đó dùng physics verifier để chọn anchor ở scale B.

3. Fine-scale search (Scale A)
   - Dùng history gốc ở scale finest.
   - Sinh các candidate ở scale A.
   - Dùng DTW consistency check so với anchors ở scale B.
   - Cuối cùng dùng physics verifier để chọn 1 trajectory cuối cùng.

### 3) Wrapper chạy trên pipeline cũ
File: [multiscale_runner.py](multiscale_runner.py)

- Tái sử dụng lớp `ERA5DataLoader`, `ChronosProposer` và `ERA5PhysicsEnergyScorer` hiện có.
- Tạo một evaluator mới tên `MultiscaleERA5Evaluator` để chạy pipeline multiscale mà không đụng vào [run.py](run.py).

## Output cuối cùng

Pipeline hiện tại trả về một dictionary gồm:

- `final_candidate`: trajectory cuối cùng được chọn
- `final_index`: chỉ số của candidate cuối cùng
- `metrics`: gồm
  - `rmse_norm`
  - `rmse_denorm`
- Các stage trung gian:
  - `coarse_stage`
  - `intermediate_stage`
  - `fine_stage`
  - `coarse_anchors`
  - `intermediate_anchors`

## Visualization

Có hỗ trợ visualization nhưng hiện tại pipeline mới chưa tự sinh các file plot như [run.py](run.py) vì mục tiêu ban đầu là tích hợp module multiscale trước. Nếu cần, có thể bổ sung ngay một hàm plot riêng để lưu:

- `multiscale_candidates.png`
- `multiscale_selected.png`
- `multiscale_comparison.png`

## Cách chạy

Sử dụng Slurm như hiện tại:

```bash
sbatch run.sh
```

Script [run.sh](run.sh) đã được cấu hình để chạy [multiscale_runner.py](multiscale_runner.py) trong môi trường conda `stllm`.

## Ghi chú

- Pipeline đã được viết để giữ nguyên cấu trúc cũ và tận dụng lại các class hiện có.
- Đã kiểm tra cú pháp bằng lệnh:

```bash
python -m py_compile run.py era5_physics_energy.py multiscale_search.py multiscale_runner.py
```
