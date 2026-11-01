# chronos1_multivar — 7 kiến trúc, Chronos-1 đa biến (t2m, u10, v10)

Toàn bộ thư mục này dùng **Chronos-1** (`amazon/chronos-t5-*`, sampling thật qua
`ChronosProposer` trong [../run.py](../run.py)) — **không phải Chronos-2**. Khác với các
script Chronos-2 ở thư mục gốc, ở đây **cả 3 biến khí tượng (t2m, u10, v10) đều được dự
báo bởi chính model**, không còn dùng gió ground-truth tương lai cho Physics Filter.

## Setting chung

| | |
|---|---|
| Val | 2015-01-01 → 2016-12-31 — chạy trước khi sweep: `python3 -c "from down import download_era5_slice_to_local as d; d('era5_val_2015_2016.nc','2015-01-01','2016-12-31')"` (xem [../down.py](../down.py)) |
| Test | 2017-01-01 → 2018-12-31 (`era5_test_2017_2018.nc` |
| Settings đánh giá | (h512,p64), (h512,p128), (h1024,p96) |
| Ngân sách | **N=20** cho mọi kiến trúc (xem cách phân bổ bên dưới) |
| Metric | MSE đã chuẩn hoá (z-score riêng từng biến), trung bình cộng 3 biến — xem `common.compute_multivar_mse` |

## Cơ chế multivariate (Chronos-1, sampling)

Chronos-1 không có cơ chế cho nhiều biến/node tương tác với nhau (khác Chronos-2). Để dự
báo cả 3 biến, hệ thống gọi `ChronosProposer.predict()` **3 lần độc lập** (1 lần/biến),
mỗi lần sinh N kịch bản bằng lấy mẫu ngẫu nhiên thật (autoregressive decode). Kịch bản thứ
*i* của 3 biến được ghép lại thành 1 bộ theo đúng chỉ số *i* — chỉ là quy ước ghép, không
phản ánh liên hệ vật lý thật giữa 3 biến (vì chúng được sinh ra hoàn toàn độc lập). Physics
Filter sau đó dùng chính gió đã dự báo (riêng theo từng candidate) để tính residual, thay
vì ground-truth tương lai như các phiên bản trước.

## Bảng 7 kiến trúc

| Kiến trúc | Verifier | N |
|---|---|---|
| `greedy` | Không có (tham chiếu, temperature=0, 1 candidate/biến) | N=0 |
| `best_of_n_physics` | Chỉ Physics Filter | N=20 |
| `trend_physics` | Chỉ so Trend bằng DTW (tắt Seasonal) | N=20 (15 fine / 5 coarse) |
| `seasonal_physics` | Chỉ so Seasonal bằng FFT (tắt Trend) | N=20 (15 fine / 5 coarse) |
| `trend_seasonal_physics` | Cả Trend (DTW) và Seasonal (FFT) | N=20 (15 fine / 5 coarse) |
| `cluster_physics` | Trend+Seasonal, nhánh Coarse xây bằng Spectral Clustering (tắt phần không gian) | N=20 (15 fine / 5 coarse) |
| `cluster_spatial_physics` | Trend+Seasonal+độ hợp lý không gian (kiến trúc đầy đủ) | N=20 (15 fine / 5 coarse) |

**Phân bổ N cho kiến trúc 2 nhánh (fine/coarse)**: dùng tỉ lệ 3:1 đã thống nhất trước đó
(N_fine=15, N_coarse=5, tổng=20) — có thể đổi qua `--n-fine`/`--n-coarse`.

**Đơn giản hoá có chủ đích**: Physics Filter chỉ áp dụng ở nhánh Fine, không áp dụng ở
nhánh Coarse (nhánh Coarse chỉ dự báo t2m, không dự báo gió ở cấp độ downsample/cụm) — để
tránh phải nhân đôi việc dự báo gió ở cả 2 cấp độ, giữ độ phức tạp trong tầm kiểm soát.

## Cấu trúc thư mục

```
chronos1_multivar/
├── common.py                          # tiện ích dùng chung (data loading, sinh candidate, tính MSE, CSV)
├── engines/
│   ├── trend_seasonal_multivar.py     # dùng cho trend/seasonal/trend_seasonal_physics
│   └── spatial_cluster_multivar.py    # dùng cho cluster_physics / cluster_spatial_physics
├── architectures/                     # 7 file, mỗi file 1 kiến trúc, chạy độc lập được
├── run_all.py                         # chạy hết 7 kiến trúc x 3 setting, ghi 1 CSV tổng hợp
└── reports/                           # CSV output
```

Các module dùng chung (`run.py`, `era5_physics_energy.py`, `dtw.py`, `frequency.py`,
`trend_seasonal_decompose.py`, `spectral_clustering.py`, `era5_graph.py`,
`spatial_decompose.py`, `multiscale.py`) **vẫn nằm ở thư mục gốc**, không di chuyển — mọi
file trong `chronos1_multivar/` tự thêm thư mục gốc vào `sys.path` để import được.

## Cách chạy

### Từng kiến trúc riêng lẻ
```bash
cd chronos1_multivar
python3 architectures/greedy.py --history-steps 512 --prediction-steps 64 --setting h512p64
python3 architectures/best_of_n_physics.py --history-steps 512 --prediction-steps 64 --setting h512p64 --n 20
python3 architectures/trend_physics.py --history-steps 512 --prediction-steps 64 --setting h512p64
python3 architectures/seasonal_physics.py --history-steps 512 --prediction-steps 64 --setting h512p64
python3 architectures/trend_seasonal_physics.py --history-steps 512 --prediction-steps 64 --setting h512p64
python3 architectures/cluster_physics.py --history-steps 512 --prediction-steps 64 --setting h512p64
python3 architectures/cluster_spatial_physics.py --history-steps 512 --prediction-steps 64 --setting h512p64
```
Mỗi lệnh ghi append 1 hàng vào `reports/results_<ten_kien_truc>.csv` riêng.

### Chạy hết 1 lần (khuyến nghị cho chạy thật trên server)
```bash
python3 run_all.py --model-id amazon/chronos-t5-base --n-windows 10
```
Ghi tất cả 7×3=21 hàng vào `reports/results_all.csv` — dùng `--settings` / `--architectures`
để lọc chạy một phần (ví dụ `--settings h512p64 --architectures greedy best_of_n_physics`).

## Backbones (nghiên cứu đa-backbone, Q3 "backbone-agnostic")

`greedy.py` và `best_of_n_physics.py` nhận cờ `--backbone` để đổi foundation model nền.
Các backbone và **cơ chế sinh "1 lịch sử → N tương lai"** (đã verify 2026-10, xem
`proposers.py`):

| `--backbone` | Model mặc định | Output head | Cơ chế N futures | Best-of-N? |
|---|---|---|---|---|
| `chronos`  | `amazon/chronos-t5-base`        | token (categorical) | **autoregressive sampling** (temperature/top-k/top-p) → N trajectory i.i.d. | ✅ |
| `moirai1`  | `Salesforce/moirai-1.1-R-small` | **mixture distribution** | `num_samples=N` draw thật từ mixture | ✅ |
| `moirai2`  | `Salesforce/moirai-2.0-R-small` | quantile (9 mức) | chỉ quantile cố định → **không i.i.d.** | ❌ baseline |
| `timesfm`  | `google/timesfm-2.0-500m-pytorch` | point + quantile | tất định (1 forward) → **không i.i.d.** | ❌ baseline |
| `timesfm3` | `google/timesfm-3.0-pytorch`    | point + quantile | tất định → **không i.i.d.** | ❌ baseline |

→ **Best-of-N chỉ có ý nghĩa với backbone sampler thật** (`chronos`, `moirai1`). Backbone
quantile (`moirai2`/`timesfm*`) chỉ dùng làm **single-pass baseline** qua `greedy.py`; nếu
ép vào `best_of_n_physics.py`, code in CẢNH BÁO và gain kỳ vọng ~0 (chính là bằng chứng cho
luận điểm "TTC-by-sampling cần backbone có sampler thật").

**Lưu ý cài đặt:**
- `moirai1`/`moirai2` cần `uni2ts` + `gluonts`; `timesfm`/`timesfm3` cần `pip install timesfm`.
- **TimesFM đa phiên bản**: `proposers.py` tự nhận diện version từ `model_id` — `2.0` dùng API
  cũ (`TimesFm`+`hparams`+`checkpoint`), `2.5` dùng API mới (`TimesFM_2p5_200M_torch.from_pretrained`
  + `.compile(ForecastConfig)` + `.forecast(horizon=...)`, đã verify từ model card HF). Mặc định
  `--backbone timesfm` là 2.5; muốn 2.0 thì `--model-id google/timesfm-2.0-500m-pytorch`.
- **TimesFM-3.0** (`--backbone timesfm3`) cũng dùng API mới; tên class 3.0 **không hardcode** —
  `proposers.py` tự dò class `*_torch` có `.from_pretrained` trong package `timesfm` đã cài
  (introspection). Nếu `pip install timesfm` của bạn chưa có class 3.0, nó báo lỗi rõ ràng kèm
  danh sách class khả dụng để bạn đối chiếu.
- **Moirai-2.0** có thể cần class `Moirai2Forecast/Moirai2Module` ở bản `uni2ts` mới;
  `proposers.py` tự thử class đó trước rồi fallback `MoiraiForecast/MoiraiModule` — xem log
  in ra để biết đường nào đang dùng.

### Chạy nghiên cứu đa-backbone (1 lệnh)
```bash
python3 run_backbones.py \
  --data-path ./era5_test_2017_2018.nc \
  --climatology-path ./era5_val_2015_2016.nc \
  --history-steps 512 --prediction-steps 64 --setting h512p64 --n-windows 37
# lọc backbone: --backbones chronos moirai1 timesfm
```
Ghi `reports/results_backbones.csv` (cột `architecture` = `<arch>__<backbone>`): greedy chạy
mọi backbone, best_of_n_physics chỉ chạy backbone sampler thật.

### Hoặc chạy lẻ 1 backbone
```bash
# single-pass baseline bất kỳ backbone:
python3 architectures/greedy.py --backbone timesfm --data-path ./era5_test_2017_2018.nc \
  --history-steps 512 --prediction-steps 64 --n-windows 37 --setting h512p64
# Best-of-N (chỉ chronos/moirai1):
python3 architectures/best_of_n_physics.py --backbone moirai1 --n 20 \
  --data-path ./era5_test_2017_2018.nc --history-steps 512 --prediction-steps 64 --n-windows 37
```

## Trạng thái kiểm chứng

Toàn bộ wiring (gồm backbone registry) kiểm thử bằng lưới nhỏ + `amazon/chronos-t5-tiny`
trên CPU, xác nhận không lỗi syntax/import. Nhánh Chronos chạy thật trên GPU T4 (Kaggle).
**Các backbone Moirai/TimesFM chưa chạy thật** — cần cài `uni2ts`/`timesfm` + checkpoint
tương ứng trên server GPU; riêng TimesFM-3.0 cần điền loader (`VERIFY_V3`) trước khi chạy.
