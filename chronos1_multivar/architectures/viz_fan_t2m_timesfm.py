import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

"""
Visualize 1 lich su t2m -> N candidate tuong lai sinh boi TimesFM, theo dung
Algorithm 3 (Best-of-N Sample) nhanh "khong co sampler that" (quantile head):

    Fix dense quantile grid G
    for i = 1..N-1:
        u_{i,t} ~ U(0,1) i.i.d. tung buoc thoi gian t
        y_i[t] = F_theta^{-1}(u_{i,t} | t, G)      # noi suy nghich dao CDF
    Luon them du bao median (muc 0.5) la 1 candidate trong N.

TimesFM KHONG ho tro autoregressive sampling that (sampler="quantile", xem
chronos1_multivar/proposers.py) - moi forward pass chi ra 1 point forecast +
luoi quantile co dinh. File nay KHONG sua proposers.py (ham .predict() o do
dang vut bo quantile, chi tra ve 1 forecast tat dinh) - viet rieng, goi thang
model.forecast() de lay du quantile, roi tu lam buoc sampling theo Algorithm 3.

Du lieu: ERA5 (vectai giong viz_fan_t2m.py, chi doi backbone Chronos -> TimesFM).
"""

DEFAULT_QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(
        description="Ve fan chart N candidate t2m tu TimesFM (Algorithm 3, quantile-head "
                     "sampling) tren du lieu ERA5, kem ground truth."
    )
    p.add_argument("--data-path", type=str, default="era5_test_2017_2018.nc")
    p.add_argument("--target-var", type=str, default="2m_temperature")
    p.add_argument("--history-steps", type=int, default=512)
    p.add_argument("--prediction-steps", type=int, default=64)
    p.add_argument("--model-id", type=str, default="google/timesfm-2.5-200m-pytorch")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--lat-idx", type=int, default=16)
    p.add_argument("--lon-idx", type=int, default=32)
    p.add_argument("--history-tail", type=int, default=96,
                   help="So buoc lich su ve ra truoc diem du bao (chi de do thi khong qua dai).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="fan_chart_t2m_timesfm.png")
    return p.parse_args()


def detect_version(model_id: str) -> str:
    m = model_id.lower()
    if "2.0" in m or "2p0" in m:
        return "2.0"
    if "3.0" in m or "3p0" in m or "timesfm-3" in m or "timesfm3" in m:
        return "3.0"
    return "2.5"  # mac dinh new-style


def find_new_style_class(timesfm_mod, version: str):
    """Tu do class loader new-style (*_torch co .from_pretrained) trong package `timesfm`."""
    tag = version.replace(".", "p")
    candidates = [
        n for n in dir(timesfm_mod)
        if "torch" in n.lower() and hasattr(getattr(timesfm_mod, n), "from_pretrained")
    ]
    for n in candidates:
        if tag in n.lower().replace("_", ""):
            return getattr(timesfm_mod, n), n
    if candidates:
        return getattr(timesfm_mod, candidates[0]), candidates[0]
    return None, None


def build_timesfm(model_id: str, device: str, history_steps: int, prediction_steps: int):
    """
    Nap model TimesFM, ho tro ca API cu (2.0: TimesFm + .forecast(inputs, freq))
    va API moi (2.5/3.0: *_torch.from_pretrained + .compile + .forecast(horizon=, inputs=)).
    """
    import timesfm

    version = detect_version(model_id)
    api = "old" if version == "2.0" else "new"
    print(f"[TimesFM] model_id={model_id} version={version} api={api} device={device}")

    if api == "old":
        backend = "gpu" if device == "cuda" else "cpu"
        hparams = timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=32,
            horizon_len=max(prediction_steps, 128),
            context_len=min(max(history_steps, 512), 2048),
            num_layers=50,
            use_positional_embedding=False,
        )
        ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id=model_id)
        model = timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)
        return model, api, version

    cls, cls_name = find_new_style_class(timesfm, version)
    if cls is None:
        available = [n for n in dir(timesfm) if "torch" in n.lower()]
        raise RuntimeError(
            f"Khong tim thay class loader new-style cho TimesFM {version}. "
            f"Cac ten *_torch co san: {available}. Cap nhat `pip install -U timesfm`."
        )
    print(f"[TimesFM] new-style class: timesfm.{cls_name}")
    model = cls.from_pretrained(model_id)
    try:
        model.compile(
            timesfm.ForecastConfig(
                max_context=min(max(history_steps, 512), 2048),
                max_horizon=max(prediction_steps, 256),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
    except TypeError as e:
        print(f"[TimesFM] .compile() khong ap dung cho {cls_name} ({e}); bo qua.")
    return model, api, version


def forecast_point_and_quantiles(model, api: str, history_1d: np.ndarray, prediction_steps: int):
    """
    Goi TimesFM cho DUY NHAT 1 chuoi (1 diem luoi lat/lon), tra ve:
      point           : (H,)
      quantile         : (H, 9)  - gia tri tai 9 muc co dinh [0.1..0.9]
      quantile_levels  : (9,)    - DEFAULT_QUANTILE_LEVELS

    API moi (2.5/3.0) tra ve quantile shape (H, 10) voi COT 0 LA MEAN (khong
    phai quantile), cot 1..9 la muc 0.1..0.9 - theo dung model card HuggingFace
    cua TimesFM 2.5 (da duoc ghi chu/verify trong chronos1_multivar/proposers.py).
    """
    inputs = [history_1d.astype(np.float32)]
    if api == "old":
        point, q = model.forecast(inputs, freq=[0])
        q = np.asarray(q)[0, :prediction_steps, :]  # (H, 9)
    else:
        point, q = model.forecast(horizon=prediction_steps, inputs=inputs)
        q = np.asarray(q)[0, :prediction_steps, :]  # (H, 10) hoac (H, 9)
        if q.shape[-1] == 10:
            q = q[:, 1:]  # bo cot 0 (mean), giu 9 cot quantile 0.1..0.9

    if q.shape[-1] != len(DEFAULT_QUANTILE_LEVELS):
        raise RuntimeError(
            f"So cot quantile tra ve ({q.shape[-1]}) khac voi gia dinh "
            f"({len(DEFAULT_QUANTILE_LEVELS)} muc). Kiem tra lai phien ban TimesFM dang cai "
            f"va cap nhat DEFAULT_QUANTILE_LEVELS/logic bo cot mean cho dung."
        )

    point = np.asarray(point, dtype=np.float32)[0, :prediction_steps]
    return point, q.astype(np.float32), DEFAULT_QUANTILE_LEVELS


def sample_best_of_n(quantile_forecast: np.ndarray, quantile_levels: np.ndarray,
                      n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Algorithm 3 (Best-of-N Sample), nhanh quantile head (khong co sampler that):
      - Co dinh luoi quantile G = quantile_levels.
      - Voi i = 1..N-1: u_{i,t} ~ U(0,1) i.i.d. TUNG BUOC THOI GIAN,
        y_i[t] = F_theta^{-1}(u_{i,t} | t, G) bang noi suy tuyen tinh tren luoi quantile.
      - Luon them du bao median (muc 0.5) la 1 candidate (dong 11 trong Algorithm 3).

    quantile_forecast: (H, Q)
    Tra ve: (n_samples, H) - (n_samples-1) candidate ngau nhien + 1 candidate median cuoi cung.
    """
    H, _Q = quantile_forecast.shape
    n_random = max(n_samples - 1, 0)

    u = rng.uniform(0.0, 1.0, size=(n_random, H)).astype(np.float32)
    samples = np.empty((n_random, H), dtype=np.float32)
    for t in range(H):
        samples[:, t] = np.interp(u[:, t], quantile_levels, quantile_forecast[t, :])

    median = np.empty((H,), dtype=np.float32)
    for t in range(H):
        median[t] = np.interp(0.5, quantile_levels, quantile_forecast[t, :])

    return np.concatenate([samples, median[np.newaxis, :]], axis=0)  # (n_samples, H)


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = xr.open_dataset(args.data_path).transpose("time", "latitude", "longitude")
    h_start, h_end = args.start_idx, args.start_idx + args.history_steps
    g_end = h_end + args.prediction_steps

    series_full = ds[args.target_var].values  # (time, lat, lon)
    history_1d = series_full[h_start:h_end, args.lat_idx, args.lon_idx]
    gt_1d = series_full[h_end:g_end, args.lat_idx, args.lon_idx]

    model, api, version = build_timesfm(
        args.model_id, device, args.history_steps, args.prediction_steps
    )
    point, quantile_forecast, quantile_levels = forecast_point_and_quantiles(
        model, api, history_1d, args.prediction_steps
    )

    candidates = sample_best_of_n(quantile_forecast, quantile_levels, args.n_samples, rng)  # (N, H)

    history_tail_1d = history_1d[-args.history_tail:]
    history_x = np.arange(-len(history_tail_1d), 0)
    future_x = np.arange(0, args.prediction_steps)

    # Noi lien mach: ghep diem cuoi lich su lam diem dau cho moi duong tuong lai,
    # de matplotlib ve lien 1 net thay vi 2 doan tach roi tai x=0.
    connect_x = np.concatenate([history_x[-1:], future_x])
    last_history = history_tail_1d[-1]

    plt.figure(figsize=(11, 5))
    plt.plot(history_x, history_tail_1d, color="black", linewidth=1.8, label="Lich su (history)")
    for i in range(args.n_samples):
        plt.plot(connect_x, np.concatenate([[last_history], candidates[i]]),
                  color="tab:green", alpha=0.25, linewidth=1.0)
    plt.plot(connect_x, np.concatenate([[last_history], candidates.mean(axis=0)]),
              color="tab:green", linewidth=2.0,
              label=f"Trung binh {args.n_samples} candidate (TimesFM, quantile-sampling)")
    plt.plot(connect_x, np.concatenate([[last_history], point]),
              color="tab:orange", linewidth=1.5, linestyle="-.",
              label="Point forecast (TimesFM, tat dinh)")
    plt.plot(connect_x, np.concatenate([[last_history], gt_1d]),
              color="tab:red", linewidth=1.8, linestyle="--", label="Ground truth")
    plt.axvline(0, color="gray", linestyle=":", linewidth=1.0)
    plt.xlabel("Buoc thoi gian (0 = thoi diem hien tai)")
    plt.ylabel(f"{args.target_var} (K)")
    plt.title(
        f"TimesFM ({version}): 1 lich su -> {args.n_samples} candidate "
        f"(Algorithm 3, quantile-head sampling) - lat={args.lat_idx}, lon={args.lon_idx}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Da luu hinh -> {args.out}")
