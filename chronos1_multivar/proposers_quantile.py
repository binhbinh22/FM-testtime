import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from proposers import (  # noqa: E402
    MoiraiProposer, TimesFMProposer, get_proposer, resolve_model_id,
)
from quantile_sampling import algo3_candidates_from_quantiles  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Mo rong da-backbone cho runner hop nhat: them cac model QUANTILE-HEAD
(Chronos-2, Moirai-2.0, TimesFM-2.x/3) chay qua Algorithm 3 (inverse-CDF
sampling, xem quantile_sampling.py) de sinh N candidate da dang - thay vi
chi 1 point forecast lap lai (nhu proposers.py goc, supports_bon=False).

KHONG sua proposers.py goc: cac proposer quantile o day KE THUA/BOC THEM
method `predict_quantiles(...)` - tra ve (q_values, q_levels, mean, std).

====================== CAN VERIFY TREN ENV THAT ======================
Chua chay (theo yeu cau). Cac diem phu thuoc API/phien ban model, danh dau
[VERIFY] - chay thu 1 batch nho tren Kaggle/cluster roi doi chieu shape:
  - Chronos-2 : BaseChronosPipeline.predict_quantiles(context, prediction_length,
                quantile_levels) -> (quantiles[B,pred,n_q], mean[B,pred]).
                User da xac nhan API predict_df co quantile_levels; predict_quantiles
                la ban tensor tuong duong tren BaseChronosPipeline. [VERIFY ten method]
  - TimesFM   : self.tfm.forecast(...) tra ve (point, quantile[B,pred,n_q]).
                Thu tu/so muc quantile MAC DINH cua TimesFM = [VERIFY] (gia su
                [0.1..0.9], 9 muc). Neu ban TimesFM tra 10 cot (co mean) -> chinh.
  - Moirai-2  : dung gluonts forecast.quantile(level) cho tung muc. [VERIFY]
                num_samples anh huong do min cua quantile uoc luong tu mau.
=====================================================================
"""

# Muc quantile mac dinh cho inverse-CDF (dense grid G trong Algorithm 3).
DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

QUANTILE_BACKBONES = {"chronos2", "moirai2", "timesfm", "timesfm3"}
NATIVE_BACKBONES = {"chronos", "moirai1"}


def _zscore_stats(history):
    """history: (T, lat, lon) -> context_2d (n_series, T), mean/std (n_series, 1)."""
    T, lat, lon = history.shape
    n_series = lat * lon
    context_2d = history.reshape(T, n_series).T.astype(np.float32)
    mean = np.mean(context_2d, axis=1, keepdims=True)
    std = np.std(context_2d, axis=1, keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    return context_2d, mean, std, lat, lon, n_series


# ============================================================
# Chronos-2 (quantile-head moi, BaseChronosPipeline)
# ============================================================
class Chronos2Proposer:
    """
    Chronos-2 qua `chronos.BaseChronosPipeline` (user da xac nhan API). La
    model quantile -> dung predict_quantiles() lay luoi quantile cho Algorithm 3.
    Co them predict() (tra median broadcast) de tuong thich greedy / coarse-anchor.
    """

    def __init__(self, model_id: str = "amazon/chronos-2", device: str = "cuda", batch_size: int = 256):
        import torch
        from chronos import BaseChronosPipeline

        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        print(f"Initializing Chronos2Proposer ({model_id}) on {self.device}...")
        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=self.device)
        print("Chronos-2 loaded.")

    def predict_quantiles(self, history, prediction_steps, quantile_levels):
        import torch

        context_2d, mean, std, lat, lon, n_series = _zscore_stats(history)
        levels = list(quantile_levels)
        q_chunks = []
        for start in range(0, n_series, self.batch_size):
            end = min(start + self.batch_size, n_series)
            ctx = [torch.tensor(context_2d[i], dtype=torch.float32) for i in range(start, end)]
            # [VERIFY] BaseChronosPipeline.predict_quantiles -> (quantiles[B,pred,n_q], mean[B,pred])
            quantiles, _m = self.pipeline.predict_quantiles(
                ctx, prediction_length=prediction_steps, quantile_levels=levels,
            )
            q_chunks.append(np.asarray(quantiles, dtype=np.float32))
        q_values = np.concatenate(q_chunks, axis=0)  # (n_series, pred, n_q) DENORM
        return q_values, np.asarray(levels), mean, std

    def predict(self, history, prediction_steps, temperature=1.0, n_samples=1):
        """Greedy / coarse-anchor: tra ve median (quantile 0.5) broadcast n_samples."""
        q_values, levels, mean, std = self.predict_quantiles(history, prediction_steps, [0.5])
        median = q_values[:, :, 0]  # (n_series, pred) denorm
        n = max(int(n_samples), 1)
        denorm = np.repeat(median[:, None, :], n, axis=1)  # (n_series, n, pred)
        return _reshape_series_to_grid(denorm, mean, std, history, prediction_steps, n)


# ============================================================
# TimesFM-2.x / 3 (quantile-head) - ke thua TimesFMProposer, them predict_quantiles
# ============================================================
class TimesFMQuantileProposer(TimesFMProposer):
    # [VERIFY] muc quantile mac dinh TimesFM tra ve (gia su 9 muc 0.1..0.9).
    TFM_QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def predict_quantiles(self, history, prediction_steps, quantile_levels=None):
        context_2d, mean, std, lat, lon, n_series = _zscore_stats(history)
        inputs = [context_2d[i] for i in range(n_series)]
        if self.api == "old":
            freq = [0] * len(inputs)
            _point, quantile = self.tfm.forecast(inputs, freq=freq)
        else:
            _point, quantile = self.tfm.forecast(horizon=prediction_steps, inputs=inputs)
        quantile = np.asarray(quantile, dtype=np.float32)  # [VERIFY] (n_series, pred, n_q)
        q_values = quantile[:, :prediction_steps, :]
        n_q = q_values.shape[-1]
        # [VERIFY] neu n_q != 9 (vd 10 cot gom mean), chinh lai mapping levels <-> cot
        levels = self.TFM_QUANTILE_LEVELS if n_q == len(self.TFM_QUANTILE_LEVELS) \
            else np.linspace(0.1, 0.9, n_q)
        return q_values, levels, mean, std


# ============================================================
# Moirai-2.0 (quantile) - ke thua MoiraiProposer, them predict_quantiles
# ============================================================
class Moirai2QuantileProposer(MoiraiProposer):
    def predict_quantiles(self, history, prediction_steps, quantile_levels, num_samples: int = 100):
        import pandas as pd
        from gluonts.dataset.common import ListDataset

        context_2d, mean, std, lat, lon, n_series = _zscore_stats(history)
        model = self._forecast_cls(
            module=self.module, prediction_length=prediction_steps, context_length=history.shape[0],
            patch_size=self.patch_size, num_samples=num_samples,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=self.predictor_batch_size, device=self.device)
        start = pd.Period("2000-01-01", freq="6H")
        dataset = ListDataset([{"start": start, "target": context_2d[i]} for i in range(n_series)], freq="6H")
        forecasts = list(predictor.predict(dataset))

        levels = list(quantile_levels)
        # [VERIFY] gluonts Forecast.quantile(level) -> (pred,). Voi SampleForecast tinh tu mau.
        q_values = np.stack(
            [np.stack([np.asarray(f.quantile(l)) for l in levels], axis=-1) for f in forecasts],
            axis=0,
        ).astype(np.float32)  # (n_series, pred, n_q)
        return q_values, np.asarray(levels), mean, std


# ============================================================
# Helpers: reshape + candidate generator chung (Algorithm 3)
# ============================================================
def _reshape_series_to_grid(denorm_series, mean, std, history, prediction_steps, n_cand):
    """
    denorm_series: (n_series, n_cand, pred) -> tra ve (denorm, norm, mean, std)
    voi denorm/norm shape (n_cand, pred, lat, lon) - khop format ChronosProposer.predict.
    """
    _T, lat, lon = history.shape
    denorm = denorm_series.transpose(1, 2, 0).reshape(n_cand, prediction_steps, lat, lon)
    mean_grid = mean.reshape(lat, lon)
    std_grid = std.reshape(lat, lon)
    norm = (denorm - mean_grid[None, None]) / std_grid[None, None]
    return denorm.astype(np.float32), norm.astype(np.float32), mean, std


def _generate_candidates_one_var(proposer, backbone, history_1var, prediction_steps,
                                 n_candidates, quantile_levels, rng):
    """Tra ve (denorm, norm, mean, std) cho 1 bien, pool = N candidate theo Algorithm 3."""
    if backbone in NATIVE_BACKBONES:
        # Nhanh sampler that: N-1 sample + median cua chinh N-1 sample do (dong 1-2, 11)
        n_draw = max(1, n_candidates - 1)
        denorm, _norm, mean, std = proposer.predict(
            history_1var, prediction_steps, temperature=1.0, n_samples=n_draw,
        )  # denorm: (n_draw, pred, lat, lon)
        median = np.median(denorm, axis=0, keepdims=True)
        denorm_all = np.concatenate([denorm, median], axis=0)  # (n_candidates, pred, lat, lon)
        _T, lat, lon = history_1var.shape
        mean_grid = mean.reshape(lat, lon)
        std_grid = std.reshape(lat, lon)
        norm_all = (denorm_all - mean_grid[None, None]) / std_grid[None, None]
        return denorm_all.astype(np.float32), norm_all.astype(np.float32), mean, std

    # Nhanh quantile: Algorithm 3 inverse-CDF
    q_values, q_levels, mean, std = proposer.predict_quantiles(
        history_1var, prediction_steps, quantile_levels,
    )
    cand_series = algo3_candidates_from_quantiles(q_values, q_levels, n_candidates, rng)  # (N, pred, n_series)
    _T, lat, lon = history_1var.shape
    denorm = cand_series.reshape(n_candidates, prediction_steps, lat, lon)
    mean_grid = mean.reshape(lat, lon)
    std_grid = std.reshape(lat, lon)
    norm = (denorm - mean_grid[None, None]) / std_grid[None, None]
    return denorm.astype(np.float32), norm.astype(np.float32), mean, std


def generate_multivar_candidates_algo3(proposer, backbone, data, prediction_steps,
                                       n_candidates, quantile_levels, rng):
    """Phien ban da-backbone cua generate_multivar_candidates, pool theo Algorithm 3."""
    return {
        "t2m": _generate_candidates_one_var(proposer, backbone, data["history_t2m"],
                                            prediction_steps, n_candidates, quantile_levels, rng),
        "u10": _generate_candidates_one_var(proposer, backbone, data["history_u10"],
                                            prediction_steps, n_candidates, quantile_levels, rng),
        "v10": _generate_candidates_one_var(proposer, backbone, data["history_v10"],
                                            prediction_steps, n_candidates, quantile_levels, rng),
    }


def get_proposer_any(backbone, model_id=None, device="cuda", history_steps=512, prediction_steps=64):
    """Factory thong nhat: native (proposers.py goc) + quantile (file nay)."""
    if backbone == "chronos2":
        return Chronos2Proposer(model_id=model_id or "amazon/chronos-2", device=device)
    if backbone in ("timesfm", "timesfm3"):
        mid = resolve_model_id(backbone, model_id)
        return TimesFMQuantileProposer(model_id=mid, device=device,
                                       history_steps=history_steps, prediction_steps=prediction_steps)
    if backbone == "moirai2":
        mid = resolve_model_id(backbone, model_id)
        return Moirai2QuantileProposer(model_id=mid, device=device)
    # chronos (1), moirai1 -> dung factory goc
    return get_proposer(backbone, model_id=model_id, device=device,
                        history_steps=history_steps, prediction_steps=prediction_steps)
