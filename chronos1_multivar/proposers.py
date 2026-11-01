import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run import ChronosProposer  # noqa: E402  (token-based autoregressive sampling)


# ============================================================
# Backbone registry cho nghien cuu DA-BACKBONE (xem README.md muc "Backbones").
#
# PHAN LOAI THEO CO CHE SINH "1 lich su -> N tuong lai" (da verify 2026-10
# qua paper/model card goc, KHONG doan):
#
#   sampler="autoregressive"  Chronos-T5 / Chronos-2
#       Token roi rac + decode tung buoc, moi sample la 1 lan random draw
#       (temperature/top-k/top-p) doc lap qua ca horizon -> N trajectory
#       i.i.d. THAT SU KHAC NHAU. Best-of-N chay dung ban chat.
#
#   sampler="mixture"         Moirai-1.x (1.0 / 1.1-R)
#       Output la tham so MIXTURE DISTRIBUTION (Student-t + Normal +
#       NegBinom + LogNormal). num_samples=N -> draw N mau THAT tu phan
#       phoi do trong 1 forward pass. Best-of-N chay dung ban chat.
#
#   sampler="quantile"        Moirai-2.0 / TimesFM-2.x / TimesFM-3
#       Chi xuat 9 quantile co dinh (0.1..0.9) + point, TAT DINH. "N tuong
#       lai" chi co the lay bang cach chon muc quantile hoac noi suy CDF ->
#       KHONG phai mau i.i.d., khong co do da dang kieu Best-of-N. Vi vay
#       cac backbone nay CHI dung lam SINGLE-PASS BASELINE (greedy), KHONG
#       dua vao vong Best-of-N (gain ky vong ~0 - chinh la bang chung cho
#       luan diem "TTC-by-sampling can bo sampler that").
# ============================================================


class MoiraiProposer:
    """
    Proposer dung Moirai (uni2ts) - CUNG interface .predict(...) voi
    ChronosProposer (run.py) de cam thang vao generate_multivar_candidates.

    Moirai-1.x: sampler="mixture" (lay mau that) -> dung cho Best-of-N.
    Moirai-2.0: sampler="quantile" -> chi nen dung single-pass (n_samples=1
                lay median) lam baseline.

    LUU Y interface: Moirai KHONG co decoding temperature (khac Chronos token
    autoregressive). Tham so `temperature` giu lai chi de tuong thich chu ky
    ham, Moirai bo qua. Do da dang giua cac sample den tu num_samples.
    """

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-1.1-R-small",
        device: str = "cuda",
        patch_size="auto",
        predictor_batch_size: int = 512,
    ):
        import torch

        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model_id = model_id
        self.patch_size = patch_size
        self.predictor_batch_size = predictor_batch_size

        print(f"Initializing MoiraiProposer ({model_id}) on {self.device}...")
        self._forecast_cls, self.module = self._load_moirai(model_id)
        print("Moirai module loaded successfully.")

    @staticmethod
    def _load_moirai(model_id: str):
        """
        Chon dung cap (ForecastClass, module) theo phien ban Moirai.

        VERIFY TREN ENV CUA BAN: ten module/class cua Moirai-2.0 trong uni2ts
        co the khac nhau giua cac ban uni2ts. Doan code duoi THU Moirai2* truoc
        (ban moi), neu khong co thi fallback ve Moirai* (1.x). In ra duong da
        dung de ban doi chieu.
        """
        is_v2 = ("moirai-2" in model_id.lower()) or ("moirai2" in model_id.lower())
        if is_v2:
            try:
                from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
                print("  [Moirai] dung Moirai2Forecast / Moirai2Module (uni2ts moirai2).")
                return Moirai2Forecast, Moirai2Module.from_pretrained(model_id)
            except Exception as e:  # noqa: BLE001
                print(f"  [Moirai] khong import duoc Moirai2* ({e}); fallback MoiraiForecast/MoiraiModule.")
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        print("  [Moirai] dung MoiraiForecast / MoiraiModule.")
        return MoiraiForecast, MoiraiModule.from_pretrained(model_id)

    def predict(self, history, prediction_steps, temperature=1.0, n_samples=1):
        import pandas as pd
        from gluonts.dataset.common import ListDataset

        if temperature != 1.0:
            print(f"[MoiraiProposer] temperature={temperature} bi bo qua (Moirai khong co decoding temperature).")

        history_steps, lat, lon = history.shape
        n_series = lat * lon
        context_2d = history.reshape(history_steps, n_series).T.astype(np.float32)  # (n_series, history_steps)

        mean = np.mean(context_2d, axis=1, keepdims=True)
        std = np.std(context_2d, axis=1, keepdims=True)
        std = np.where(std < 1e-5, 1.0, std)

        n_samples_eff = max(int(n_samples), 1)

        model = self._forecast_cls(
            module=self.module,
            prediction_length=prediction_steps,
            context_length=history_steps,
            patch_size=self.patch_size,
            num_samples=n_samples_eff,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=self.predictor_batch_size, device=self.device)

        start = pd.Period("2000-01-01", freq="6H")  # moc gia, khong anh huong forecast
        dataset = ListDataset(
            [{"start": start, "target": context_2d[i]} for i in range(n_series)], freq="6H",
        )
        forecasts = list(predictor.predict(dataset))

        # f.samples: (n_samples_eff, prediction_steps). Voi Moirai-2.0 (quantile)
        # uni2ts van tra ve SampleForecast; single-pass baseline nen goi n_samples=1.
        all_samples = np.stack([f.samples for f in forecasts], axis=0)  # (n_series, n_samples_eff, prediction_steps)
        forecast_denorm = all_samples.astype(np.float32)  # Moirai output da o don vi goc

        mean_expanded = mean[:, np.newaxis, :]
        std_expanded = std[:, np.newaxis, :]
        forecast_norm = (forecast_denorm - mean_expanded) / std_expanded

        def reshape_to_spatial(arr):
            arr = arr.transpose(1, 2, 0)  # -> (n_samples, prediction_steps, n_series)
            return arr.reshape(n_samples_eff, prediction_steps, lat, lon)

        return reshape_to_spatial(forecast_denorm), reshape_to_spatial(forecast_norm), mean, std


class TimesFMProposer:
    """
    Proposer dung TimesFM (Google) - SINGLE-PASS BASELINE (sampler="quantile",
    TAT DINH). Tra ve point forecast lam "1 candidate" duy nhat, dung cung
    interface .predict(...) de chay qua greedy.py.

    CO CHE: TimesFM KHONG lay mau ngau nhien. Moi forward pass ra 1 point
    forecast + 9/10 quantile co dinh. Vi vay n_samples>1 KHONG tao duoc
    candidate da dang -> chi nen goi qua greedy (n_samples=1). Neu n_samples>1,
    broadcast cung 1 forecast ra N ban (Best-of-N chon trung nhau -> gain 0,
    dung nhu ky vong cho backbone quantile).

    HO TRO 3 API (tu nhan dien tu model_id):
      * "2.0"  API CU : timesfm.TimesFm(hparams, checkpoint) + .forecast(inputs, freq)
                        -> (point[B,H], quantile[B,H,Q])
      * "2.5"  API MOI: timesfm.TimesFM_2p5_200M_torch.from_pretrained(...) +
                        .compile(ForecastConfig(...)) + .forecast(horizon=H, inputs=[...])
                        -> (point[B,H], quantile[B,H,10])   (DA VERIFY tu model card HF)
      * "3.0"  API MOI (cung ho new-style). Ten class 3.0 KHONG hardcode -> TU
                        DO class *_torch co .from_pretrained trong package
                        `timesfm` DA CAI tren env cua ban (introspection, dung
                        ground-truth that thay vi doan). Neu khong tim thay,
                        bao loi ro rang.
    """

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        device: str = "cuda",
        history_steps: int = 512,
        prediction_steps: int = 64,
        per_core_batch_size: int = 32,
    ):
        import torch

        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model_id = model_id
        self.history_steps = history_steps
        self.prediction_steps = prediction_steps
        self.per_core_batch_size = per_core_batch_size
        self.version = self._detect_version(model_id)
        self.api = "old" if self.version == "2.0" else "new"

        print(f"Initializing TimesFMProposer ({model_id}, version={self.version}, api={self.api}) on {self.device}...")
        self.tfm = self._build_model()
        print("TimesFM loaded successfully.")

    @staticmethod
    def _detect_version(model_id: str) -> str:
        m = model_id.lower()
        if "2.0" in m or "2p0" in m:
            return "2.0"
        if "2.5" in m or "2p5" in m:
            return "2.5"
        if "3.0" in m or "3p0" in m or "timesfm-3" in m or "timesfm3" in m:
            return "3.0"
        return "2.5"  # mac dinh new-style (hien tai)

    @staticmethod
    def _find_new_style_class(timesfm_mod, version: str):
        """
        Tu DO class loader new-style (*_torch co .from_pretrained) tu package
        `timesfm` da cai - uu tien class co tag khop version (2.5->'2p5',
        3.0->'3p0'). Dung introspection => lay dung ten class THAT cua ban
        timesfm dang cai, khong phu thuoc guess cua nguoi viet.
        """
        tag = version.replace(".", "p")  # "2.5"->"2p5", "3.0"->"3p0"
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

    def _build_model(self):
        import timesfm

        if self.api == "old":
            # ---- TimesFM 2.0 (API cu, on dinh) ----
            backend = "gpu" if self.device == "cuda" else "cpu"
            hparams = timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=self.per_core_batch_size,
                horizon_len=max(self.prediction_steps, 128),      # >= prediction_steps
                context_len=min(max(self.history_steps, 512), 2048),
                num_layers=50,
                use_positional_embedding=False,
            )
            ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.model_id)
            return timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)

        # ---- TimesFM 2.5 / 3.0 (API moi: from_pretrained + compile + forecast) ----
        cls, cls_name = self._find_new_style_class(timesfm, self.version)
        if cls is None:
            available = [n for n in dir(timesfm) if "torch" in n.lower()]
            raise RuntimeError(
                f"Khong tim thay class loader new-style cho TimesFM {self.version} trong package `timesfm`. "
                f"Cac ten *_torch co san: {available}. Cap nhat `pip install -U timesfm` hoac chi dinh "
                f"model_id/class dung voi ban da cai."
            )
        print(f"  [TimesFM] new-style class: timesfm.{cls_name}")
        model = cls.from_pretrained(self.model_id)
        # 2.5: .compile(ForecastConfig) la buoc bat buoc. 3.0 (TimesFM3Torch) co
        # the co API khac -> .compile(ForecastConfig) se loi TypeError (trung
        # ten voi torch.nn.Module.compile). Lam best-effort: thu compile, neu
        # khong ap dung thi bo qua (mot so ban khong can buoc nay).
        try:
            model.compile(
                timesfm.ForecastConfig(
                    max_context=min(max(self.history_steps, 512), 2048),
                    max_horizon=max(self.prediction_steps, 256),
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
        except TypeError as e:
            print(f"  [TimesFM] .compile(ForecastConfig) khong ap dung cho {cls_name} ({e}); bo qua buoc compile.")
        return model

    def _forecast_point(self, inputs, prediction_steps):
        """Tra ve point_forecast dang (n_series, prediction_steps) cho ca 2 API."""
        if self.api == "old":
            freq = [0] * len(inputs)  # 0 = high-frequency (6-hourly)
            point, _q = self.tfm.forecast(inputs, freq=freq)
        else:
            point, _q = self.tfm.forecast(horizon=prediction_steps, inputs=inputs)
        return np.asarray(point, dtype=np.float32)[:, :prediction_steps]

    def predict(self, history, prediction_steps, temperature=1.0, n_samples=1):
        history_steps, lat, lon = history.shape
        n_series = lat * lon
        context_2d = history.reshape(history_steps, n_series).T.astype(np.float32)  # (n_series, history_steps)

        mean = np.mean(context_2d, axis=1, keepdims=True)
        std = np.std(context_2d, axis=1, keepdims=True)
        std = np.where(std < 1e-5, 1.0, std)

        inputs = [context_2d[i] for i in range(n_series)]
        point_forecast = self._forecast_point(inputs, prediction_steps)  # (n_series, H)

        n_samples_eff = max(int(n_samples), 1)
        if n_samples_eff > 1:
            print(f"[TimesFMProposer] n_samples={n_samples_eff} nhung TimesFM tat dinh "
                  f"-> broadcast cung 1 forecast (Best-of-N se trung nhau). Chi nen dung greedy.")
        forecast_denorm = np.repeat(point_forecast[:, np.newaxis, :], n_samples_eff, axis=1)  # (n_series, N, H)

        mean_expanded = mean[:, np.newaxis, :]
        std_expanded = std[:, np.newaxis, :]
        forecast_norm = (forecast_denorm - mean_expanded) / std_expanded

        def reshape_to_spatial(arr):
            arr = arr.transpose(1, 2, 0)  # -> (n_samples, prediction_steps, n_series)
            return arr.reshape(n_samples_eff, prediction_steps, lat, lon)

        return reshape_to_spatial(forecast_denorm), reshape_to_spatial(forecast_norm), mean, std


# ============================================================
# Registry + factory
# ============================================================

BACKBONE_REGISTRY = {
    "chronos": {
        "default_model_id": "amazon/chronos-t5-base",
        "sampler": "autoregressive",
        "supports_bon": True,
    },
    "moirai1": {
        "default_model_id": "Salesforce/moirai-1.1-R-small",
        "sampler": "mixture",
        "supports_bon": True,
    },
    "moirai2": {
        "default_model_id": "Salesforce/moirai-2.0-R-small",
        "sampler": "quantile",
        "supports_bon": False,
    },
    "timesfm": {
        "default_model_id": "google/timesfm-2.5-200m-pytorch",   # API moi (da verify); 2.0 van chay qua --model-id
        "sampler": "quantile",
        "supports_bon": False,
    },
    "timesfm3": {
        "default_model_id": "google/timesfm-3.0-pytorch",
        "sampler": "quantile",
        "supports_bon": False,
    },
}


def resolve_model_id(backbone: str, model_id: str = None) -> str:
    if model_id:
        return model_id
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(f"backbone '{backbone}' khong co trong registry: {list(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[backbone]["default_model_id"]


def get_proposer(backbone: str, model_id: str = None, device: str = "cuda",
                 history_steps: int = 512, prediction_steps: int = 64, **kwargs):
    """
    Tra ve 1 proposer dung interface .predict(history, prediction_steps,
    temperature, n_samples) -> (denorm, norm, mean, std) theo dung backbone.
    """
    model_id = resolve_model_id(backbone, model_id)

    if backbone == "chronos":
        return ChronosProposer(model_id=model_id, device=device)
    if backbone in ("moirai1", "moirai2"):
        return MoiraiProposer(model_id=model_id, device=device, **kwargs)
    if backbone in ("timesfm", "timesfm3"):
        # version tu nhan dien tu model_id trong TimesFMProposor._detect_version
        return TimesFMProposer(model_id=model_id, device=device,
                               history_steps=history_steps, prediction_steps=prediction_steps, **kwargs)
    raise ValueError(f"backbone '{backbone}' chua duoc ho tro. Chon: {list(BACKBONE_REGISTRY)}")
