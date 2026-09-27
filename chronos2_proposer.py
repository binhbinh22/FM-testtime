import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


# ============================================================
# CAN CU DA TRA CUU TRUC TIEP tu source code + docstring that cua
# chronos-forecasting>=2.0 (KHONG doan, xem tham chieu cuoi file):
#
# - Chronos-2 KHONG autoregressive-sample nhu Chronos T5 (ChronosProposer
#   trong run.py) - day la model encoder-only, ra QUANTILE truc tiep
#   trong 1 lan forward, khong co tham so num_samples.
# - Multivariate/group attention THAT giua nhieu series chi xay ra khi
#   bat `cross_learning=True` trong predict_df(). MAC DINH LA False - neu
#   khong bat, Chronos-2 van du bao TUNG series DOC LAP, y het
#   ChronosProposer cu (mat het diem khac biet ma yeu cau la "chay
#   multivariate cho Chronos-2 chu khong phai Chronos").
# - Docstring goc canh bao: cross_learning phu thuoc batch_size, "batch
#   qua lon se lech khoi kich thuoc group toi da dung khi pretrain -
#   khuyen nghi dung batch ~100 (dung trong Chronos-2 technical report)".
#   -> KHONG duoc goi cross_learning=True cho ca V=2048 node trong 1 lan;
#   phai chia batch ~100 node/lan.
# - Output DataFrame co cot: id, timestamp, target_name, predictions,
#   roi cac cot quantile dat ten theo string cua muc quantile (vd "0.5").
# ============================================================


def _is_quantile_col(col: str) -> bool:
    try:
        float(col)
        return True
    except (TypeError, ValueError):
        return False


class Chronos2Proposer:
    """
    Proposer dung THAT Chronos-2 (chronos-forecasting>=2.0,
    class Chronos2Pipeline qua BaseChronosPipeline.from_pretrained),
    THAY THE ChronosProposer trong run.py (Chronos T5 co - goi TUNG
    series doc lap qua vong lap batch, KHONG co tuong tac giua cac
    series du history duoc flatten thanh lat*lon series rieng biet).

    DIEM MAU CHOT de co "multivariate that" (group attention giua cac
    node/cum) dung nhu yeu cau va dung mo ta ARCHITECTURE_1_MCTS_
    SPATIOTEMPORAL.md muc 2.2 ("nhieu bien...duoc xu ly dong thoi qua
    group attention"): bat `cross_learning=True`. Neu tat co nay,
    Chronos-2 se du bao tung series doc lap - khong khac gi Chronos T5
    ve mat tuong tac khong gian.

    batch_size mac dinh = 100 (KHONG dung default 256 cua thu vien) -
    theo dung khuyen nghi chinh thuc cho cross_learning (~kich thuoc
    group luc pretrain). Voi nhanh Fine (V=2048 node), engine se goi lop
    nay theo tung batch node - MAC DINH chia theo thu tu phang lien tuc
    (contiguous chunk), CO THE truyen group_order de nhom theo cum
    Spectral Clustering thay vi thu tu ngau nhien (xem TODO trong
    docstring predict()).

    CHUA KIEM CHUNG qua API that - moi truong lam viec hien tai khong co
    GPU/torch/chronos-forecasting cai san. Da tra cuu source code +
    docstring THAT qua WebFetch (khong doan mo hinh), nhung BAT BUOC
    chay thu 1 lan tren may co GPU + du lieu that truoc khi tin dung ket
    qua - dung tinh than "Kiem chung qua API that" con de trong
    ARCHITECTURE_1_MCTS_SPATIOTEMPORAL.md muc 4. Rui ro cao nhat: dinh
    dang chinh xac cua ten cot quantile tra ve (vd "0.5" vs "0.50") va
    thu tu hang sau pivot - da code phong thu (bao loi tuong minh thay
    vi im lang sai), xem self-test o cuoi file (dung mock, khong can
    model that) de kiem tra logic parse.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device: str = "cuda",
        batch_size: int = 100,
        freq: str = "6h",
    ):
        import torch
        from chronos import BaseChronosPipeline

        # Tu dong roi ve CPU neu khong co GPU - giong het logic
        # ChronosProposer.__init__ trong run.py, tranh crash ngay tu dau
        # tren may khong co CUDA (vd may local de chay self-test/CI).
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA khong san co, chuyen sang CPU (se cham hon nhieu).")
            device = "cpu"

        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device)
        self.batch_size = batch_size
        self.freq = freq

    def _quantile_levels_for(self, n_samples: int, temperature: float) -> List[float]:
        """
        CHUA CHOT - mo dang tranh luan trong bao cao kien truc (muc 2.4,
        "cach sinh N candidate tu Chronos-2"). Dung PHUONG AN 1 da de
        xuat ("duong muc quantile dung chung cho moi node"): sinh N
        candidate TRONG 1 LAN GOI predict_df, moi candidate ung voi 1
        muc quantile ap dung DONG THOI cho moi series trong batch (dam
        bao candidate la 1 truong THONG NHAT ve khong gian, khong tron
        lan muc cao o node nay voi muc thap o node khac).

        Re hon PHUONG AN 2 (nhieu context noise, phai goi lai N lan) -
        nhung chua duoc doi chieu thuc nghiem xem phuong an nao tot hon.

        temperature dieu chinh BE RONG khoang quantile duoc lay quanh
        median (0.5): cao -> trai rong ve 2 duoi (da dang hon), thap ->
        co cum sat median.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples phai >= 1, nhan {n_samples}")
        if n_samples == 1:
            return [0.5]

        eff_temp = max(float(temperature), 1e-4)
        half_width = min(0.49, 0.49 * eff_temp)
        lo, hi = 0.5 - half_width, 0.5 + half_width
        return [float(q) for q in np.linspace(lo, hi, n_samples)]

    def _predict_batch(
        self,
        history_flat_batch: np.ndarray,  # (n_batch, T)
        batch_ids: np.ndarray,
        prediction_steps: int,
        quantile_levels: List[float],
    ) -> np.ndarray:
        """Goi predict_df cho 1 batch (<= self.batch_size series), tra ve (n_q, pred, n_batch)."""
        T = history_flat_batch.shape[1]
        timestamps = pd.date_range("2000-01-01", periods=T, freq=self.freq)

        df = pd.DataFrame(
            {
                "item_id": np.repeat(batch_ids, T),
                "timestamp": np.tile(timestamps, len(batch_ids)),
                "target": history_flat_batch.reshape(-1),
            }
        )

        pred_df = self.pipeline.predict_df(
            df,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=prediction_steps,
            quantile_levels=quantile_levels,
            batch_size=self.batch_size,
            cross_learning=True,  # BAT BUOC de co group attention that giua cac series trong batch
        )

        id_col = "id" if "id" in pred_df.columns else "item_id"
        quantile_cols_available = {c: c for c in pred_df.columns if _is_quantile_col(c)}

        out = np.empty((len(quantile_levels), prediction_steps, len(batch_ids)), dtype=np.float64)
        for k, q in enumerate(quantile_levels):
            col = None
            for c in quantile_cols_available:
                if abs(float(c) - q) < 1e-6:
                    col = c
                    break
            if col is None:
                raise KeyError(
                    f"Khong tim thay cot quantile khop voi muc {q} trong pred_df. "
                    f"Cac cot quantile hien co: {list(quantile_cols_available)}. "
                    "API that co the dat ten cot khac voi gia dinh - can doi chieu lai."
                )
            pivot = pred_df.pivot(index=id_col, columns="timestamp", values=col)
            pivot = pivot.reindex(index=batch_ids)
            if pivot.isnull().values.any():
                raise ValueError(
                    f"pivot co gia tri NaN sau reindex theo batch_ids - id_column tra ve tu "
                    f"predict_df co the khong khop batch_ids da gui di ({id_col})."
                )
            out[k] = pivot.values.T

        return out

    def predict(
        self,
        history: np.ndarray,
        prediction_steps: int,
        temperature: float = 1.0,
        n_samples: int = 10,
        group_order: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Interface GIONG HET ChronosProposer.predict (run.py) de cam thang
        vao SpatialClusterSearchEngine ma khong doi engine:
            history: (T, a, b)
        Return: forecast_denorm, forecast_norm (n_samples, prediction_steps, a, b),
                mean, std (a*b, 1)

        group_order: (a*b,) thu tu index de gom nhom truoc khi chia batch
        cross_learning (TODO - nen truyen nhan cum tu spectral_clustering.py
        de moi batch la 1 vung khi hau lien quan, thay vi chia theo thu
        tu phang mac dinh). None = giu nguyen thu tu phang (lat ngoai,
        lon trong).
        """
        T, a, b = history.shape
        n_series = a * b
        history_flat = history.reshape(T, n_series).T.astype(np.float64)  # (n_series, T)

        mean = np.mean(history_flat, axis=1, keepdims=True)
        std = np.std(history_flat, axis=1, keepdims=True)
        std = np.where(std < 1e-5, 1.0, std)

        quantile_levels = self._quantile_levels_for(n_samples, temperature)

        order = np.arange(n_series) if group_order is None else np.asarray(group_order)
        if order.shape != (n_series,):
            raise ValueError(f"group_order phai shape ({n_series},), nhan {order.shape}")

        all_denorm_ordered = np.empty((n_samples, prediction_steps, n_series), dtype=np.float64)

        for start in range(0, n_series, self.batch_size):
            end = min(start + self.batch_size, n_series)
            idx = order[start:end]
            batch_out = self._predict_batch(
                history_flat[idx], np.arange(start, end), prediction_steps, quantile_levels
            )  # (n_q, pred, n_batch)
            all_denorm_ordered[:, :, idx] = batch_out

        mean_expanded = mean[None, None, :, 0]
        std_expanded = std[None, None, :, 0]
        all_norm_ordered = (all_denorm_ordered - mean_expanded) / std_expanded

        forecast_denorm = all_denorm_ordered.reshape(n_samples, prediction_steps, a, b)
        forecast_norm = all_norm_ordered.reshape(n_samples, prediction_steps, a, b)
        return forecast_denorm, forecast_norm, mean, std


if __name__ == "__main__":
    # Self-test BANG MOCK (khong can torch/chronos-forecasting/GPU) - chi
    # kiem tra logic parse pred_df -> mang numpy dung thu tu, KHONG kiem
    # tra chat luong du bao thuc (khong the lam duoc trong moi truong
    # nay - xem canh bao trong docstring class).
    class _FakePipeline:
        """Gia lap predict_df dung DUNG schema that da tra cuu: cot id,
        timestamp, target_name, predictions, <quantile strings>."""

        def predict_df(self, df, id_column, timestamp_column, target,
                        prediction_length, quantile_levels, batch_size, cross_learning):
            assert cross_learning is True, "phai luon bat cross_learning=True"
            ids = df[id_column].unique()
            last_val = df.groupby(id_column)[target].last()
            timestamps = pd.date_range("2099-01-01", periods=prediction_length, freq="6h")

            rows = []
            for series_id in ids:
                base = float(last_val[series_id])
                for t in timestamps:
                    row = {"id": series_id, "timestamp": t, "target_name": target, "predictions": base}
                    for q in quantile_levels:
                        # gia lap: quantile cao hon -> gia tri cao hon, tuyen tinh theo (q-0.5)
                        row[str(q)] = base + (q - 0.5) * 10.0
                    rows.append(row)
            return pd.DataFrame(rows)

    np.random.seed(0)
    proposer = Chronos2Proposer.__new__(Chronos2Proposer)  # bo qua __init__ (khong load model that)
    proposer.pipeline = _FakePipeline()
    proposer.batch_size = 5  # co tinh nho de test wiring chia batch
    proposer.freq = "6h"

    T, a, b = 10, 3, 4  # n_series = 12 > batch_size=5 -> phai chia >=3 batch
    history = 280.0 + np.arange(T)[:, None, None] * 0.0 + np.random.randn(T, a, b)

    n_samples = 4
    forecast_denorm, forecast_norm, mean, std = proposer.predict(
        history, prediction_steps=6, temperature=1.0, n_samples=n_samples
    )

    print("forecast_denorm.shape:", forecast_denorm.shape, "(ky vong", (n_samples, 6, a, b), ")")
    assert forecast_denorm.shape == (n_samples, 6, a, b)
    assert forecast_norm.shape == (n_samples, 6, a, b)
    assert mean.shape == (a * b, 1)
    assert std.shape == (a * b, 1)

    # Quantile cao hon (index cuoi trong linspace) phai cho gia tri cao hon
    # (dung quy uoc gia lap trong _FakePipeline) - kiem tra thu tu khong bi dao lon khi ghep batch.
    assert np.all(forecast_denorm[-1] >= forecast_denorm[0]), (
        "Candidate ung voi quantile cao nhat phai >= candidate quantile thap nhat - "
        "sai thu tu nghia la logic ghep batch/pivot bi loi."
    )

    print("mean/std shape OK, thu tu quantile-per-candidate OK sau khi ghep nhieu batch.")

    print("\nKiem tra group_order (gom theo nhom tuy y thay vi thu tu phang):")
    custom_order = np.random.permutation(a * b)
    fd2, fn2, mean2, std2 = proposer.predict(
        history, prediction_steps=6, temperature=1.0, n_samples=n_samples, group_order=custom_order
    )
    assert fd2.shape == forecast_denorm.shape
    print("group_order OK (khong crash, dung shape).")

    print("\nOK (mock self-test - CAN chay lai voi model that truoc khi dung cho ket qua bao cao).")
