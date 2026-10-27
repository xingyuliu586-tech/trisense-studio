"""Safe, explicit feature contract. Uploaded pickle files are never deserialized."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np

SCHEMA = "mosei-aligned-50-v1"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
SHAPES = {"text": (50, 768), "audio": (50, 74), "vision": (50, 35), "text_bert": (3, 50)}


class FeatureError(ValueError):
    pass


@dataclass
class Features:
    arrays: dict
    synthetic: bool = False

    @property
    def length(self):
        return int(self.arrays["text_bert"][1].sum())

    def clone(self):
        return Features({k: v.copy() for k, v in self.arrays.items()}, self.synthetic)

    def availability(self):
        tokens = self.arrays["text_bert"]
        return np.stack([
            (tokens[1] != 0) & (tokens[0] != 100),
            np.any(self.arrays["audio"] != 0, axis=-1),
            np.any(self.arrays["vision"] != 0, axis=-1),
        ], axis=-1)


def read_features(source: bytes | Path) -> Features:
    if isinstance(source, Path):
        if source.stat().st_size > MAX_UPLOAD_BYTES:
            raise FeatureError("文件超过 8 MB；请上传单个样本的 NPZ 特征。")
        raw = source.read_bytes()
    else:
        raw = source
    if len(raw) > MAX_UPLOAD_BYTES:
        raise FeatureError("文件超过 8 MB。")
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) > 12 or sum(x.file_size for x in entries) > MAX_UPLOAD_BYTES:
                raise FeatureError("NPZ 解压后体积或字段数量超过限制。")
            if len({e.filename for e in entries}) != len(entries):
                raise FeatureError("NPZ 包含重复字段。")
        with np.load(BytesIO(raw), allow_pickle=False) as z:
            if "schema" not in z or np.asarray(z["schema"]).shape != () or str(z["schema"]) != SCHEMA:
                raise FeatureError("特征来源不兼容：需要 schema=mosei-aligned-50-v1。第一问自提取特征不能直接套用本模型。")
            arrays = {}
            for key, shape in SHAPES.items():
                if key not in z:
                    raise FeatureError(f"缺少字段 {key}，需要形状 {shape}。")
                arr = z[key]
                if arr.shape != shape or arr.dtype.kind not in "fiu":
                    raise FeatureError(f"{key} 必须是数值数组，形状为 {shape}；实际为 {arr.shape}。")
                if not np.isfinite(arr).all() or np.max(np.abs(arr.astype(float))) > 1e8:
                    raise FeatureError(f"{key} 包含非有限值或异常数值。")
                arrays[key] = arr.astype(np.float32 if key != "text_bert" else np.int64)
                if key == "text_bert" and not np.array_equal(arr, arrays[key]):
                    raise FeatureError("text_bert 必须包含整数。")
            tokens = arrays["text_bert"]
            if not np.isin(tokens[1], [0, 1]).all() or not np.isin(tokens[2], [0, 1]).all():
                raise FeatureError("attention_mask 与 token_type_ids 只能为 0 或 1。")
            if (tokens[0] < 0).any() or (tokens[0] >= 30522).any():
                raise FeatureError("token ID 超出 bert-base-uncased 词表范围。")
            length = int(tokens[1].sum())
            if length < 3 or not np.array_equal(tokens[1], np.arange(50) < length):
                raise FeatureError("有效词元需为连续前缀，长度为 3–50（含 CLS / SEP）。")
            for key in ("audio", "vision"):
                if np.any(arrays[key][length:] != 0):
                    raise FeatureError(f"{key} 在填充位置必须为零。")
            synthetic = False
            if "synthetic" in z:
                flag = z["synthetic"]
                if flag.shape != () or flag.dtype.kind not in "biu" or int(flag) not in (0, 1):
                    raise FeatureError("synthetic 必须是单个布尔值。")
                synthetic = bool(flag)
    except FeatureError:
        raise
    except (ValueError, KeyError, OSError, zipfile.BadZipFile, EOFError) as exc:
        raise FeatureError("无法读取安全 NPZ 数值文件；不接受 PKL、对象数组或损坏的压缩包。") from exc
    return Features(arrays, synthetic)


def mask_segment(sample: Features, mode: str, rate: float, location: str):
    altered = sample.clone()
    n = sample.length - 2
    count = int(round(n * rate)) if rate else 0
    if rate > 0:
        count = max(1, count)
    start = {"early": 1, "middle": 1 + (n - count) // 2, "late": 1 + n - count}[location]
    positions = list(range(start, start + count)) if mode != "none" else []
    for key in ("audio", "vision"):
        if mode in (key, "audio+vision"):
            altered.arrays[key][positions] = 0
    return altered, positions


def feature_view(sample: Features):
    length = sample.length
    masks = sample.availability()
    matrices = {}
    for column, name in enumerate(("text", "audio", "vision")):
        chosen = np.linspace(0, sample.arrays[name].shape[1] - 1, 24, dtype=int)
        values = sample.arrays[name][:length, chosen].astype(float)
        valid = masks[:length, column]
        shown = np.zeros_like(values)
        if valid.any():
            subset = values[valid]
            shown[valid] = np.clip((subset - subset.mean(0)) / np.maximum(subset.std(0), 1e-6), -3, 3)
        matrices[name] = {"channels": (chosen + 1).tolist(), "values": np.round(shown.T, 3).tolist()}
    return {"length": length, "content_positions": max(0, length - 2),
            "available": masks[:length].T.astype(int).tolist(),
            "coverage": np.round(masks[1:length-1].mean(axis=0), 4).tolist(),
            "heatmaps": matrices}
