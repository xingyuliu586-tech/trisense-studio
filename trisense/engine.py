"""CPU inference, exact three-player Shapley and audio/vision window occlusion."""
import hashlib
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .features import Features, feature_view, mask_segment
from .model import MaskedFusionNet

LABELS = ["负向", "中性", "正向"]
MODES = ["文本", "语音", "视觉"]
ROOT = Path(__file__).resolve().parent.parent


class Engine:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self._models = {}
        self.lock = threading.RLock()
        self.normalization = json.loads((self.root / "assets/normalization.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((self.root / "assets/provenance.json").read_text(encoding="utf-8"))
        torch.set_num_threads(min(4, torch.get_num_threads()))

    def _model(self, name):
        if name not in self._models:
            path = self.root / "models" / f"{name}.pt"
            expected = self.manifest["checkpoints"][path.name]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError("模型文件校验失败，请重新下载原始模型文件。")
            model = MaskedFusionNet()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self._models[name] = model.eval()
        return self._models[name]

    def tensors(self, sample):
        arrays = sample.arrays
        converted = [torch.from_numpy(arrays["text"][None].copy())]
        for name in ("audio", "vision"):
            raw = arrays[name]
            stats = self.normalization[name]
            z = (raw - np.array(stats["mean"], np.float32)) / np.array(stats["std"], np.float32)
            z = np.clip(z, -10, 10)
            z[np.all(raw == 0, axis=-1)] = 0
            converted.append(torch.from_numpy(z[None].astype(np.float32)))
        converted.append(torch.from_numpy(sample.availability()[None].copy()))
        return converted

    def forward(self, inputs, predictor):
        if predictor == "q3":
            logit, reg = self._model("q3_predictor")(*inputs)
            return logit.softmax(-1).numpy(), reg.clamp(-3, 3).numpy()
        if predictor != "ensemble":
            raise ValueError("未知预测器。")
        logit_b, reg_b = self._model("q2_baseline")(*inputs)
        logit_r, reg_r = self._model("q2_robust")(*inputs)
        prob = .75 * logit_b.softmax(-1) + .25 * logit_r.softmax(-1)
        reg = (.75 * reg_b + .25 * reg_r).clamp(-3, 3)
        return prob.numpy(), reg.numpy()

    @staticmethod
    def describe(prob, reg):
        cls = int(np.argmax(prob))
        return {"class_id": cls, "label": LABELS[cls], "strength": float(reg),
                "probabilities": prob.tolist(), "confidence": float(prob[cls])}

    def shapley(self, inputs, predictor, target):
        variants = [x.repeat(8, *([1] * (x.ndim - 1))) for x in inputs]
        for subset in range(8):
            for j in range(3):
                if not subset & (1 << j):
                    variants[j][subset] = 0
                    variants[3][subset, :, j] = False
        probs, _ = self.forward(variants, predictor)
        values = probs[:, target].astype(float)
        phi = np.zeros(3)
        for j in range(3):
            for subset in range(8):
                if not subset & (1 << j):
                    k = subset.bit_count()
                    phi[j] += math.factorial(k) * math.factorial(2-k) / 6 * (values[subset | (1 << j)] - values[subset])
        magnitude = np.abs(phi)
        relative = magnitude / magnitude.sum() if magnitude.sum() > 1e-10 else np.zeros(3)
        residual = float(phi.sum() - (values[7] - values[0]))
        if abs(residual) > 1e-5:
            raise RuntimeError("模态贡献守恒校验失败。")
        return {"signed": phi.tolist(), "relative": relative.tolist(),
                "primary": MODES[int(relative.argmax())] if relative.sum() else "无可归因差异",
                "coalitions": values.tolist(), "empty_probability": float(values[0]),
                "full_probability": float(values[7]), "efficiency_error": abs(residual),
                "target": LABELS[target]}

    def local_effects(self, inputs, predictor, length, target, original):
        stop = length - 1
        width = min(3, stop - 1)
        starts = list(range(1, stop - width + 1, 2))
        if not starts or starts[-1] != stop - width:
            starts.append(stop - width)
        result = {}
        for name, j in (("audio", 1), ("vision", 2)):
            if not bool(inputs[3][0, 1:length-1, j].any()):
                result[name] = []
                continue
            variants = [x.repeat(len(starts), *([1] * (x.ndim - 1))) for x in inputs]
            for i, start in enumerate(starts):
                variants[j][i, start:start + width] = 0
                variants[3][i, start:start + width, j] = False
            probs, _ = self.forward(variants, predictor)
            result[name] = [{"start": int(s), "stop": int(s + width),
                             "delta": float(original - probs[i, target])} for i, s in enumerate(starts)]
        return result

    def analyze(self, sample: Features, predictor="q3", mode="none", rate=0.0, location="middle"):
        started = time.perf_counter()
        altered, positions = mask_segment(sample, mode, rate, location)
        with self.lock, torch.inference_mode():
            original_inputs = self.tensors(sample)
            inputs = self.tensors(altered)
            before_prob, before_reg = self.forward(original_inputs, predictor)
            probs, reg = self.forward(inputs, predictor)
            result = self.describe(probs[0], reg[0])
            attribution = self.shapley(inputs, predictor, result["class_id"])
            local = self.local_effects(inputs, predictor, altered.length, result["class_id"], result["confidence"])
        return {"source": "live", "predictor": predictor, "synthetic": sample.synthetic,
                "prediction": result, "baseline": self.describe(before_prob[0], before_reg[0]),
                "intervention": {"mode": mode, "rate": rate, "location": location, "positions": positions},
                "attribution": attribution, "local_effects": local, "features": feature_view(altered),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "notes": ["类别概率未经校准；不是预测正确率。", "归因解释模型在指定遮挡基线下的响应，不表示因果关系。",
                          "音视频缺失为实时模拟；局部文本缺失需重新编码 BERT，本版本不执行此操作。"]}
