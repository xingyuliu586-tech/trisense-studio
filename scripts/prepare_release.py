"""Export aggregate results + owned checkpoints from an existing research workspace.

Never exports raw videos, transcripts, original features, sample labels or local paths.
Run only when refreshing the bundled research snapshot, not on ordinary startup.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trisense.features import SCHEMA


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.research_root.resolve()
    out = source / "output"
    q2 = read_json(out / "q2/metrics.json")
    q3 = read_json(out / "q3/metrics.json")
    config = read_json(out / "q2/config.json")
    norm = config["normalization"]
    if read_json(out / "q3/config.json")["normalization"] != norm:
        raise ValueError("Q2/Q3 normalization differs; store separate scalers before exporting.")
    (ROOT / "models").mkdir(exist_ok=True)
    checkpoint_hashes = {}
    for question, name in (("q2", "q2_baseline"), ("q2", "q2_robust"), ("q3", "q3_predictor")):
        path = out / question / f"{name}.pt"
        shutil.copy2(path, ROOT / "models" / path.name)
        checkpoint_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_names = ["q1/audit.json", "q2/config.json", "q2/metrics.json", "q3/metrics.json",
                    "q3/附件4_第三题预测与解释.csv", "paper/comparison/summary.csv",
                    "paper/comparison/stress_summary.csv", "paper/comparison/protocol.json"]
    provenance = {"schema": 1, "snapshot_date": "2026-10-27", "checkpoints": checkpoint_hashes,
                  "source_hashes": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in source_names},
                  "policy": "Only derived numeric results, experiment settings and own trained checkpoints. No original dataset is redistributed."}
    comparison = []
    for row in read_csv(out / "paper/comparison/summary.csv"):
        comparison.append({k: v if k == "model" else float(v) for k, v in row.items()})
    stress = []
    for row in read_csv(out / "paper/comparison/stress_summary.csv"):
        stress.append({k: v if k in ("model", "scenario") else float(v) for k, v in row.items()})
    records = []
    for row in read_csv(out / "q3/附件4_第三题预测与解释.csv"):
        signed = [float(row[f"{m}有向贡献"]) for m in ("文本", "语音", "视觉")]
        relative = [float(row[f"{m}作用程度"]) for m in ("文本", "语音", "视觉")]
        records.append({"id": f"S{row['样本编号']}", "source": "recorded", "predictor": "q3", "synthetic": False,
                        "prediction": {"class_id": int(row["情感极性编码"]), "label": row["情感极性"],
                                       "strength": float(row["情感强度"]), "confidence": float(row["预测类别概率_未校准"]),
                                       "probabilities": None},
                        "attribution": {"signed": signed, "relative": relative, "primary": row["主要参考模态"],
                                        "efficiency_error": None, "target": row["情感极性"]},
                        "evidence": {"text_positions": row["文本证据位置"], "audio_seconds": row["语音关键时段秒"],
                                     "vision_seconds": row["视觉关键时段秒"], "vision_frame": row["视觉关键帧编号"],
                                     "vision_status": row["视觉证据状态"]},
                        "features": None, "local_effects": {}, "baseline": None,
                        "notes": ["原实验结果快照，尚未在当前界面执行推理。", "原始视频、文本与特征不包含在公开仓库中。"]})
    report = {"snapshot_date": "2026-10-27", "train_n": 3395, "valid_n": 728,
              "q1": read_json(out / "q1/audit.json"), "q2": q2["validation_selected"],
              "q3": q3["validation"], "attachment4": q3["attachment4"],
              "comparison": comparison, "stress": stress,
              "protocol": read_json(out / "paper/comparison/protocol.json"),
              "training": {name: q2[name]["history"] for name in ("baseline", "robust")},
              "selection": q2["selection"],
              "limitations": ["所有报告性能均为附件二内部验证指标；验证集参与选模，不是独立测试成绩。",
                              "附件三与附件四没有真实标签，只能展示预测和解释。",
                              "四模型对比使用三随机种子、无缺失增强；Q2 的最终集成为另一组实验。",
                              "自提取的音视频特征与题给特征定义不同，不可仅凭维度相同混用。"]}
    write_json(ROOT / "assets/research.json", report)
    write_json(ROOT / "assets/cases.json", records)
    write_json(ROOT / "assets/normalization.json", norm)
    write_json(ROOT / "assets/provenance.json", provenance)
    (ROOT / "examples").mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    length = 20
    tokens = np.zeros((3, 50), np.int64)
    tokens[0, :length] = np.r_[101, np.arange(2000, 2000 + length - 2), 102]
    tokens[1, :length] = 1
    features = {"text": rng.normal(0, .3, (50, 768)).astype(np.float32)}
    features["text"][length:] = 0
    for name, dim in (("audio", 74), ("vision", 35)):
        arr = np.zeros((50, dim), np.float32)
        arr[1:length-1] = (rng.normal(0, .5, (length-2, dim)) * np.array(norm[name]["std"]) + np.array(norm[name]["mean"])).astype(np.float32)
        features[name] = arr
    np.savez_compressed(ROOT / "examples/synthetic_sample.npz", schema=np.array(SCHEMA), synthetic=np.array(True), text_bert=tokens, **features)
    print("Prepared 3 checkpoints, aggregate experiment snapshot, 20 de-identified numeric records and a synthetic API example.")


if __name__ == "__main__":
    main()
