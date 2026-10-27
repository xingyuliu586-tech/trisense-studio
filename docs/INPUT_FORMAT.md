# 单样本输入规范

模型采用固定的词元级对齐接口。`schema` 是输入方对兼容性的声明；软件校验结构与数值，不能自动证明任意特征与原训练分布相同。

| 字段 | 形状 | 内容 |
|---|---|---|
| schema | 标量字符串 | `mosei-aligned-50-v1` |
| text | (50, 768) | 与原题给特征一致的 BERT 隐藏状态 |
| audio | (50, 74) | 与原题给通道顺序、定义一致的原尺度声学特征 |
| vision | (50, 35) | 与原题给通道顺序、定义一致的原尺度视觉特征 |
| text_bert | (3, 50) | input_ids、attention_mask、token_type_ids |
| synthetic | 可选布尔标量 | 合成样例必须为 true |

数组禁止对象类型，数值必须有限。token ID 处于 0～30521；mask / segment ID 为 0 或 1。有效 mask 是连续前缀，长度 3～50，约定首尾为 CLS / SEP，内容位于 `[1, L−1)`。声学 / 视觉填充位置必须为零；整行零代表不可用。文本可用性由 mask 和 ID!=100（UNK）判定。

声学、视觉输入不能预先用自己的数据再次标准化。软件使用 `assets/normalization.json` 中仅由训练集非零行计算的均值和标准差，截断 z-score 至 ±10，并恢复原全零行。网页热图的样本内标准化仅影响显示，不进入模型。

NPZ 文件与解压后的数组合计均最多 8 MB。软件只读取四个必要数组与元信息，不解压到文件系统，也不保存上传文件。不要包含身份、转写、视频等无关信息。

生成文件示例（这里的变量应来自兼容特征，而非任意拼接）：

```python
import numpy as np
np.savez_compressed(
    'sample.npz', schema=np.array('mosei-aligned-50-v1'),
    synthetic=np.array(False), text=text, audio=audio,
    vision=vision, text_bert=text_bert,
)
```

真实数据没有特征时不能用随机数代替再去掉 synthetic 标记。随附 `examples/synthetic_sample.npz` 始终作为接口演示数据。

导出索引全部从 0 开始。局部窗口的 stop 是右开区间。界面缺失比例以 `L−2` 个内容位置为分母，取最近整数，正比例至少遮挡一个位置；原本缺失的位置包含在选中区间内，因此“叠加比例”并不保证新增同等数量的缺失行。
