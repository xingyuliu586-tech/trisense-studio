# GitHub Pages 网页版

网址：https://xingyuliu586-tech.github.io/trisense-studio/

仓库：[xingyuliu586-tech/trisense-studio](https://github.com/xingyuliu586-tech/trisense-studio)。网页从仓库 `main` 分支的 `/docs` 目录发布。

网页会从 GitHub Pages 下载三个自训练下游模型的数值参数、训练归一化统计和历史结果，然后在当前浏览器内执行推理、精确 Shapley 和音视频遮挡实验。上传的 NPZ 在浏览器内解析与计算；静态站点没有接收样本的后端接口。浏览器仍需联网获取网页与模型文件，首次加载模型可能比本地 Python 版稍慢。

网页版支持 Q3 独立模型、Q2 两模型集成、兼容 NPZ 导入、20 条历史结果浏览、四架构实验图表、JSON/CSV 导出。浏览器 CPU 数值实现与 Python 版用四种输入条件对比，预测类别一致，概率、强度和模态贡献误差小于 0.002。这个检验使用合成样例，不等于重新评价真实数据集。

Pages 只支持静态托管，不能执行仓库中的 FastAPI 服务。`/api/*` 路径仅用于下载后运行的 Python 版；网页内部使用同一界面约定和浏览器计算适配器。原始比赛数据没有公开；20 条专项样本仍是历史结果，未导入特征时不能重新推理。

变更前端或模型后，执行 `python scripts/prepare_pages.py` 重新生成 `/docs` 网页资源，再运行 `node tests/browser_parity.cjs`，最后发布更新。Pages 设置需选 `Deploy from a branch` → `main` → `/docs`。仓库根目录仍是完整 Python 软件。
