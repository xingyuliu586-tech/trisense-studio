# GitHub 发布指南

仓库：[xingyuliu586-tech/trisense-studio](https://github.com/xingyuliu586-tech/trisense-studio)。topics 可使用 `pytorch`、`fastapi`、`multimodal`、`sentiment-analysis`、`explainable-ai`。

只上传这个软件目录，不能直接把整个数学建模工作区加入仓库。软件目录已经包含忽略规则；发布打包使用白名单，进一步隔离原数据和本机环境。

仓库根目录应能直接看到 README.md、run.py 和 trisense/，避免再套一层无意义的文件夹。静态网页版位于 `docs/`；GitHub Pages 的发布源设为 `main` 分支的 `/docs` 文件夹。

使用 Git 时，在解压后的独立软件目录执行：

```bash
git init
git add README.md LICENSE requirements.txt requirements-dev.txt run.py start.cmd setup.ps1 .gitignore .github trisense web models assets examples tests scripts docs
git status --short
git commit -m "Build local multimodal sentiment analysis workbench"
git branch -M main
git remote add origin https://github.com/xingyuliu586-tech/trisense-studio.git
git push -u origin main
```

提交前确认列表没有 `local_data`、私人文件或本机环境。模型文件每个约 0.67 MB，不需要 Git LFS。CI 推送后才会运行；没有远程结果时，不在 README 放“CI passing”徽章。

演示顺序：

1. 首页说明研究来源和内部验证指标。
2. 历史样本展示可追溯的预测与解释记录。
3. 合成样例运行 Q3，再切换 Q2 集成；说明这是接口演示。
4. 100% 遮挡视觉，检查视觉贡献和局部窗口消失。
5. 四架构对比与训练协议，最后导出结果。

如有可合法使用的原始特征，在本机导入后演示真实样本重新推理会更有说服力。未导入时，不能剪辑成“原视频现场识别”。

生成发布包：`python scripts/package_release.py`。脚本把 ZIP 和 SHA-256 放到被忽略的 `dist/`，不会自动联网或推送仓库。
