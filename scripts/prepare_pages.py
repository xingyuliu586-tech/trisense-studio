"""Build GitHub Pages assets from the tested local application and numeric data."""
import base64
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'docs'


def main():
    index=(ROOT/'web/index.html').read_text(encoding='utf-8')
    index=index.replace('/static/mark.svg','./mark.svg').replace('/static/style.css','./style.css')
    index=index.replace('<script src="/static/app.js" defer></script>',
        '<script src="./browser-engine.js" defer></script>\n'
        '  <script src="./static-api.js" defer></script>\n'
        '  <script src="./app.js" defer></script>')
    index=index.replace('v1.0.0 · CPU 推理 · 无云端调用',
                        'v1.0.0 · 浏览器推理 · 文件不上传服务器')
    index=index.replace('本地计算','浏览器计算')
    index=index.replace('<span class="top-tag">多模态情感分析</span>',
                        '<a class="top-tag" href="https://github.com/xingyuliu586-tech/trisense-studio" target="_blank" rel="noopener">GitHub 源码 ↗</a>')
    index=index.replace('本机已导入特征，可以重新推理。','可重新计算。')
    index=index.replace('上传文件只在内存中处理，软件不向外部服务发送数据。',
                        '导入的 NPZ 在浏览器内处理；静态站点不接收上传文件。')
    index=index.replace('服务默认仅监听 127.0.0.1；没有账户系统，不应直接作为公网服务部署。',
                        '网页版没有服务器推理；运行位置是当前浏览器。完整 Python 桌面版见 GitHub 仓库。')
    index=index.replace('href="/api/example"','href="./synthetic_sample.npz"')
    index=index.replace('href="/docs"','href="./api.html"')
    index=index.replace('本地 API 文档 ↗','本地版 API 说明 ↗')
    index=index.replace('实时推理路径','浏览器推理路径')
    index=index.replace('无需在线模型服务','模型参数在浏览器加载')
    (SITE/'index.html').write_text(index,encoding='utf-8')
    api=(ROOT/'web/api.html').read_text(encoding='utf-8').replace('href="/static/style.css"','href="./style.css"').replace('href="/"','href="./"')
    api=api.replace('<h1>TriSense Studio API</h1>',
                    '<h1>TriSense Studio 本地版 API</h1>')
    api=api.replace('本页完全本地加载。',
                    '此文档描述下载后运行的 Python 本地版；GitHub Pages 网页直接在浏览器中运算。')
    api=api.replace('<a href="/openapi.json">下载 OpenAPI 规范</a>',
                    '下载并启动 Python 版后，在 /openapi.json 查看机器可读规范')
    (SITE/'api.html').write_text(api,encoding='utf-8')
    for name in ('style.css','app.js','mark.svg'):
        shutil.copy2(ROOT/'web'/name,SITE/name)
    for name in ('research.json','cases.json'):
        shutil.copy2(ROOT/'assets'/name,SITE/name)
    shutil.copy2(ROOT/'examples/synthetic_sample.npz',SITE/'synthetic_sample.npz')
    (SITE/'.nojekyll').write_text('',encoding='utf-8')
    with np.load(ROOT/'examples/synthetic_sample.npz',allow_pickle=False) as sample:
        plain={key:sample[key].tolist() for key in sample.files}
    (SITE/'synthetic_sample.json').write_text(json.dumps(plain,separators=(',',':')),encoding='utf-8')
    modeldir=SITE/'browser-models';modeldir.mkdir(exist_ok=True)
    for name in ('q3_predictor','q2_baseline','q2_robust'):
        state=torch.load(ROOT/'models'/f'{name}.pt',map_location='cpu',weights_only=True)
        layers={}
        for key,tensor in state.items():
            arr=tensor.detach().cpu().numpy().astype('<f4',copy=False)
            layers[key]={'shape':list(arr.shape),'data':base64.b64encode(arr.tobytes()).decode('ascii')}
        (modeldir/f'{name}.json').write_text(json.dumps(layers,separators=(',',':')),encoding='utf-8')
    stats=json.loads((ROOT/'assets/normalization.json').read_text(encoding='utf-8'))
    (SITE/'normalization.json').write_text(json.dumps(stats,separators=(',',':')),encoding='utf-8')
    print('Prepared static GitHub Pages site in docs/.')


if __name__=='__main__':main()
