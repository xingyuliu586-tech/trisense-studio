"""Local-only HTTP application. No cloud calls, persistent uploads or pickle API."""
from functools import lru_cache
import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .engine import Engine, ROOT
from .features import FeatureError, MAX_UPLOAD_BYTES, feature_view, read_features

Predictor = Literal["q3", "ensemble"]
Mode = Literal["none", "audio", "vision", "audio+vision"]
Location = Literal["early", "middle", "late"]


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str = Field(default="synthetic", pattern=r"^(S(0[1-9]|1[0-9]|20)|synthetic)$")
    predictor: Predictor = "q3"
    mode: Mode = "none"
    rate: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    location: Location = "middle"


class BodyLimitMiddleware:
    """Bound actual body bytes, including chunked multipart, before multipart parsing."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            chunk = event.get("body", b"")
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES + 1024 * 1024:
                return await JSONResponse({"detail": "上传请求超过大小限制（文件上限 8 MB）。"}, 413)(scope, receive, send)
            chunks.append(chunk)
            if not event.get("more_body", False):
                break
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


@lru_cache(maxsize=1)
def engine():
    return Engine()


def load_asset(name):
    return json.loads((ROOT / "assets" / name).read_text(encoding="utf-8"))


def sample_path(sample_id):
    if sample_id == "synthetic":
        return ROOT / "examples/synthetic_sample.npz"
    known = {r["id"] for r in load_asset("cases.json")}
    if sample_id not in known:
        raise HTTPException(404, "未找到该样本。")
    return ROOT / "local_data" / f"{sample_id}.npz"


app = FastAPI(title="TriSense Studio", version=__version__, description="本地多模态情感分析与解释 API。", docs_url=None, redoc_url=None)
app.add_middleware(BodyLimitMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def browser_boundary(request, call_next):
    origin = request.headers.get("origin")
    if request.method == "POST" and origin and urlsplit(origin).netloc.lower() != request.headers.get("host", "").lower():
        return JSONResponse({"detail": "只接受来自当前本地页面的操作。"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(FeatureError)
async def feature_error(_request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.get("/api/health")
def health():
    ready = all((ROOT / "models" / f"{name}.pt").exists() for name in ("q3_predictor", "q2_baseline", "q2_robust"))
    return {"status": "ok", "version": __version__, "models_present": ready, "device": "CPU", "network_required": False}


@app.get("/api/research")
def research():
    return load_asset("research.json")


@app.get("/api/samples")
def samples():
    return [{"id": row["id"], "label": row["prediction"]["label"],
             "strength": row["prediction"]["strength"], "live_available": sample_path(row["id"]).exists(),
             "synthetic": False} for row in load_asset("cases.json")] + [
                 {"id": "synthetic", "label": "合成接口样例", "live_available": True, "synthetic": True}]


@app.get("/api/samples/{sample_id}")
def sample_record(sample_id: str):
    path = sample_path(sample_id)
    if sample_id == "synthetic":
        return {"id": sample_id, "source": "synthetic", "synthetic": True, "predictor": "q3",
                "prediction": None, "attribution": None, "features": feature_view(read_features(path)),
                "live_available": True, "notes": ["随机生成的特征，仅验证软件流程；不代表真实视频情感。"]}
    record = next(r for r in load_asset("cases.json") if r["id"] == sample_id)
    record["live_available"] = path.exists()
    if path.exists():
        record["features"] = feature_view(read_features(path))
    return record


@app.post("/api/analyze")
def analyze(request: AnalysisRequest):
    path = sample_path(request.sample_id)
    if not path.exists():
        raise HTTPException(409, "本机尚未导入此样本的特征。请上传兼容 NPZ，或选择合成接口样例试运行。")
    result = engine().analyze(read_features(path), request.predictor, request.mode, request.rate, request.location)
    result["id"] = request.sample_id
    return result


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), predictor: Predictor = Form("q3"),
                 mode: Mode = Form("none"), rate: float = Form(0.0, ge=0, le=1), location: Location = Form("middle")):
    if not (file.filename or "").lower().endswith(".npz"):
        raise HTTPException(415, "仅接受 .npz 数值特征文件；不接受视频或 pickle 文件。")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    # The JSON and multipart routes share the same finite, bounded parameter validation.
    import math
    if not math.isfinite(rate):
        raise HTTPException(422, "缺失比例必须为有限数值。")
    sample = read_features(raw)
    result = await run_in_threadpool(engine().analyze, sample, predictor, mode, rate, location)
    result["id"] = "uploaded"
    return result


@app.get("/api/example")
def example():
    return FileResponse(ROOT / "examples/synthetic_sample.npz", filename="synthetic_sample.npz")


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/docs", include_in_schema=False)
def api_docs():
    return FileResponse(ROOT / "web/api.html")


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
