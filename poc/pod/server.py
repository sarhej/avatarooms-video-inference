"""LTX-2 POC server — FastAPI wrapper around the diffusers LTX2 pipeline.

Single-process, single-GPU. Designed for short-lived RunPod rentals to
evaluate LTX-2 inference performance, quality, and parallel-batching
headroom before committing to a production self-host architecture.

NOT FOR PRODUCTION USE. No persistent storage, no Firestore, no auth
beyond a static bearer token, no queue, no retry. The runner machine
drives load and captures all metrics.

Endpoints
---------
GET  /healthz   always 200 (liveness)
GET  /readyz    200 once model is loaded, else 503
GET  /info      model variant, dtype, peak VRAM, throughput counters
POST /generate  text-to-video, optionally batched via num_videos_per_prompt

Environment
-----------
POD_AUTH_TOKEN  required, shared secret with the runner
LTX2_VARIANT    distilled-fp8 (default) | dev | distilled-bf16
LTX2_REVISION   optional HF revision/tag (default main)
HF_HOME         HF cache dir (default /workspace/hf)
HF_TOKEN        optional HF token for gated weights
PORT            default 8000
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("ltx2-pod")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POD_AUTH_TOKEN = os.environ.get("POD_AUTH_TOKEN", "")
LTX2_VARIANT = os.environ.get("LTX2_VARIANT", "distilled-fp8").strip()
LTX2_REVISION = os.environ.get("LTX2_REVISION", "main").strip()
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf")
PORT = int(os.environ.get("PORT", "8000"))

# Variant → (huggingface model id, torch dtype, recommended sampler steps)
VARIANT_CONFIG: dict[str, dict[str, Any]] = {
    "distilled-fp8": {
        "model_id": "Lightricks/LTX-2",
        "weights_file": "ltx-2.3-22b-distilled-fp8.safetensors",
        "dtype": "fp8_e4m3",
        "num_inference_steps": 8,
        "needs_ada_or_newer": True,
    },
    "distilled-bf16": {
        "model_id": "Lightricks/LTX-2",
        "weights_file": "ltx-2.3-22b-distilled-1.1.safetensors",
        "dtype": "bfloat16",
        "num_inference_steps": 8,
        "needs_ada_or_newer": False,
    },
    "dev": {
        "model_id": "Lightricks/LTX-2",
        "weights_file": "ltx-2.3-22b-dev.safetensors",
        "dtype": "bfloat16",
        "num_inference_steps": 30,
        "needs_ada_or_newer": False,
    },
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    pipeline: Any = None
    variant: str = ""
    dtype: str = ""
    model_loaded_at_ms: int | None = None
    load_duration_ms: int | None = None
    gpu_name: str = ""
    gpu_total_vram_mb: int = 0
    counters: dict[str, Any] = field(
        default_factory=lambda: {
            "generations_total": 0,
            "generations_failed": 0,
            "videos_produced_total": 0,
            "inference_seconds_total": 0.0,
            "peak_vram_mb": 0,
        }
    )


STATE = PipelineState()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """LTX-2 generation request.

    Field names mirror the diffusers `LTX2Pipeline.__call__` signature
    so the server is a near-1:1 passthrough.
    """

    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = ""
    duration_seconds: int = Field(default=10, ge=2, le=20)
    resolution: str = Field(default="1080p", pattern="^(720p|1080p)$")
    aspect_ratio: str = Field(default="9:16", pattern="^(9:16|16:9|1:1)$")
    fps: int = Field(default=24, ge=8, le=30)
    seed: int = 42
    num_videos_per_prompt: int = Field(default=1, ge=1, le=4)
    audio_enabled: bool = True
    guidance_scale_video: float = 3.0
    guidance_scale_audio: float = 7.0
    cross_modal_guidance_scale: float = 3.0


class VideoOutput(BaseModel):
    index: int
    data_b64: str
    frames: int
    width: int
    height: int
    fps: int
    has_audio: bool


class GenerateMetrics(BaseModel):
    wall_clock_ms: int
    inference_ms: int
    decode_ms: int
    mux_ms: int
    peak_vram_mb: int
    num_inference_steps: int
    batch_size: int


class GenerateResponse(BaseModel):
    variant: str
    dtype: str
    videos: list[VideoOutput]
    metrics: GenerateMetrics


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="LTX-2 POC server", version="0.1.0")


@app.on_event("startup")
def _load_model_on_startup() -> None:
    """Load the LTX-2 pipeline once at process start.

    Lazy-loading on first /generate would block the runner for ~60-120s
    on the first request and skew our latency measurements. Loading
    upfront makes /readyz the source of truth for "is the pod hot?".
    """
    if not POD_AUTH_TOKEN:
        log.warning("POD_AUTH_TOKEN not set — server will reject all generate calls")

    cfg = VARIANT_CONFIG.get(LTX2_VARIANT)
    if cfg is None:
        log.error("Unknown LTX2_VARIANT=%r. Valid: %s", LTX2_VARIANT, list(VARIANT_CONFIG))
        sys.exit(2)

    log.info("Loading LTX-2 variant=%s dtype=%s …", LTX2_VARIANT, cfg["dtype"])
    t0 = time.time()

    import torch
    from diffusers import LTX2Pipeline

    STATE.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda"
    STATE.gpu_total_vram_mb = (
        int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        if torch.cuda.is_available()
        else 0
    )
    log.info("GPU: %s, total VRAM: %d MB", STATE.gpu_name, STATE.gpu_total_vram_mb)

    if cfg["dtype"] == "fp8_e4m3":
        # FP8 cannot be a global torch default dtype — PyTorch 2.8 has FP8
        # *tensor* types but no Float8_e4m3fnStorage backend, so
        # `torch.set_default_dtype(torch.float8_e4m3fn)` raises TypeError.
        # diffusers' default behavior is to cascade torch_dtype into every
        # sub-module, including the T5 text encoder (a `transformers` model
        # which calls set_default_dtype during from_pretrained → crash).
        # Workaround: dict-typed torch_dtype puts FP8 only on the diffusion
        # transformer; text encoder + VAE stay in BF16.
        torch_dtype_param: Any = {
            "transformer": torch.float8_e4m3fn,
            "text_encoder": torch.bfloat16,
            "vae": torch.bfloat16,
            "default": torch.bfloat16,
        }
    else:
        torch_dtype_param = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[cfg["dtype"]]

    pipe = LTX2Pipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch_dtype_param,
        cache_dir=HF_HOME,
        revision=LTX2_REVISION,
    )
    pipe = pipe.to("cuda")

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        # Tiled VAE decode keeps activation memory under control during the
        # final upscale stage. Free real estate for batching.
        pipe.vae.enable_tiling()

    STATE.pipeline = pipe
    STATE.variant = LTX2_VARIANT
    STATE.dtype = cfg["dtype"]
    STATE.load_duration_ms = int((time.time() - t0) * 1000)
    STATE.model_loaded_at_ms = int(time.time() * 1000)
    log.info("Pipeline ready in %.1fs", STATE.load_duration_ms / 1000.0)


def _require_auth(authorization: str | None) -> None:
    expected = f"Bearer {POD_AUTH_TOKEN}"
    if not POD_AUTH_TOKEN or authorization != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


def _peak_vram_mb() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        return 0


def _reset_peak_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _resolution_to_hw(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    base = {"720p": 720, "1080p": 1080}[resolution]
    if aspect_ratio == "9:16":
        return (base * 9 // 16, base)
    if aspect_ratio == "16:9":
        return (base, base * 9 // 16)
    return (base, base)


def _encode_video(
    frames: Any,
    audio: Any,
    fps: int,
    out_dir: Path,
    index: int,
) -> tuple[Path, int, int, bool]:
    """Write a numpy/tensor frame sequence (+ optional audio) to mp4.

    Returns (path, mux_ms, frame_count, has_audio).
    """
    from diffusers.utils import export_to_video

    silent_path = out_dir / f"video_{index}_silent.mp4"
    export_to_video(frames, str(silent_path), fps=fps)
    frame_count = len(frames) if hasattr(frames, "__len__") else 0

    if audio is None:
        return silent_path, 0, frame_count, False

    audio_path = out_dir / f"audio_{index}.wav"
    _write_audio_wav(audio, audio_path)

    final_path = out_dir / f"video_{index}.mp4"
    mux_t0 = time.time()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(silent_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(final_path),
        ],
        check=True,
    )
    mux_ms = int((time.time() - mux_t0) * 1000)
    return final_path, mux_ms, frame_count, True


def _select_video_frames(videos_frames: Any, i: int) -> Any:
    """Pick the i-th video out of a diffusers LTX-2 frames result.

    The pipeline returns either a list (length = num_videos_per_prompt)
    of per-video frame arrays, OR a single tensor/array with batch as
    the leading dimension when num_videos_per_prompt > 1. Handle both.
    """
    if isinstance(videos_frames, list):
        return videos_frames[i]
    if hasattr(videos_frames, "ndim") and videos_frames.ndim == 5:
        return videos_frames[i]
    return videos_frames


def _write_audio_wav(audio: Any, dest: Path) -> None:
    """Write LTX-2 audio output to a WAV file at 24 kHz stereo.

    The diffusers LTX2 pipeline returns audio as a torch tensor of shape
    `(channels, samples)` or `(samples,)`. We accept both.
    """
    import soundfile as sf
    import torch

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    elif audio.ndim == 2 and audio.shape[0] in (1, 2) and audio.shape[1] > 2:
        audio = audio.T
    sf.write(str(dest), audio, samplerate=24000)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> JSONResponse:
    if STATE.pipeline is None:
        return JSONResponse(
            {"status": "not_ready", "reason": "pipeline not loaded"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return JSONResponse({"status": "ready", "variant": STATE.variant})


@app.get("/info")
def info() -> dict[str, Any]:
    return {
        "variant": STATE.variant,
        "dtype": STATE.dtype,
        "gpu_name": STATE.gpu_name,
        "gpu_total_vram_mb": STATE.gpu_total_vram_mb,
        "model_loaded_at_ms": STATE.model_loaded_at_ms,
        "load_duration_ms": STATE.load_duration_ms,
        "counters": STATE.counters,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(
    request_body: GenerateRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> GenerateResponse:
    _require_auth(authorization)
    if STATE.pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not loaded")

    cfg = VARIANT_CONFIG[STATE.variant]
    width, height = _resolution_to_hw(request_body.resolution, request_body.aspect_ratio)
    num_frames = request_body.duration_seconds * request_body.fps

    log.info(
        "generate variant=%s prompt=%r dur=%ds res=%s ar=%s batch=%d audio=%s seed=%d",
        STATE.variant,
        request_body.prompt[:60] + ("…" if len(request_body.prompt) > 60 else ""),
        request_body.duration_seconds,
        request_body.resolution,
        request_body.aspect_ratio,
        request_body.num_videos_per_prompt,
        request_body.audio_enabled,
        request_body.seed,
    )

    import torch

    _reset_peak_vram()
    wall_t0 = time.time()

    try:
        generator = torch.Generator(device="cuda").manual_seed(request_body.seed)
        inf_t0 = time.time()
        with torch.no_grad():
            result = STATE.pipeline(
                prompt=request_body.prompt,
                negative_prompt=request_body.negative_prompt or None,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=cfg["num_inference_steps"],
                num_videos_per_prompt=request_body.num_videos_per_prompt,
                guidance_scale=request_body.guidance_scale_video,
                generator=generator,
                output_type="np",
                return_dict=True,
            )
        inference_ms = int((time.time() - inf_t0) * 1000)

        videos_frames = getattr(result, "frames", None) or result["frames"]
        audios = getattr(result, "audio", None) if request_body.audio_enabled else None
        if isinstance(audios, (list, tuple)) and len(audios) != request_body.num_videos_per_prompt:
            audios = None
        if audios is None:
            audios = [None] * request_body.num_videos_per_prompt

        decode_t0 = time.time()
        outputs: list[VideoOutput] = []
        total_mux_ms = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for i in range(request_body.num_videos_per_prompt):
                frames_i = _select_video_frames(videos_frames, i)
                path, mux_ms, frame_count, has_audio = _encode_video(
                    frames=frames_i,
                    audio=audios[i],
                    fps=request_body.fps,
                    out_dir=tmp,
                    index=i,
                )
                total_mux_ms += mux_ms
                outputs.append(
                    VideoOutput(
                        index=i,
                        data_b64=base64.b64encode(path.read_bytes()).decode("ascii"),
                        frames=frame_count,
                        width=width,
                        height=height,
                        fps=request_body.fps,
                        has_audio=has_audio,
                    )
                )
        decode_ms = int((time.time() - decode_t0) * 1000) - total_mux_ms
        wall_ms = int((time.time() - wall_t0) * 1000)
        peak_vram = _peak_vram_mb()

        STATE.counters["generations_total"] += 1
        STATE.counters["videos_produced_total"] += len(outputs)
        STATE.counters["inference_seconds_total"] += inference_ms / 1000.0
        STATE.counters["peak_vram_mb"] = max(STATE.counters["peak_vram_mb"], peak_vram)

        return GenerateResponse(
            variant=STATE.variant,
            dtype=STATE.dtype,
            videos=outputs,
            metrics=GenerateMetrics(
                wall_clock_ms=wall_ms,
                inference_ms=inference_ms,
                decode_ms=decode_ms,
                mux_ms=total_mux_ms,
                peak_vram_mb=peak_vram,
                num_inference_steps=cfg["num_inference_steps"],
                batch_size=request_body.num_videos_per_prompt,
            ),
        )

    except Exception as exc:
        STATE.counters["generations_failed"] += 1
        log.exception("generate failed: %r", exc)
        raise HTTPException(status_code=500, detail=f"generation_failed: {exc!r}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
