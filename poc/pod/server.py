"""LTX-2 POC server — FastAPI wrapper around the diffusers LTX2 two-stage pipeline.

Single-process, single-GPU. Designed for short-lived RunPod rentals to
evaluate LTX-2 inference performance, quality, and headroom before
committing to a production self-host architecture.

Implements the production-quality two-stage flow per the official LTX-2
model card and diffusers LTX2Pipeline docs:

  Stage 1: base LTX-2 (Lightricks/LTX-2, 19B BF16) at HALF resolution,
           40 steps, CFG 4.0, default scheduler, distilled LoRA
           DISABLED. Output: video + audio latents.
  Stage 1.5: LTX2LatentUpsamplePipeline doubles spatial dimensions.
  Stage 2: same base pipe, FlowMatchEulerDiscreteScheduler (no dynamic
           shifting), distilled LoRA ENABLED, 3 steps, CFG 1.0, sigmas
           = STAGE_2_DISTILLED_SIGMA_VALUES, noise_scale = sigmas[0].
           Output: video frames (np) + audio tensor.

NOT FOR PRODUCTION USE. No persistent storage, no auth beyond a static
bearer token, no retry, no observability beyond /info counters. The
runner machine drives load and captures all metrics.

Endpoints
---------
GET    /healthz             always 200 (liveness)
GET    /readyz              200 once both pipes are loaded, else 503
GET    /info                variant, dtype, peak VRAM, counters, job stats
POST   /generate            SYNC text-to-video, single video, base64-encoded
                            response. Subject to upstream proxy timeouts
                            (Cloudflare in front of RunPod = ~120s ceiling).
                            Use this only for short clips (≤ 8s @ 720p).

POST   /jobs                ASYNC text-to-video. Returns 202 + job_id
                            immediately; the model runs in a background
                            worker thread. Survives proxy timeouts since
                            the client polls, never waits.
GET    /jobs/{job_id}       Job status JSON (no bytes).
GET    /jobs/{job_id}/video Raw MP4 bytes (Content-Type: video/mp4).
                            Returns 202 if not done yet, 500 if failed,
                            404 if expired/unknown.
DELETE /jobs/{job_id}       Drop job from registry (frees memory).

Environment
-----------
POD_AUTH_TOKEN              required, shared secret with the runner
LTX2_VARIANT                only "two-stage-distilled" is supported
LTX2_REVISION               optional HF revision/tag (default main)
LTX2_BASE_OFFLOAD           "1" enables model_cpu_offload on the base pipe
                            (saves VRAM on smaller GPUs; on 80 GB H100
                            leave off for speed)
LTX2_JOB_TTL_SECONDS        how long completed jobs are retained in
                            memory (default 600 = 10 min)
LTX2_JOB_REGISTRY_MAX_SIZE  hard cap on number of jobs in memory
                            (default 50; oldest terminal jobs evicted)
HF_HOME                     HF cache dir (default /workspace/hf)
HF_TOKEN                    optional HF token for gated weights
PORT                        default 8000
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock, Thread
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("ltx2-pod")

# Serializes GPU work. Both the sync /generate endpoint and the async
# /jobs worker thread acquire this lock — never both at once, never
# overlapping requests. Single-flight by construction.
GENERATE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POD_AUTH_TOKEN = os.environ.get("POD_AUTH_TOKEN", "")
LTX2_VARIANT = os.environ.get("LTX2_VARIANT", "two-stage-distilled").strip()
LTX2_REVISION = os.environ.get("LTX2_REVISION", "main").strip()
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf")
PORT = int(os.environ.get("PORT", "8000"))
# Whether to call pipe.enable_model_cpu_offload on the BASE pipe. On 80 GB
# H100 it fits without offload (faster); on smaller GPUs offload reduces
# VRAM at the cost of latency. The latent upsampler always uses CPU offload.
LTX2_BASE_OFFLOAD = os.environ.get("LTX2_BASE_OFFLOAD", "0").strip() == "1"

# Async job registry tunables. Completed jobs sit in RAM holding the MP4
# bytes until either (a) TTL expires, (b) the registry hits its size cap
# and they get evicted in age order, or (c) the client explicitly DELETEs.
JOB_TTL_SECONDS = int(os.environ.get("LTX2_JOB_TTL_SECONDS", "600"))
JOB_REGISTRY_MAX_SIZE = int(os.environ.get("LTX2_JOB_REGISTRY_MAX_SIZE", "50"))

# Two-stage production pipeline using the Lightricks 19B base + distilled
# LoRA + 2x latent upsampler. This is the recommended path per the official
# LTX-2 model card on HuggingFace and the diffusers LTX2Pipeline docs.
#
#   stage 1: base 40 steps, CFG 4.0          -> half-res latents
#   stage 1.5: LTX2LatentUpsamplePipeline    -> full-res latents
#   stage 2: base + distilled-LoRA, 3 steps, -> full-res pixels + audio
#            CFG 1.0, sigmas = STAGE_2_DISTILLED_SIGMA_VALUES
#
# Single-stage and FP8 are intentionally NOT supported here. Reasons:
#   - 8-step single-stage on the BASE checkpoint produces garbage; the 8
#     steps only make sense WITH the distilled LoRA from rootonchair OR via
#     two-stage with stage-2 sigmas.
#   - FP8 in diffusers is not natively supported. Lightricks' own
#     ltx-pipelines package has fp8-cast / fp8-scaled-mm modes but that's
#     a different runtime (not diffusers).
VARIANT_CONFIG: dict[str, dict[str, Any]] = {
    "two-stage-distilled": {
        # Fast distilled recipe per official LTX-2 model card. Two-stage
        # flow with distilled LoRA on stage 2. ~140s wall-clock for 10s
        # @ 720p portrait. Quality acceptable for ambient/scene content.
        "pipeline_mode": "two_stage_distilled",
        "model_id": "Lightricks/LTX-2",
        "lora_weight_name": "ltx-2-19b-distilled-lora-384.safetensors",
        "lora_adapter_name": "stage_2_distilled",
        "dtype": "bfloat16",
        "stage1_steps": 40,
        "stage1_cfg": 4.0,
        "stage2_steps": 3,
        "stage2_cfg": 1.0,
    },
    "single-stage-dev": {
        # Quality-mode recipe per official LTX-2 model card. Single-stage
        # direct generation at full target resolution, no upsampler, no
        # LoRA. ~5-7x slower than distilled but visibly higher quality.
        # Use this for premium tier, A/B comparisons, and any case where
        # quality > cost. Frees ~5 GB VRAM at load (no LoRA, no upsampler).
        "pipeline_mode": "single_stage_dev",
        "model_id": "Lightricks/LTX-2",
        "lora_weight_name": None,
        "lora_adapter_name": None,
        "dtype": "bfloat16",
        "steps": 30,
        "cfg": 5.0,
    },
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    pipeline: Any = None  # LTX2Pipeline with LoRA pre-loaded but disabled
    upsample_pipe: Any = None  # LTX2LatentUpsamplePipeline
    stage1_scheduler: Any = None  # default scheduler from the base pipe
    stage2_scheduler: Any = None  # FlowMatchEulerDiscreteScheduler variant for stage 2
    stage_2_sigmas: list[float] = field(default_factory=list)
    audio_sample_rate: int = 24000
    lora_adapter_name: str = ""
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
            "stage1_seconds_total": 0.0,
            "upsample_seconds_total": 0.0,
            "stage2_seconds_total": 0.0,
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
    num_videos_per_prompt: int = Field(default=1, ge=1, le=1)
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


class VideoMeta(BaseModel):
    """Video output metadata without the byte payload.

    Used by the async /jobs API. Fetch the actual MP4 from
    `GET /jobs/{id}/video` which returns raw bytes (no base64, no JSON).
    """

    index: int
    frames: int
    width: int
    height: int
    fps: int
    has_audio: bool


class JobCreateResponse(BaseModel):
    job_id: str
    state: str  # always "queued" at creation
    status_url: str  # relative path to GET /jobs/{id}
    video_url: str  # relative path to GET /jobs/{id}/video
    created_at_ms: int


class JobStatusResponse(BaseModel):
    job_id: str
    state: str  # "queued" | "running" | "done" | "failed"
    created_at_ms: int
    started_at_ms: int | None = None
    completed_at_ms: int | None = None
    metrics: GenerateMetrics | None = None
    video: VideoMeta | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Job registry (in-memory, thread-safe, TTL + size-capped)
# ---------------------------------------------------------------------------


@dataclass
class Job:
    job_id: str
    request: GenerateRequest
    state: str  # "queued" | "running" | "done" | "failed"
    created_at_ms: int
    started_at_ms: int | None = None
    completed_at_ms: int | None = None
    metrics: GenerateMetrics | None = None
    video_meta: VideoMeta | None = None
    video_bytes: bytes | None = None
    error: str | None = None


class JobRegistry:
    """Thread-safe in-memory job store. No disk persistence — jobs die
    with the process. Designed for POC use only.

    Eviction policy:
      - Completed jobs (done/failed) auto-expire after JOB_TTL_SECONDS
      - When adding a new job and the registry is full, the oldest
        terminal job is evicted to make room. Running jobs are never
        evicted (would orphan the worker thread).
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = RLock()

    def create(self, request: GenerateRequest) -> Job:
        with self._lock:
            self._gc_expired()
            self._evict_if_full()
            job_id = uuid.uuid4().hex
            job = Job(
                job_id=job_id,
                request=request,
                state="queued",
                created_at_ms=int(time.time() * 1000),
            )
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._gc_expired()
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)

    def remove(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._gc_expired()
            by_state: dict[str, int] = {}
            total_bytes = 0
            for j in self._jobs.values():
                by_state[j.state] = by_state.get(j.state, 0) + 1
                if j.video_bytes:
                    total_bytes += len(j.video_bytes)
            return {
                "count": len(self._jobs),
                "by_state": by_state,
                "cached_bytes": total_bytes,
                "ttl_seconds": JOB_TTL_SECONDS,
                "max_size": JOB_REGISTRY_MAX_SIZE,
            }

    def _gc_expired(self) -> None:
        cutoff_ms = int(time.time() * 1000) - JOB_TTL_SECONDS * 1000
        expired = [
            jid
            for jid, j in self._jobs.items()
            if j.completed_at_ms and j.completed_at_ms < cutoff_ms
        ]
        for jid in expired:
            self._jobs.pop(jid, None)

    def _evict_if_full(self) -> None:
        if len(self._jobs) < JOB_REGISTRY_MAX_SIZE:
            return
        terminal = sorted(
            (j for j in self._jobs.values() if j.state in ("done", "failed")),
            key=lambda j: j.completed_at_ms or 0,
        )
        for j in terminal:
            self._jobs.pop(j.job_id, None)
            if len(self._jobs) < JOB_REGISTRY_MAX_SIZE:
                return


JOBS = JobRegistry()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="LTX-2 POC server", version="0.2.0")


@app.on_event("startup")
def _load_model_on_startup() -> None:
    """Load the LTX-2 pipeline(s) once at process start.

    What gets loaded depends on cfg["pipeline_mode"]:

      two_stage_distilled (LTX2_VARIANT=two-stage-distilled):
        1. Base LTX2Pipeline (Lightricks/LTX-2, ~19B, BF16)
        2. Distilled stage-2 LoRA weights (~1 GB), loaded but DISABLED
        3. FlowMatchEulerDiscreteScheduler for stage 2 (dynamic shift off)
        4. LTX2LatentUpsamplerModel + LTX2LatentUpsamplePipeline

      single_stage_dev (LTX2_VARIANT=single-stage-dev):
        1. Base LTX2Pipeline only
        Skips LoRA + upsampler entirely (~5 GB VRAM saved at load).

    Loading takes ~100-180s (already cached weights) — we want /readyz to
    be the source of truth for "is the pod hot?", so we pay this cost up
    front instead of on the first /generate.
    """
    if not POD_AUTH_TOKEN:
        log.warning("POD_AUTH_TOKEN not set — server will reject all generate calls")

    cfg = VARIANT_CONFIG.get(LTX2_VARIANT)
    if cfg is None:
        log.error("Unknown LTX2_VARIANT=%r. Valid: %s", LTX2_VARIANT, list(VARIANT_CONFIG))
        sys.exit(2)
    pipeline_mode = cfg["pipeline_mode"]
    if pipeline_mode not in ("two_stage_distilled", "single_stage_dev"):
        log.error("Unknown pipeline_mode=%r for variant=%r", pipeline_mode, LTX2_VARIANT)
        sys.exit(2)

    log.info("Loading LTX-2 pipeline (variant=%s, mode=%s) …", LTX2_VARIANT, pipeline_mode)
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

    torch_dtype = torch.bfloat16

    # ---------- 1. Base pipeline (always loaded) ----------
    log.info("Loading base LTX2Pipeline from %s", cfg["model_id"])
    pipe = LTX2Pipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch_dtype,
        cache_dir=HF_HOME,
        revision=LTX2_REVISION,
    )
    if LTX2_BASE_OFFLOAD:
        log.info("Enabling model CPU offload on base pipe (LTX2_BASE_OFFLOAD=1)")
        pipe.enable_model_cpu_offload(device="cuda")
    else:
        pipe = pipe.to("cuda")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    # ---------- 2. Stage-2 distilled LoRA (only for two-stage-distilled) ----------
    if cfg.get("lora_weight_name"):
        log.info("Loading stage-2 distilled LoRA (%s)", cfg["lora_weight_name"])
        pipe.load_lora_weights(
            cfg["model_id"],
            adapter_name=cfg["lora_adapter_name"],
            weight_name=cfg["lora_weight_name"],
        )
        # Start DISABLED — stage 1 must run on the base weights only.
        pipe.set_adapters([cfg["lora_adapter_name"]], [0.0])
        STATE.lora_adapter_name = cfg["lora_adapter_name"]
    else:
        log.info("Skipping LoRA load (variant has no LoRA)")
        STATE.lora_adapter_name = ""

    # ---------- 3. Schedulers ----------
    # Stage 1 keeps the default scheduler the pipe was built with. For
    # single-stage dev, this is the only scheduler we ever use.
    STATE.stage1_scheduler = pipe.scheduler
    if pipeline_mode == "two_stage_distilled":
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

        # Stage 2 needs a FlowMatchEulerDiscreteScheduler with the docs'
        # specific config (dynamic shifting off, no shift terminal).
        STATE.stage2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_dynamic_shifting=False,
            shift_terminal=None,
        )
        STATE.stage_2_sigmas = list(STAGE_2_DISTILLED_SIGMA_VALUES)
    else:
        STATE.stage2_scheduler = None
        STATE.stage_2_sigmas = []

    # ---------- 4. Latent upsampler (only for two-stage-distilled) ----------
    if pipeline_mode == "two_stage_distilled":
        from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

        log.info("Loading latent upsampler (subfolder=latent_upsampler)")
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            cfg["model_id"],
            subfolder="latent_upsampler",
            torch_dtype=torch_dtype,
            cache_dir=HF_HOME,
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=pipe.vae,
            latent_upsampler=latent_upsampler,
        )
        # Upsampler always uses CPU offload — it's only needed between
        # stages, never concurrent with either; offload saves VRAM during
        # the big stages.
        upsample_pipe.enable_model_cpu_offload(device="cuda")
        STATE.upsample_pipe = upsample_pipe
    else:
        log.info("Skipping latent upsampler load (single-stage variant)")
        STATE.upsample_pipe = None

    # ---------- 5. Audio sample rate ----------
    try:
        STATE.audio_sample_rate = int(pipe.vocoder.config.output_sampling_rate)
    except AttributeError:
        log.warning("pipe.vocoder.config.output_sampling_rate missing — defaulting to 24000")
        STATE.audio_sample_rate = 24000

    STATE.pipeline = pipe
    STATE.variant = LTX2_VARIANT
    STATE.dtype = "bfloat16"
    STATE.load_duration_ms = int((time.time() - t0) * 1000)
    STATE.model_loaded_at_ms = int(time.time() * 1000)
    log.info(
        "Pipeline ready in %.1fs  audio_sr=%d  base_offload=%s",
        STATE.load_duration_ms / 1000.0,
        STATE.audio_sample_rate,
        LTX2_BASE_OFFLOAD,
    )


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


def _round32(x: int) -> int:
    """Round to nearest multiple of 32. LTX-2 hard-rejects non-multiple-of-32 w/h."""
    return max(32, ((x + 16) // 32) * 32)


def _resolution_to_stage_dims(
    resolution: str,
    aspect_ratio: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Map shorthand → ((stage1_w, stage1_h), (stage2_w, stage2_h)).

    LTX-2's two-stage pipeline runs stage 1 at half resolution, then the
    latent upsampler doubles W and H, so stage 2 is at the full target.
    Both stages must be mod-32 — we snap the half-size and use exactly 2×
    for stage 2 to guarantee both are valid.

    "720p" / "1080p" labels the SHORT side (phone portrait convention:
    1080p portrait = 1080 wide × 1920 tall).
    """
    short_side = {"720p": 720, "1080p": 1080}[resolution]
    long_side = short_side * 16 // 9  # 1280 for 720p, 1920 for 1080p
    if aspect_ratio == "9:16":
        full_w, full_h = short_side, long_side
    elif aspect_ratio == "16:9":
        full_w, full_h = long_side, short_side
    else:  # "1:1"
        full_w, full_h = short_side, short_side
    # Stage 1 = half, snapped to mod 32. Stage 2 = exactly 2× stage 1.
    stage1_w = _round32(full_w // 2)
    stage1_h = _round32(full_h // 2)
    stage2_w = stage1_w * 2
    stage2_h = stage1_h * 2
    return (stage1_w, stage1_h), (stage2_w, stage2_h)


def _encode_mp4_bytes(
    frames_np: Any,
    audio_tensor: Any,
    fps: int,
    audio_sample_rate: int,
) -> bytes:
    """Write a numpy frame sequence + optional audio tensor to MP4 bytes.

    Uses diffusers' `encode_video()` helper, which handles audio resampling
    and stream muxing the same way Lightricks does it internally. Falls
    back to silent export if audio is missing.
    """
    import tempfile
    from diffusers.pipelines.ltx2.export_utils import encode_video

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.mp4"
        if audio_tensor is None:
            # No audio → still use export_to_video for the video-only path
            # so we don't ship two muxing paths.
            from diffusers.utils import export_to_video

            export_to_video(frames_np, str(out_path), fps=fps)
        else:
            try:
                import torch as _torch

                if isinstance(audio_tensor, _torch.Tensor):
                    audio_np = audio_tensor.float().cpu()
                else:
                    audio_np = audio_tensor
            except Exception:
                audio_np = audio_tensor
            encode_video(
                frames_np,
                fps=float(fps),
                audio=audio_np,
                audio_sample_rate=audio_sample_rate,
                output_path=str(out_path),
            )
        return out_path.read_bytes()


def _set_stage(stage: int) -> None:
    """Configure the base pipe for stage 1 (LoRA off, default sched) or
    stage 2 (LoRA on, FlowMatchEuler variant). Idempotent across requests.

    For single-stage variants (no LoRA, no alternate scheduler), this is
    a safe no-op. Caller can always invoke it.
    """
    cfg = VARIANT_CONFIG[STATE.variant]
    if cfg["pipeline_mode"] != "two_stage_distilled":
        # Single-stage variants: nothing to toggle.
        return
    pipe = STATE.pipeline
    if stage == 1:
        pipe.scheduler = STATE.stage1_scheduler
        pipe.set_adapters([STATE.lora_adapter_name], [0.0])
    elif stage == 2:
        pipe.scheduler = STATE.stage2_scheduler
        pipe.set_adapters([STATE.lora_adapter_name], [1.0])
    else:
        raise ValueError(f"Unknown stage {stage}")


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
        "jobs": JOBS.stats(),
    }


def _execute_generation(req: GenerateRequest) -> dict[str, Any]:
    """Dispatch to the variant-specific generation path.

    Caller MUST hold GENERATE_LOCK. This function does NOT acquire it —
    that's the responsibility of the HTTP handler (sync /generate) or
    the worker thread (async /jobs).

    Raises plain Exception on failure; the caller converts to HTTPException
    (sync) or stores in the Job error field (async).

    Returns dict with keys:
      mp4_bytes  bytes        the muxed MP4
      frames     int          number of video frames produced
      width      int          final width
      height     int          final height
      fps        int          output framerate (== requested fps)
      has_audio  bool         whether the MP4 has an audio track
      metrics    GenerateMetrics
    """
    cfg = VARIANT_CONFIG[STATE.variant]
    mode = cfg["pipeline_mode"]
    if mode == "two_stage_distilled":
        return _execute_two_stage_distilled(req, cfg)
    if mode == "single_stage_dev":
        return _execute_single_stage_dev(req, cfg)
    raise RuntimeError(f"Unsupported pipeline_mode: {mode}")


def _execute_two_stage_distilled(
    req: GenerateRequest,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Two-stage distilled flow: base stage1 (half res) → upsample → distilled stage2."""

    (stage1_w, stage1_h), (stage2_w, stage2_h) = _resolution_to_stage_dims(
        req.resolution, req.aspect_ratio
    )
    # LTX-2: num_frames must be (8k + 1). Round to NEAREST valid value.
    raw_frames = req.duration_seconds * req.fps
    num_frames = max(9, ((raw_frames - 1 + 4) // 8) * 8 + 1)
    frame_rate = float(req.fps)

    log.info(
        "generate prompt=%r dur=%ds (%d frames) fr=%.1f s1=%dx%d s2=%dx%d audio=%s seed=%d",
        req.prompt[:60] + ("…" if len(req.prompt) > 60 else ""),
        req.duration_seconds,
        num_frames,
        frame_rate,
        stage1_w,
        stage1_h,
        stage2_w,
        stage2_h,
        req.audio_enabled,
        req.seed,
    )

    import torch

    _reset_peak_vram()
    wall_t0 = time.time()
    pipe = STATE.pipeline
    upsample_pipe = STATE.upsample_pipe

    try:
        generator = torch.Generator(device="cuda").manual_seed(req.seed)

        # ----- Stage 1 -----
        _set_stage(1)
        s1_t0 = time.time()
        with torch.no_grad():
            video_latent, audio_latent = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or None,
                width=stage1_w,
                height=stage1_h,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=int(cfg["stage1_steps"]),
                sigmas=None,
                guidance_scale=float(cfg["stage1_cfg"]),
                generator=generator,
                output_type="latent",
                return_dict=False,
            )
        stage1_ms = int((time.time() - s1_t0) * 1000)

        # ----- Stage 1.5: latent upsampling -----
        up_t0 = time.time()
        with torch.no_grad():
            (upscaled_video_latent,) = upsample_pipe(
                latents=video_latent,
                output_type="latent",
                return_dict=False,
            )
        upsample_ms = int((time.time() - up_t0) * 1000)

        # ----- Stage 2 -----
        _set_stage(2)
        s2_t0 = time.time()
        with torch.no_grad():
            video, audio = pipe(
                latents=upscaled_video_latent,
                audio_latents=audio_latent,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or None,
                width=stage2_w,
                height=stage2_h,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=int(cfg["stage2_steps"]),
                sigmas=STATE.stage_2_sigmas,
                noise_scale=STATE.stage_2_sigmas[0],
                guidance_scale=float(cfg["stage2_cfg"]),
                generator=generator,
                output_type="np",
                return_dict=False,
            )
        stage2_ms = int((time.time() - s2_t0) * 1000)
        inference_ms = stage1_ms + upsample_ms + stage2_ms

        video_frames = video[0]
        audio_track = audio[0] if (req.audio_enabled and audio is not None) else None
        frame_count = (
            int(video_frames.shape[0]) if hasattr(video_frames, "shape") else len(video_frames)
        )

        mux_t0 = time.time()
        mp4_bytes = _encode_mp4_bytes(
            frames_np=video_frames,
            audio_tensor=audio_track,
            fps=req.fps,
            audio_sample_rate=STATE.audio_sample_rate,
        )
        mux_ms = int((time.time() - mux_t0) * 1000)

        wall_ms = int((time.time() - wall_t0) * 1000)
        peak_vram = _peak_vram_mb()

        STATE.counters["generations_total"] += 1
        STATE.counters["videos_produced_total"] += 1
        STATE.counters["inference_seconds_total"] += inference_ms / 1000.0
        STATE.counters["stage1_seconds_total"] += stage1_ms / 1000.0
        STATE.counters["upsample_seconds_total"] += upsample_ms / 1000.0
        STATE.counters["stage2_seconds_total"] += stage2_ms / 1000.0
        STATE.counters["peak_vram_mb"] = max(STATE.counters["peak_vram_mb"], peak_vram)

        log.info(
            "generate done wall=%dms s1=%dms up=%dms s2=%dms mux=%dms peak_vram=%dMB bytes=%d",
            wall_ms,
            stage1_ms,
            upsample_ms,
            stage2_ms,
            mux_ms,
            peak_vram,
            len(mp4_bytes),
        )

        metrics = GenerateMetrics(
            wall_clock_ms=wall_ms,
            inference_ms=inference_ms,
            decode_ms=stage2_ms,  # final decode happens inside stage 2
            mux_ms=mux_ms,
            peak_vram_mb=peak_vram,
            num_inference_steps=int(cfg["stage1_steps"]) + int(cfg["stage2_steps"]),
            batch_size=1,
        )

        return {
            "mp4_bytes": mp4_bytes,
            "frames": frame_count,
            "width": stage2_w,
            "height": stage2_h,
            "fps": req.fps,
            "has_audio": audio_track is not None,
            "metrics": metrics,
        }

    finally:
        # Always leave the pipe in stage-1 state for the next caller, even
        # on exception, so the next request starts from a known state.
        try:
            _set_stage(1)
        except Exception:
            pass


def _execute_single_stage_dev(
    req: GenerateRequest,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Single-stage quality mode: direct generation at full target resolution.

    No latent upsampler, no LoRA, no scheduler swap. The full N-step
    denoise runs at the final resolution. ~5-7x slower than the two-stage
    distilled flow but visibly higher quality — this is the documented
    Lightricks "dev" / production-quality recipe.

    Designed for H200/B200-class GPUs. On H100 80 GB this fits at 720p
    portrait but with very little headroom; you'll want LTX2_BASE_OFFLOAD=1
    for any margin.
    """
    # We want the FULL target resolution (no half-res stage 1 anymore).
    # Reuse _resolution_to_stage_dims and take its stage_2 output.
    (_, _), (full_w, full_h) = _resolution_to_stage_dims(
        req.resolution, req.aspect_ratio
    )
    # LTX-2: num_frames must be (8k + 1). Round to NEAREST valid value.
    raw_frames = req.duration_seconds * req.fps
    num_frames = max(9, ((raw_frames - 1 + 4) // 8) * 8 + 1)
    frame_rate = float(req.fps)

    steps = int(cfg["steps"])
    cfg_scale = float(cfg["cfg"])

    log.info(
        "generate(dev) prompt=%r dur=%ds (%d frames) fr=%.1f wxh=%dx%d steps=%d cfg=%.1f audio=%s seed=%d",
        req.prompt[:60] + ("…" if len(req.prompt) > 60 else ""),
        req.duration_seconds,
        num_frames,
        frame_rate,
        full_w,
        full_h,
        steps,
        cfg_scale,
        req.audio_enabled,
        req.seed,
    )

    import torch

    _reset_peak_vram()
    wall_t0 = time.time()
    pipe = STATE.pipeline

    generator = torch.Generator(device="cuda").manual_seed(req.seed)
    inf_t0 = time.time()
    with torch.no_grad():
        video, audio = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            width=full_w,
            height=full_h,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            output_type="np",
            return_dict=False,
        )
    inference_ms = int((time.time() - inf_t0) * 1000)

    video_frames = video[0]
    audio_track = audio[0] if (req.audio_enabled and audio is not None) else None
    frame_count = (
        int(video_frames.shape[0]) if hasattr(video_frames, "shape") else len(video_frames)
    )

    mux_t0 = time.time()
    mp4_bytes = _encode_mp4_bytes(
        frames_np=video_frames,
        audio_tensor=audio_track,
        fps=req.fps,
        audio_sample_rate=STATE.audio_sample_rate,
    )
    mux_ms = int((time.time() - mux_t0) * 1000)

    wall_ms = int((time.time() - wall_t0) * 1000)
    peak_vram = _peak_vram_mb()

    STATE.counters["generations_total"] += 1
    STATE.counters["videos_produced_total"] += 1
    STATE.counters["inference_seconds_total"] += inference_ms / 1000.0
    STATE.counters["peak_vram_mb"] = max(STATE.counters["peak_vram_mb"], peak_vram)
    # stage1/upsample/stage2 counters intentionally not updated — they
    # don't apply to single-stage variants. /info still reports them as
    # cumulative across the pod's lifetime regardless of variant, which
    # is consistent.

    log.info(
        "generate(dev) done wall=%dms infer=%dms mux=%dms peak_vram=%dMB bytes=%d",
        wall_ms,
        inference_ms,
        mux_ms,
        peak_vram,
        len(mp4_bytes),
    )

    metrics = GenerateMetrics(
        wall_clock_ms=wall_ms,
        inference_ms=inference_ms,
        decode_ms=0,  # not separately measurable in single-stage
        mux_ms=mux_ms,
        peak_vram_mb=peak_vram,
        num_inference_steps=steps,
        batch_size=1,
    )

    return {
        "mp4_bytes": mp4_bytes,
        "frames": frame_count,
        "width": full_w,
        "height": full_h,
        "fps": req.fps,
        "has_audio": audio_track is not None,
        "metrics": metrics,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(
    request_body: GenerateRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> GenerateResponse:
    """SYNCHRONOUS two-stage LTX-2 generation.

    The pipeline contract follows the official LTX-2 model card:
      - Stage 1 runs at HALF the requested final resolution, 40 steps,
        CFG 4.0, default scheduler, no LoRA. Output: latents.
      - The latent upsampler doubles W and H spatially.
      - Stage 2 runs at the FULL final resolution, 3 steps, CFG 1.0,
        FlowMatchEulerDiscreteScheduler (no dynamic shifting), distilled
        LoRA enabled, using STAGE_2_DISTILLED_SIGMA_VALUES and
        noise_scale=sigmas[0] to renoise the upscaled latent.

    WARNING: this endpoint blocks the HTTP connection for the full
    generation. Through RunPod's Cloudflare proxy the wire timeout is
    ~120s, so clips >= ~10s WILL hit HTTP 524 even though the pod
    completes successfully. For long clips, use POST /jobs (async).
    """
    _require_auth(authorization)
    if STATE.pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not loaded")
    if request_body.num_videos_per_prompt != 1:
        raise HTTPException(
            status_code=400,
            detail="num_videos_per_prompt > 1 is not supported by the two-stage pipeline",
        )
    if not GENERATE_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="generation already in progress; this pod is single-flight",
        )
    try:
        result = _execute_generation(request_body)
        return GenerateResponse(
            variant=STATE.variant,
            dtype=STATE.dtype,
            videos=[
                VideoOutput(
                    index=0,
                    data_b64=base64.b64encode(result["mp4_bytes"]).decode("ascii"),
                    frames=result["frames"],
                    width=result["width"],
                    height=result["height"],
                    fps=result["fps"],
                    has_audio=result["has_audio"],
                )
            ],
            metrics=result["metrics"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        STATE.counters["generations_failed"] += 1
        log.exception("generate failed: %r", exc)
        raise HTTPException(status_code=500, detail=f"generation_failed: {exc!r}") from exc
    finally:
        GENERATE_LOCK.release()


# ---------------------------------------------------------------------------
# Async jobs API
# ---------------------------------------------------------------------------


def _worker_run_job(job_id: str) -> None:
    """Worker thread body: block on GENERATE_LOCK, run the job, store result.

    Multiple workers (one per submitted job) race for the same lock — the
    OS mutex gives us FIFO-ish scheduling and single-flight by construction.
    Workers never run concurrently with each other or with sync /generate.
    """
    # NOTE: this blocks (no timeout). If the pod is wedged the job sits
    # in "queued" forever; that's acceptable for POC. Production would
    # add a queue timeout that transitions queued → failed.
    GENERATE_LOCK.acquire()
    try:
        job = JOBS.get(job_id)
        if job is None:
            log.warning("worker for %s: job evicted before run", job_id)
            return
        JOBS.update(
            job_id,
            state="running",
            started_at_ms=int(time.time() * 1000),
        )
        try:
            result = _execute_generation(job.request)
            JOBS.update(
                job_id,
                state="done",
                completed_at_ms=int(time.time() * 1000),
                video_bytes=result["mp4_bytes"],
                video_meta=VideoMeta(
                    index=0,
                    frames=result["frames"],
                    width=result["width"],
                    height=result["height"],
                    fps=result["fps"],
                    has_audio=result["has_audio"],
                ),
                metrics=result["metrics"],
            )
            log.info("job %s done (%d bytes)", job_id, len(result["mp4_bytes"]))
        except Exception as exc:
            STATE.counters["generations_failed"] += 1
            log.exception("job %s failed: %r", job_id, exc)
            JOBS.update(
                job_id,
                state="failed",
                completed_at_ms=int(time.time() * 1000),
                error=repr(exc),
            )
    finally:
        GENERATE_LOCK.release()


@app.post("/jobs", response_model=JobCreateResponse, status_code=202)
def jobs_create(
    request_body: GenerateRequest,
    authorization: str | None = Header(default=None),
) -> JobCreateResponse:
    """Submit an async generation job. Returns 202 + job_id immediately.

    The model runs in a background thread; poll GET /jobs/{job_id} for
    status. Fetch the MP4 from GET /jobs/{job_id}/video once state == done.

    This is the path that survives proxy timeouts — the HTTP call here
    returns within milliseconds. Recommended for any clip >= 10s.
    """
    _require_auth(authorization)
    if STATE.pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not loaded")
    if request_body.num_videos_per_prompt != 1:
        raise HTTPException(
            status_code=400,
            detail="num_videos_per_prompt > 1 is not supported by the two-stage pipeline",
        )

    job = JOBS.create(request_body)
    worker = Thread(
        target=_worker_run_job,
        args=(job.job_id,),
        daemon=True,
        name=f"ltx2-job-{job.job_id[:8]}",
    )
    worker.start()
    log.info(
        "job %s queued dur=%ds res=%s ar=%s",
        job.job_id,
        request_body.duration_seconds,
        request_body.resolution,
        request_body.aspect_ratio,
    )
    return JobCreateResponse(
        job_id=job.job_id,
        state=job.state,
        status_url=f"/jobs/{job.job_id}",
        video_url=f"/jobs/{job.job_id}/video",
        created_at_ms=job.created_at_ms,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def jobs_get(
    job_id: str,
    authorization: str | None = Header(default=None),
) -> JobStatusResponse:
    """Poll job status. Returns metadata only — no bytes."""
    _require_auth(authorization)
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found or expired")
    return JobStatusResponse(
        job_id=job.job_id,
        state=job.state,
        created_at_ms=job.created_at_ms,
        started_at_ms=job.started_at_ms,
        completed_at_ms=job.completed_at_ms,
        metrics=job.metrics,
        video=job.video_meta,
        error=job.error,
    )


@app.get("/jobs/{job_id}/video")
def jobs_video(
    job_id: str,
    authorization: str | None = Header(default=None),
) -> Response:
    """Fetch the MP4 bytes for a completed job.

    Returns raw `video/mp4` (no JSON, no base64). This is the endpoint
    designed to stream cleanly through Cloudflare — the bytes are
    already on disk in memory, first byte goes out within milliseconds
    of the request hitting the pod.

    Status:
      200  - job done; body is the MP4
      202  - job queued or running; body is JSON {state, retry_after_seconds}
      404  - job unknown or expired
      500  - job failed; body is JSON {state, error}
    """
    _require_auth(authorization)
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found or expired")
    if job.state == "failed":
        return JSONResponse(
            {"state": "failed", "error": job.error or "unknown"},
            status_code=500,
        )
    if job.state in ("queued", "running"):
        return JSONResponse(
            {"state": job.state, "retry_after_seconds": 5},
            status_code=202,
        )
    # state == "done"
    if job.video_bytes is None:
        raise HTTPException(status_code=500, detail="job done but bytes not available")
    return Response(
        content=job.video_bytes,
        media_type="video/mp4",
        headers={"Content-Length": str(len(job.video_bytes))},
    )


@app.delete("/jobs/{job_id}")
def jobs_delete(
    job_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Drop a job from the registry. Frees its cached MP4 bytes.

    Note: this does NOT cancel a running job — the worker will complete
    its work and then find an evicted slot. Worker logs the eviction.
    """
    _require_auth(authorization)
    ok = JOBS.remove(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found or expired")
    return {"deleted": True, "job_id": job_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
