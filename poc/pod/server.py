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
bearer token, no queue, no retry. The runner machine drives load and
captures all metrics.

Endpoints
---------
GET  /healthz   always 200 (liveness)
GET  /readyz    200 once both pipes are loaded, else 503
GET  /info      variant, dtype, peak VRAM, throughput + per-stage counters
POST /generate  text-to-video (two-stage), single video per request

Environment
-----------
POD_AUTH_TOKEN     required, shared secret with the runner
LTX2_VARIANT       only "two-stage-distilled" is supported in this build
LTX2_REVISION      optional HF revision/tag (default main)
LTX2_BASE_OFFLOAD  "1" enables model_cpu_offload on the base pipe (saves
                   VRAM on smaller GPUs; on 80 GB H100 leave off for speed)
HF_HOME            HF cache dir (default /workspace/hf)
HF_TOKEN           optional HF token for gated weights
PORT               default 8000
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import threading
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
        "model_id": "Lightricks/LTX-2",
        "lora_weight_name": "ltx-2-19b-distilled-lora-384.safetensors",
        "lora_adapter_name": "stage_2_distilled",
        "dtype": "bfloat16",
        "stage1_steps": 40,
        "stage1_cfg": 4.0,
        "stage2_steps": 3,
        "stage2_cfg": 1.0,
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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="LTX-2 POC server", version="0.1.0")


@app.on_event("startup")
def _load_model_on_startup() -> None:
    """Load the LTX-2 two-stage pipeline once at process start.

    What we load:
      1. Base LTX2Pipeline (Lightricks/LTX-2, ~19B, BF16) — used for both
         stage 1 (default scheduler, LoRA off) and stage 2 (alt scheduler,
         distilled LoRA on).
      2. The stage-2 distilled LoRA weights, loaded into the base pipe but
         left DISABLED until stage 2.
      3. A second `FlowMatchEulerDiscreteScheduler` configured for stage 2
         (use_dynamic_shifting=False, shift_terminal=None).
      4. LTX2LatentUpsamplerModel + LTX2LatentUpsamplePipeline for 2x
         spatial upsampling between stages.

    Loading takes ~2 min (already cached) — we want /readyz to be the
    source of truth for "is the pod hot?", so we pay this cost up front
    instead of on the first /generate.
    """
    if not POD_AUTH_TOKEN:
        log.warning("POD_AUTH_TOKEN not set — server will reject all generate calls")

    cfg = VARIANT_CONFIG.get(LTX2_VARIANT)
    if cfg is None:
        log.error("Unknown LTX2_VARIANT=%r. Valid: %s", LTX2_VARIANT, list(VARIANT_CONFIG))
        sys.exit(2)

    log.info("Loading LTX-2 two-stage pipeline (variant=%s) …", LTX2_VARIANT)
    t0 = time.time()

    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, LTX2Pipeline
    from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

    STATE.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda"
    STATE.gpu_total_vram_mb = (
        int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        if torch.cuda.is_available()
        else 0
    )
    log.info("GPU: %s, total VRAM: %d MB", STATE.gpu_name, STATE.gpu_total_vram_mb)

    torch_dtype = torch.bfloat16

    # ---------- 1. Base pipeline ----------
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

    # ---------- 2. Distilled stage-2 LoRA ----------
    log.info("Loading stage-2 distilled LoRA (%s)", cfg["lora_weight_name"])
    pipe.load_lora_weights(
        cfg["model_id"],
        adapter_name=cfg["lora_adapter_name"],
        weight_name=cfg["lora_weight_name"],
    )
    # Start DISABLED — stage 1 must run on the base weights only.
    pipe.set_adapters([cfg["lora_adapter_name"]], [0.0])

    # ---------- 3. Schedulers ----------
    # Stage 1 keeps the default scheduler the pipe was built with.
    STATE.stage1_scheduler = pipe.scheduler
    # Stage 2 needs a FlowMatchEulerDiscreteScheduler with the docs' specific
    # config (dynamic shifting off, no shift terminal). Build it now so we
    # don't pay reconfiguration cost on every request.
    STATE.stage2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_dynamic_shifting=False,
        shift_terminal=None,
    )
    STATE.stage_2_sigmas = list(STAGE_2_DISTILLED_SIGMA_VALUES)

    # ---------- 4. Latent upsampler ----------
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
    # Upsampler always uses CPU offload — it's only needed between stage 1
    # and stage 2, never concurrent with either; offload saves VRAM during
    # the big stages.
    upsample_pipe.enable_model_cpu_offload(device="cuda")

    # ---------- 5. Audio sample rate ----------
    try:
        STATE.audio_sample_rate = int(pipe.vocoder.config.output_sampling_rate)
    except AttributeError:
        log.warning("pipe.vocoder.config.output_sampling_rate missing — defaulting to 24000")
        STATE.audio_sample_rate = 24000

    STATE.pipeline = pipe
    STATE.upsample_pipe = upsample_pipe
    STATE.lora_adapter_name = cfg["lora_adapter_name"]
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
    """
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
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(
    request_body: GenerateRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> GenerateResponse:
    """Two-stage LTX-2 generation: base → upsample → distilled-LoRA stage 2.

    The pipeline contract follows the official LTX-2 model card:
      - Stage 1 runs at HALF the requested final resolution, 40 steps,
        CFG 4.0, default scheduler, no LoRA. Output: latents.
      - The latent upsampler doubles W and H spatially.
      - Stage 2 runs at the FULL final resolution, 3 steps, CFG 1.0,
        FlowMatchEulerDiscreteScheduler (no dynamic shifting), distilled
        LoRA enabled, using STAGE_2_DISTILLED_SIGMA_VALUES and
        noise_scale=sigmas[0] to renoise the upscaled latent.
    """
    _require_auth(authorization)
    if STATE.pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not loaded")
    if not GENERATE_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="generation already in progress; this pod currently supports single-flight only",
        )

    cfg = VARIANT_CONFIG[STATE.variant]
    if request_body.num_videos_per_prompt != 1:
        # Diffusers LTX2Pipeline supports num_videos_per_prompt > 1 in
        # principle, but the two-stage flow is single-batch in every
        # documented example. Reject for now to keep the pipeline path
        # deterministic; we can revisit batching later for the eval run.
        raise HTTPException(
            status_code=400,
            detail="num_videos_per_prompt > 1 is not supported by the two-stage pipeline",
        )

    (stage1_w, stage1_h), (stage2_w, stage2_h) = _resolution_to_stage_dims(
        request_body.resolution, request_body.aspect_ratio
    )
    # LTX-2: num_frames must be (8k + 1). Round to NEAREST valid value.
    raw_frames = request_body.duration_seconds * request_body.fps
    num_frames = max(9, ((raw_frames - 1 + 4) // 8) * 8 + 1)
    frame_rate = float(request_body.fps)

    log.info(
        "generate prompt=%r dur=%ds (%d frames) fr=%.1f s1=%dx%d s2=%dx%d audio=%s seed=%d",
        request_body.prompt[:60] + ("…" if len(request_body.prompt) > 60 else ""),
        request_body.duration_seconds,
        num_frames,
        frame_rate,
        stage1_w,
        stage1_h,
        stage2_w,
        stage2_h,
        request_body.audio_enabled,
        request_body.seed,
    )

    import torch

    _reset_peak_vram()
    wall_t0 = time.time()
    pipe = STATE.pipeline
    upsample_pipe = STATE.upsample_pipe

    try:
        generator = torch.Generator(device="cuda").manual_seed(request_body.seed)

        # ----- Stage 1 -----
        _set_stage(1)
        s1_t0 = time.time()
        with torch.no_grad():
            video_latent, audio_latent = pipe(
                prompt=request_body.prompt,
                negative_prompt=request_body.negative_prompt or None,
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
                prompt=request_body.prompt,
                negative_prompt=request_body.negative_prompt or None,
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

        # video is a list of numpy arrays (frames, H, W, 3); audio is a list
        # of torch tensors. With num_videos_per_prompt=1, take index 0.
        video_frames = video[0]
        audio_track = audio[0] if (request_body.audio_enabled and audio is not None) else None
        frame_count = int(video_frames.shape[0]) if hasattr(video_frames, "shape") else len(video_frames)

        mux_t0 = time.time()
        mp4_bytes = _encode_mp4_bytes(
            frames_np=video_frames,
            audio_tensor=audio_track,
            fps=request_body.fps,
            audio_sample_rate=STATE.audio_sample_rate,
        )
        mux_ms = int((time.time() - mux_t0) * 1000)

        wall_ms = int((time.time() - wall_t0) * 1000)
        peak_vram = _peak_vram_mb()
        outputs = [
            VideoOutput(
                index=0,
                data_b64=base64.b64encode(mp4_bytes).decode("ascii"),
                frames=frame_count,
                width=stage2_w,
                height=stage2_h,
                fps=request_body.fps,
                has_audio=audio_track is not None,
            )
        ]

        STATE.counters["generations_total"] += 1
        STATE.counters["videos_produced_total"] += 1
        STATE.counters["inference_seconds_total"] += inference_ms / 1000.0
        STATE.counters["stage1_seconds_total"] += stage1_ms / 1000.0
        STATE.counters["upsample_seconds_total"] += upsample_ms / 1000.0
        STATE.counters["stage2_seconds_total"] += stage2_ms / 1000.0
        STATE.counters["peak_vram_mb"] = max(STATE.counters["peak_vram_mb"], peak_vram)

        log.info(
            "generate done wall=%dms s1=%dms up=%dms s2=%dms mux=%dms peak_vram=%dMB",
            wall_ms,
            stage1_ms,
            upsample_ms,
            stage2_ms,
            mux_ms,
            peak_vram,
        )

        return GenerateResponse(
            variant=STATE.variant,
            dtype=STATE.dtype,
            videos=outputs,
            metrics=GenerateMetrics(
                wall_clock_ms=wall_ms,
                inference_ms=inference_ms,
                decode_ms=stage2_ms,  # final decode happens inside stage 2
                mux_ms=mux_ms,
                peak_vram_mb=peak_vram,
                num_inference_steps=int(cfg["stage1_steps"]) + int(cfg["stage2_steps"]),
                batch_size=1,
            ),
        )

    except Exception as exc:
        STATE.counters["generations_failed"] += 1
        log.exception("generate failed: %r", exc)
        # Best-effort: restore stage 1 settings so the next request starts clean.
        try:
            _set_stage(1)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"generation_failed: {exc!r}") from exc
    finally:
        # Always leave the pipe in stage-1 state for the next caller.
        try:
            _set_stage(1)
        except Exception:
            pass
        GENERATE_LOCK.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
