"""Faster H.264/AAC MP4 mux for the LTX-2 POC server.

Derived from HuggingFace diffusers
``diffusers.pipelines.ltx2.export_utils.encode_video`` (Apache-2.0),
with explicit **libx264 preset / CRF / threads** and optional **h264_nvenc**
so mux is not stuck on x264's implicit ``medium`` preset (slow on CPU).

Environment (all optional)
---------------------------
LTX2_MUX_ENCODER
    ``libx264`` (default) or ``h264_nvenc``. NVENC is tried first when set
    to ``h264_nvenc``; on failure we log and fall back to libx264.

LTX2_MUX_X264_PRESET
    x264 preset name. Default: ``veryfast``. Use ``ultrafast`` for maximum
    speed at POC quality; ``faster`` / ``fast`` if you want a bit more
    compression efficiency.

LTX2_MUX_X264_CRF
    Constant rate factor for libx264. Default: ``23``.

LTX2_MUX_X264_TUNE
    Optional x264 ``tune`` value, e.g. ``film`` or ``zerolatency``. Empty
    disables (default).

LTX2_MUX_THREADS
    x264 ``threads`` option as string. ``0`` = auto (default).

LTX2_MUX_NVENC_PRESET
    NVENC preset (``p1`` … ``p7``). Default: ``p4``.

LTX2_MUX_NVENC_CQ
    NVENC constant-quality target when using VBR+CQ. Default: ``28``.

LTX2_MUX_PROGRESS
    Set to ``1`` to show tqdm progress during chunk encode (matches upstream
    UX). Default ``0`` to keep pod logs quiet.

LTX2_MUX_VIDEO_CHUNKS
    Split frames into N chunks along time before encoding (same idea as
    upstream ``video_chunks_number``). Default ``1`` (fastest for our sizes).
"""

from __future__ import annotations

import logging
import os
from fractions import Fraction
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("ltx2-mux")

_mux_config_logged = False


def _read_mux_env() -> dict[str, Any]:
    return {
        "encoder": os.environ.get("LTX2_MUX_ENCODER", "libx264").strip().lower(),
        "x264_preset": os.environ.get("LTX2_MUX_X264_PRESET", "veryfast").strip(),
        "x264_crf": os.environ.get("LTX2_MUX_X264_CRF", "23").strip(),
        "x264_tune": os.environ.get("LTX2_MUX_X264_TUNE", "").strip(),
        "x264_threads": os.environ.get("LTX2_MUX_THREADS", "0").strip(),
        "nvenc_preset": os.environ.get("LTX2_MUX_NVENC_PRESET", "p4").strip(),
        "nvenc_cq": os.environ.get("LTX2_MUX_NVENC_CQ", "28").strip(),
        "show_progress": os.environ.get("LTX2_MUX_PROGRESS", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        ),
        "video_chunks": max(1, int(os.environ.get("LTX2_MUX_VIDEO_CHUNKS", "1"))),
    }


def _log_mux_once(cfg: dict[str, Any], effective_encoder: str) -> None:
    global _mux_config_logged
    if _mux_config_logged:
        return
    _mux_config_logged = True
    log.info(
        "mux: encoder=%s (requested=%s) x264_preset=%s crf=%s threads=%s "
        "nvenc_preset=%s chunks=%d progress=%s",
        effective_encoder,
        cfg["encoder"],
        cfg["x264_preset"],
        cfg["x264_crf"],
        cfg["x264_threads"],
        cfg["nvenc_preset"],
        cfg["video_chunks"],
        cfg["show_progress"],
    )


def _frames_to_uint8_hwc(video: Any) -> np.ndarray:
    """Return uint8 ``[T, H, W, C]`` RGB."""
    import torch
    from PIL import Image

    if isinstance(video, list) and video and isinstance(video[0], Image.Image):
        arrs = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in video]
        return np.stack(arrs, axis=0)

    if isinstance(video, np.ndarray):
        v = video
        if v.dtype == np.uint8:
            return v
        in_01 = np.logical_and(v >= 0.0, v <= 1.0)
        if v.size and np.all(in_01):
            return (np.clip(v, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return np.clip(v, 0, 255).astype(np.uint8)

    if isinstance(video, torch.Tensor):
        t = video.detach().cpu()
        if t.dtype.is_floating_point and float(t.max()) <= 1.0 + 1e-3:
            t = (t * 255.0).clamp(0, 255)
        return t.to(torch.uint8).numpy()

    raise TypeError(f"Unsupported video type for mux: {type(video)!r}")


def _prepare_audio_stream(container: Any, audio_sample_rate: int) -> Any:
    import av

    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(container: Any, audio_stream: Any, frame_in: Any) -> None:
    import av

    cc = audio_stream.codec_context
    target_format = cc.format or "fltp"
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate
    resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )
    audio_next_pts = 0
    for rframe in resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))
    for packet in audio_stream.encode():
        container.mux(packet)


def _write_audio(
    container: Any,
    audio_stream: Any,
    samples: Any,
    audio_sample_rate: int,
) -> None:
    import av
    import torch

    if not isinstance(samples, torch.Tensor):
        samples = torch.as_tensor(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T
    if samples.shape[1] != 2:
        raise ValueError(f"Expected stereo audio; got shape {tuple(samples.shape)}.")

    if samples.dtype != torch.int16:
        samples = torch.clip(samples.float(), -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate
    _resample_audio(container, audio_stream, frame_in)


def _open_video_stream(container: Any, width: int, height: int, fps: int, encoder: str, cfg: dict[str, Any]):
    import av

    if encoder == "h264_nvenc":
        stream = container.add_stream("h264_nvenc", rate=int(fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "preset": cfg["nvenc_preset"],
            "rc": "vbr",
            "cq": cfg["nvenc_cq"],
        }
        return stream

    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    opts: dict[str, str] = {
        "preset": cfg["x264_preset"],
        "crf": cfg["x264_crf"],
        "threads": cfg["x264_threads"],
    }
    if cfg["x264_tune"]:
        opts["tune"] = cfg["x264_tune"]
    stream.options = opts
    return stream


def _encode_once(
    video_u8: np.ndarray,
    *,
    fps: int,
    audio: Any | None,
    audio_sample_rate: int | None,
    output_path: Path,
    encoder: str,
    cfg: dict[str, Any],
) -> None:
    import av
    import torch

    n_chunks = cfg["video_chunks"]
    raw_chunks = [c for c in np.array_split(video_u8, n_chunks, axis=0) if c.shape[0] > 0]
    if not raw_chunks:
        raise ValueError("mux: empty video after chunking")

    first = raw_chunks[0]
    _, height, width, _ = first.shape

    container = av.open(str(output_path), mode="w")
    try:
        vstream = _open_video_stream(container, width, height, fps, encoder, cfg)
        audio_stream = None
        if audio is not None:
            if audio_sample_rate is None:
                raise ValueError("audio_sample_rate is required when audio is set")
            audio_stream = _prepare_audio_stream(container, audio_sample_rate)

        seq = chain([first], raw_chunks[1:])
        if cfg["show_progress"]:
            from tqdm import tqdm

            seq = tqdm(seq, total=len(raw_chunks), desc="Encoding video chunks")

        for video_chunk in seq:
            for frame_array in video_chunk:
                frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in vstream.encode(frame):
                    container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)

        if audio_stream is not None:
            _write_audio(container, audio_stream, audio, int(audio_sample_rate))

    finally:
        container.close()


def encode_mp4_file(
    video: Any,
    *,
    fps: float,
    audio: Any | None,
    audio_sample_rate: int | None,
    output_path: str | Path,
) -> None:
    """Encode RGB frames + optional stereo float waveform to MP4 on disk."""
    cfg = _read_mux_env()
    out = Path(output_path)
    video_u8 = _frames_to_uint8_hwc(video)
    fps_i = max(1, int(round(fps)))

    want = cfg["encoder"]
    if want == "h264_nvenc":
        try:
            _encode_once(
                video_u8,
                fps=fps_i,
                audio=audio,
                audio_sample_rate=audio_sample_rate,
                output_path=out,
                encoder="h264_nvenc",
                cfg=cfg,
            )
            _log_mux_once(cfg, "h264_nvenc")
            return
        except Exception as exc:
            log.warning("h264_nvenc mux failed (%s); falling back to libx264", exc)

    _encode_once(
        video_u8,
        fps=fps_i,
        audio=audio,
        audio_sample_rate=audio_sample_rate,
        output_path=out,
        encoder="libx264",
        cfg=cfg,
    )
    _log_mux_once(cfg, "libx264")
