"""LTX-2 POC eval runner — drives a self-hosted pod through the 40-prompt set.

Counterpart to `poc/pod/server.py`. Run this from your local machine (or
any low-cost VM) while the H100 pod is up. Posts each prompt to the pod's
`/generate` endpoint, decodes the base64 mp4 (one per batch slot), and
records per-call metrics to a CSV next to the clips.

Usage
-----
    export POD_URL=https://<your-pod>.proxy.runpod.net   # no trailing slash
    export POD_AUTH_TOKEN=<shared secret>

    # Full primary config (distilled FP8, batch=1, all 40 prompts)
    python run_pod_eval.py --config distilled_fp8_batch1

    # Subset: skip categories
    python run_pod_eval.py --config distilled_fp8_batch1 --categories C1 C3

    # Force run only specific prompt IDs
    python run_pod_eval.py --config distilled_fp8_batch1 --ids EN-1 CINE-1

    # Dry run — print the plan, no calls
    python run_pod_eval.py --config distilled_fp8_batch2 --dry-run

Outputs
-------
    runs/<config_id>/<prompt_id>__<batch_index>.mp4
    runs/<config_id>/metrics.csv
    runs/<config_id>/info.json          (server /info dump at start)
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import requests
    import yaml
except ImportError:
    sys.exit("Missing deps. Run: pip install -r requirements.txt")

from prompts import PROMPTS, Prompt

_THIS_DIR = Path(__file__).resolve().parent
CONFIGS_FILE = _THIS_DIR / "configs.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalConfig:
    id: str
    variant: str
    batch: int
    resolution: str
    aspect_ratio: str
    note: str
    prompt_categories: list[str] | None
    prompt_ids: list[str] | None


def load_configs(path: Path) -> dict[str, EvalConfig]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    cfgs: dict[str, EvalConfig] = {}
    for entry in raw["configs"]:
        cfg = EvalConfig(
            id=entry["id"],
            variant=entry["variant"],
            batch=int(entry.get("batch", 1)),
            resolution=entry.get("resolution", "1080p"),
            aspect_ratio=entry.get("aspect_ratio", "9:16"),
            note=entry.get("note", ""),
            prompt_categories=entry.get("prompt_categories"),
            prompt_ids=entry.get("prompt_ids"),
        )
        cfgs[cfg.id] = cfg
    return cfgs


def select_prompts(
    cfg: EvalConfig,
    *,
    cli_categories: list[str] | None,
    cli_ids: list[str] | None,
) -> list[Prompt]:
    if cli_ids:
        return [p for p in PROMPTS if p.id in cli_ids]
    if cli_categories:
        return [p for p in PROMPTS if p.category in cli_categories]
    if cfg.prompt_ids:
        return [p for p in PROMPTS if p.id in cfg.prompt_ids]
    if cfg.prompt_categories:
        return [p for p in PROMPTS if p.category in cfg.prompt_categories]
    return list(PROMPTS)


# ---------------------------------------------------------------------------
# Pod interaction
# ---------------------------------------------------------------------------


def pod_get(url: str, path: str, token: str, timeout: float = 10.0) -> dict[str, Any]:
    resp = requests.get(
        f"{url}{path}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def pod_check_ready(url: str, token: str) -> dict[str, Any]:
    """Block until /readyz returns 200 (up to 10 min)."""
    deadline = time.time() + 600
    last_err = ""
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/readyz", timeout=5)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as exc:
            last_err = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"pod did not become ready in 10 min — last: {last_err}")


def pod_generate(
    url: str,
    token: str,
    *,
    prompt: Prompt,
    cfg: EvalConfig,
    seed: int,
    timeout_s: int,
) -> dict[str, Any]:
    body = {
        "prompt": prompt.text,
        "duration_seconds": prompt.duration,
        "resolution": cfg.resolution,
        "aspect_ratio": cfg.aspect_ratio,
        "fps": 24,
        "seed": seed,
        "num_videos_per_prompt": cfg.batch,
        "audio_enabled": True,
    }
    resp = requests.post(
        f"{url}/generate",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout_s,
    )
    if resp.status_code != 200:
        return {
            "_error": f"HTTP {resp.status_code}: {resp.text[:400]}",
        }
    return resp.json()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

METRICS_COLUMNS = [
    "config_id",
    "variant",
    "dtype",
    "prompt_id",
    "category",
    "lang",
    "duration_s",
    "batch_size",
    "batch_index",
    "seed",
    "wall_clock_ms",
    "inference_ms",
    "decode_ms",
    "mux_ms",
    "peak_vram_mb",
    "num_inference_steps",
    "file_kb",
    "has_audio",
    "success",
    "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LTX-2 POC pod eval runner")
    p.add_argument("--config", required=True, help="Config id from configs.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--categories", nargs="*", help="Override: only these categories")
    p.add_argument("--ids", nargs="*", help="Override: only these prompt IDs")
    p.add_argument("--timeout-s", type=int, default=600, help="Per-request timeout")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--runs-dir", default="runs", help="Output root directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    pod_url = os.environ.get("POD_URL", "").rstrip("/")
    pod_token = os.environ.get("POD_AUTH_TOKEN", "")
    if not args.dry_run:
        if not pod_url:
            print("ERROR: POD_URL env var not set", file=sys.stderr)
            return 2
        if not pod_token:
            print("ERROR: POD_AUTH_TOKEN env var not set", file=sys.stderr)
            return 2

    configs = load_configs(CONFIGS_FILE)
    if args.config not in configs:
        print(f"ERROR: unknown config {args.config!r}. Available: {list(configs)}", file=sys.stderr)
        return 2
    cfg = configs[args.config]
    selected = select_prompts(cfg, cli_categories=args.categories, cli_ids=args.ids)

    out_dir = Path(args.runs_dir) / cfg.id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:    {cfg.id}  ({cfg.note})")
    print(
        f"Variant:   {cfg.variant}  batch={cfg.batch}  res={cfg.resolution}  ar={cfg.aspect_ratio}"
    )
    print(f"Prompts:   {len(selected)} selected  seed={args.seed}")
    print(f"Output:    {out_dir.resolve()}")
    print()

    if args.dry_run:
        for p in selected:
            print(f"  - {p.id:<8} {p.category} {p.lang} {p.duration}s  → {cfg.batch} video(s)")
        return 0

    print("Probing /readyz …")
    ready = pod_check_ready(pod_url, pod_token)
    print(f"  ready: variant={ready.get('variant')}")
    if ready.get("variant") != cfg.variant:
        print(
            f"  WARNING: pod is running variant={ready.get('variant')!r} but config asks for "
            f"{cfg.variant!r}. Restart the pod with LTX2_VARIANT={cfg.variant} before running.",
            file=sys.stderr,
        )

    info = pod_get(pod_url, "/info", pod_token)
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(f"  GPU: {info.get('gpu_name')}  {info.get('gpu_total_vram_mb')} MB total VRAM")
    print(f"  Load duration: {info.get('load_duration_ms')} ms")
    print()

    metrics_path = out_dir / "metrics.csv"
    new_file = not metrics_path.exists()
    with metrics_path.open("a", newline="") as mf:
        w = csv.writer(mf)
        if new_file:
            w.writerow(METRICS_COLUMNS)
            mf.flush()

        for idx, prompt in enumerate(selected, start=1):
            existing = list(out_dir.glob(f"{prompt.id}__*.mp4"))
            if len(existing) >= cfg.batch:
                print(f"·  [{idx:>2}/{len(selected)}] skip {prompt.id} (already on disk)")
                continue

            t0 = time.time()
            result = pod_generate(
                url=pod_url,
                token=pod_token,
                prompt=prompt,
                cfg=cfg,
                seed=args.seed,
                timeout_s=args.timeout_s,
            )
            wall = int((time.time() - t0) * 1000)

            if "_error" in result:
                err = str(result["_error"])
                print(f"✗  [{idx:>2}/{len(selected)}] {prompt.id} failed in {wall}ms: {err[:160]}")
                w.writerow(
                    [
                        cfg.id,
                        cfg.variant,
                        "",
                        prompt.id,
                        prompt.category,
                        prompt.lang,
                        prompt.duration,
                        cfg.batch,
                        0,
                        args.seed,
                        wall,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        "false",
                        "false",
                        err,
                    ]
                )
                mf.flush()
                continue

            variant = result.get("variant", "")
            dtype = result.get("dtype", "")
            m = result.get("metrics", {})
            videos = result.get("videos", [])

            for vid in videos:
                index = int(vid["index"])
                out_path = out_dir / f"{prompt.id}__{index}.mp4"
                data = base64.b64decode(vid["data_b64"])
                out_path.write_bytes(data)
                w.writerow(
                    [
                        cfg.id,
                        variant,
                        dtype,
                        prompt.id,
                        prompt.category,
                        prompt.lang,
                        prompt.duration,
                        cfg.batch,
                        index,
                        args.seed,
                        wall,
                        int(m.get("inference_ms", 0)),
                        int(m.get("decode_ms", 0)),
                        int(m.get("mux_ms", 0)),
                        int(m.get("peak_vram_mb", 0)),
                        int(m.get("num_inference_steps", 0)),
                        round(len(data) / 1024.0, 1),
                        "true" if vid.get("has_audio") else "false",
                        "true",
                        "",
                    ]
                )
            mf.flush()

            inf_ms = int(m.get("inference_ms", 0))
            peak = int(m.get("peak_vram_mb", 0))
            print(
                f"✓  [{idx:>2}/{len(selected)}] {prompt.id:<8} "
                f"{wall}ms wall  {inf_ms}ms infer  peak={peak}MB  "
                f"× {len(videos)} clips"
            )

    final_info = pod_get(pod_url, "/info", pod_token)
    (out_dir / "info_final.json").write_text(json.dumps(final_info, indent=2))
    print()
    print(f"Done. Clips + metrics: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
