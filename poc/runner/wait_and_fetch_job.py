#!/usr/bin/env python3
"""Poll a pod /jobs/{id} until done, then download the MP4.

Why this exists
---------------
LTX-2 logs show diffusers tqdm (e.g. 30/30) *before* video mux (ffmpeg/PyAV).
The server's job state stays ``running`` until **both** inference and mux
finish, then flips to ``done``. If you only watch tqdm, it looks "done"
while GET /jobs still says running — that is expected.

You can also probe ``GET /jobs/{id}/video``: HTTP 202 + JSON means still
running; HTTP 200 + ``video/mp4`` means the file is ready.

Usage::

  export POD_AUTH_TOKEN=...
  python3 poc/runner/wait_and_fetch_job.py \\
    --pod https://xxxxx-8000.proxy.runpod.net \\
    --job 4cb5849828874956a88cc3da705d0145 \\
    --out poc/outputs/paris.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pod", required=True, help="Pod base URL (no trailing slash)")
    p.add_argument("--job", required=True, help="Job UUID from POST /jobs")
    p.add_argument("--out", required=True, type=Path, help="Output .mp4 path")
    p.add_argument(
        "--max-wait-min",
        type=float,
        default=60.0,
        help="Give up after this many minutes (default: 60)",
    )
    args = p.parse_args()

    token = os.environ.get("POD_AUTH_TOKEN", "")
    if not token:
        print("POD_AUTH_TOKEN is not set", file=sys.stderr)
        sys.exit(2)

    pod = args.pod.rstrip("/")
    hdr = {"Authorization": f"Bearer {token}"}
    jid = args.job
    deadline = time.time() + args.max_wait_min * 60.0

    while time.time() < deadline:
        # Prefer /video: 200 means bytes are ready even if we skip /jobs.
        vr = requests.get(f"{pod}/jobs/{jid}/video", headers=hdr, timeout=120)
        if vr.status_code == 200:
            is_mp4 = len(vr.content) >= 12 and vr.content[4:8] == b"ftyp"
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(vr.content)
            tag = "mp4" if is_mp4 else "raw"
            print(f"saved {args.out.resolve()} ({len(vr.content)} bytes, {tag})")
            return
        if vr.status_code == 202:
            try:
                hint = vr.json().get("retry_after_seconds", 5)
            except Exception:
                hint = 5
        elif vr.status_code == 404:
            print("job not found or expired (404)", file=sys.stderr)
            sys.exit(1)
        elif vr.status_code == 500:
            print("job failed:", vr.text[:500], file=sys.stderr)
            sys.exit(1)
        else:
            hint = 5

        st = requests.get(f"{pod}/jobs/{jid}", headers=hdr, timeout=60)
        if st.status_code != 200:
            print(st.status_code, st.text[:200], file=sys.stderr)
            time.sleep(hint)
            continue
        body = st.json()
        state = body.get("state")
        ts = time.strftime("%H:%M:%S")
        print(f"{ts}  state={state}  (video HTTP {vr.status_code})")
        if state == "failed":
            print("error:", body.get("error"), file=sys.stderr)
            sys.exit(1)
        time.sleep(min(float(hint), 30.0))

    print("timeout — job still not finished", file=sys.stderr)
    sys.exit(3)


if __name__ == "__main__":
    main()
