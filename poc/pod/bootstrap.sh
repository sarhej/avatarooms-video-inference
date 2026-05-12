#!/usr/bin/env bash
# LTX-2 POC pod bootstrap.
#
# Run on a freshly-rented RunPod H100 pod. Expects the "PyTorch 2.7 / CUDA 12.7"
# template (or any image with torch>=2.7, cuda>=12.7, python>=3.10 already
# installed). Idempotent — safe to re-run.
#
#   curl -sSL https://raw.githubusercontent.com/sarhej/avatarooms-video-inference/main/poc/pod/bootstrap.sh | bash
#
# Required env vars:
#   POD_AUTH_TOKEN   shared secret with the runner machine
#   HF_TOKEN         HuggingFace token with read access to Lightricks/LTX-2
#
# Optional env vars:
#   LTX2_VARIANT     distilled-fp8 (default) | distilled-bf16 | dev
#   PORT             default 8000

set -euo pipefail

WORKDIR="/workspace/poc"
REPO_URL="${REPO_URL:-https://github.com/sarhej/avatarooms-video-inference.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
HF_HOME="${HF_HOME:-/workspace/hf}"
LTX2_VARIANT="${LTX2_VARIANT:-distilled-fp8}"
PORT="${PORT:-8000}"

log() { printf '\n[bootstrap] %s\n' "$*"; }

show_disk() {
  echo "    df -h:"
  df -h / /workspace 2>/dev/null | sed 's/^/      /' || true
}

# ---------------------------------------------------------------------------
# CRITICAL: cleanup partial state BEFORE any mkdir, because /workspace may
# be 100% full from a previous failed HF download. mkdir would fail with
# "No space left on device" before we get a chance to free space.
# ---------------------------------------------------------------------------

echo "[bootstrap] Pre-cleanup disk state"
show_disk

echo "[bootstrap] Inventory of /workspace BEFORE cleanup:"
ls -la /workspace/ 2>/dev/null | head -50 || true
echo "    sizes (top 20):"
du -sh /workspace/* /workspace/.[!.]* 2>/dev/null | sort -h | tail -20 || true

# AGGRESSIVE cleanup. Targeted cleanup in v2 didn't work because the 100 GB
# wasn't under /workspace/hf — it was somewhere else we didn't enumerate.
# Wipe EVERYTHING under /workspace except lost+found, unless we have a
# sentinel proving a previous bootstrap completed successfully.
if [[ -f /workspace/.bootstrap-complete ]]; then
  echo "[bootstrap] /workspace/.bootstrap-complete sentinel found — preserving cache"
else
  echo "[bootstrap] No sentinel — AGGRESSIVE cleanup: wiping ALL of /workspace contents (except lost+found)"
  find /workspace -mindepth 1 -maxdepth 1 ! -name 'lost+found' -exec rm -rf {} + 2>/dev/null || true
fi

echo "[bootstrap] Post-cleanup disk state"
show_disk
echo "    /workspace contents after cleanup:"
ls -la /workspace/ 2>/dev/null | head -10 || true

# Now that we've freed space, route temp/cache dirs onto the volume disk
# (/workspace, 100+ GB) instead of the container root (/tmp on the 40 GB
# container disk). HF's xet downloader writes large temp files via
# _download_to_tmp_and_move during chunked downloads.
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pipcache}"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------

log "Sanity check: GPU + Python versions"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "ERROR: nvidia-smi failed — no GPU?"; exit 1;
}
python3 --version
python3 -c 'import torch, sys; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

if [[ -z "${POD_AUTH_TOKEN:-}" ]]; then
  echo "ERROR: POD_AUTH_TOKEN must be set"; exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN must be set (needed to download Lightricks/LTX-2)"; exit 2
fi

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------

log "Installing system packages (ffmpeg, git, build tools)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  ffmpeg git curl ca-certificates \
  libsndfile1 libsndfile1-dev \
  >/dev/null
log "Disk state after apt"
show_disk

# ---------------------------------------------------------------------------
# 2. Pull the POC code
# ---------------------------------------------------------------------------

log "Checking out repo: ${REPO_URL} (${REPO_BRANCH})"
mkdir -p "${WORKDIR}"
if [[ -d "${WORKDIR}/.git" ]]; then
  cd "${WORKDIR}"
  git fetch --depth=1 origin "${REPO_BRANCH}"
  git checkout -B "${REPO_BRANCH}" "origin/${REPO_BRANCH}"
else
  git clone --depth=1 --branch "${REPO_BRANCH}" "${REPO_URL}" "${WORKDIR}"
  cd "${WORKDIR}"
fi
log "Disk state after repo checkout"
show_disk

# ---------------------------------------------------------------------------
# 3. Python packages
# ---------------------------------------------------------------------------

log "Installing Python deps from poc/pod/requirements.txt"
python3 -m pip install --upgrade --quiet pip
# Force-reinstall diffusers from git main. The default pip behavior is to
# treat any installed `diffusers` package as satisfying the `diffusers @ git+...`
# constraint, so an old PyPI 0.36.0 wheel stays in place. --force-reinstall
# guarantees we end up with the actual git-main commit.
log "Force-reinstalling diffusers from git main"
python3 -m pip uninstall -y diffusers 2>&1 | tail -2 || true
python3 -m pip install -r poc/pod/requirements.txt 2>&1 | grep -E "(Successfully installed|already satisfied|ERROR|Collecting diffusers|Building wheel)" | head -20

log "Installed versions (diffusers / torch / transformers / accelerate)"
python3 -m pip show diffusers torch transformers accelerate 2>/dev/null | \
  grep -E "^(Name|Version|Location):" | sed 's/^/    /'

log "Python interpreter info"
which python3
python3 -c 'import sys; print("    sys.executable:", sys.executable); print("    sys.version:", sys.version)'

log "Disk state after pip install"
show_disk

# ---------------------------------------------------------------------------
# 4. HuggingFace login + warm cache
# ---------------------------------------------------------------------------

log "Configuring HF cache at ${HF_HOME}"
mkdir -p "${HF_HOME}"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1  # faster downloads
# xet downloads (HF's new chunked storage) write temp files via TMPDIR.
# Already pointed at /workspace/tmp at top of script.
export HF_XET_CACHE_DIR="${HF_HOME}/xet"
mkdir -p "${HF_XET_CACHE_DIR}"
python3 -m pip install --quiet hf_transfer

log "Pre-downloading LTX-2 (variant=${LTX2_VARIANT}) — this takes 5-15 min"
log "Disk state before HF download"
show_disk
python3 - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

token = os.environ["HF_TOKEN"]
variant = os.environ.get("LTX2_VARIANT", "distilled-fp8")

# Map variant → which files we actually need. Skip the variants we
# don't plan to load to save bandwidth/disk.
allow_patterns_by_variant = {
    "distilled-fp8":  ["*.json", "*.txt", "*distilled-fp8*", "scheduler/*", "tokenizer*/*", "text_encoder/*", "vae/*"],
    "distilled-bf16": ["*.json", "*.txt", "*distilled-1.1*", "scheduler/*", "tokenizer*/*", "text_encoder/*", "vae/*"],
    "dev":            ["*.json", "*.txt", "*-dev.safetensors", "scheduler/*", "tokenizer*/*", "text_encoder/*", "vae/*"],
}

snapshot_download(
    repo_id="Lightricks/LTX-2",
    cache_dir=os.environ.get("HF_HOME", "/workspace/hf"),
    token=token,
    allow_patterns=allow_patterns_by_variant.get(variant),
    max_workers=4,
)
print("LTX-2 weights cached.")
PYEOF

# Mark bootstrap as complete so the next restart preserves the cache
# instead of wiping it. This sentinel is at /workspace/.bootstrap-complete
# (not under /workspace/hf) so it survives if HF cache layout changes.
touch /workspace/.bootstrap-complete
log "Disk state after HF download"
show_disk

# ---------------------------------------------------------------------------
# 5. Launch the server
# ---------------------------------------------------------------------------

log "Launching pod server on port ${PORT} (variant=${LTX2_VARIANT})"
cd "${WORKDIR}"
export POD_AUTH_TOKEN
export LTX2_VARIANT
export HF_HOME
export PORT

# Use nohup + tee so the user can disconnect their SSH/web shell and the
# server keeps running. Logs to /workspace/poc/server.log so the runner
# can tail it.
exec python3 -u poc/pod/server.py 2>&1 | tee /workspace/poc/server.log
