#!/usr/bin/env bash
set -euo pipefail

COMFY_DIR=/opt/ComfyUI
DATA_DIR=/data
MODEL_REPO=${MODEL_REPO:-SeeSee21/Z-Image-Turbo-AIO}
MODEL_FILE=${MODEL_FILE:-z-image-turbo-fp8-aio.safetensors}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-z-image-turbo-fp8-aio.safetensors}
INSTALL_OPTIONAL_NODES=${INSTALL_OPTIONAL_NODES:-true}
COMFYUI_ARGS=${COMFYUI_ARGS:---listen 0.0.0.0 --port 8188 --disable-auto-launch}

mkdir -p "$DATA_DIR/hf-cache" "$COMFY_DIR/models/checkpoints" "$COMFY_DIR/custom_nodes"

# Download checkpoint only at container runtime, never at image build time.
if [ ! -f "$COMFY_DIR/models/checkpoints/$CHECKPOINT_NAME" ]; then
  echo "[bootstrap] Downloading $MODEL_REPO/$MODEL_FILE -> models/checkpoints/$CHECKPOINT_NAME"
  python - <<PY
from huggingface_hub import hf_hub_download
import os, shutil
repo = os.environ.get('MODEL_REPO', '$MODEL_REPO')
filename = os.environ.get('MODEL_FILE', '$MODEL_FILE')
dst = os.path.join('$COMFY_DIR', 'models', 'checkpoints', os.environ.get('CHECKPOINT_NAME', '$CHECKPOINT_NAME'))
path = hf_hub_download(repo_id=repo, filename=filename, cache_dir=os.environ.get('HF_HOME', '/data/hf-cache'))
os.makedirs(os.path.dirname(dst), exist_ok=True)
if not os.path.exists(dst):
    try:
        os.symlink(path, dst)
    except OSError:
        shutil.copy2(path, dst)
print(path)
PY
else
  echo "[bootstrap] Checkpoint already present: models/checkpoints/$CHECKPOINT_NAME"
fi

# Optional nodes are not required by the API workflow, which uses core nodes only.
# They are useful if you later open the UI and import the official AIO workflow JSON.
if [ "$INSTALL_OPTIONAL_NODES" = "true" ]; then
  cd "$COMFY_DIR/custom_nodes"
  if [ ! -d rgthree-comfy ]; then
    echo "[bootstrap] Installing rgthree-comfy"
    git clone --depth=1 https://github.com/rgthree/rgthree-comfy.git rgthree-comfy || true
  fi
  if [ ! -d comfyui_image_metadata_extension ]; then
    echo "[bootstrap] Installing comfyui_image_metadata_extension"
    git clone --depth=1 https://github.com/nkchocoai/ComfyUI-ImageMetadataExtension.git comfyui_image_metadata_extension || true
  fi
fi

cd "$COMFY_DIR"
echo "[bootstrap] Starting ComfyUI: python main.py $COMFYUI_ARGS"
exec python main.py $COMFYUI_ARGS
