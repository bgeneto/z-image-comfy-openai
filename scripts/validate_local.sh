#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m compileall api
python - <<'PY'
from api.workflow import build_zimage_aio_prompt
w = build_zimage_aio_prompt(
    prompt='hello', checkpoint_name='z-image-turbo-fp8-aio.safetensors',
    width=1448, height=1448, batch_size=1, seed=1,
    steps=9, cfg=1.0, sampler_name='res_multistep', scheduler='simple',
    denoise=1.0, filename_prefix='test')
assert w['4']['inputs']['ckpt_name'] == 'z-image-turbo-fp8-aio.safetensors'
assert w['3']['inputs']['sampler_name'] == 'res_multistep'
assert w['9']['class_type'] == 'SaveImage'
print('local validation OK')
PY
