# Z-Image-Turbo-AIO FP8: headless ComfyUI + OpenAI-compatible image endpoint

This bundle runs **SeeSee21/Z-Image-Turbo-AIO Photorealistic FP8** on a headless ComfyUI backend and exposes an OpenAI-like endpoint:

```text
POST /v1/images/generations
GET  /v1/models
GET  /health
```

It is built for unattended server use: the checkpoint is downloaded at container runtime into `./data`, not baked into the Docker image.

## What this package does

- Starts ComfyUI without requiring browser/UI interaction.
- Downloads `z-image-turbo-fp8-aio.safetensors` from `SeeSee21/Z-Image-Turbo-AIO` into `ComfyUI/models/checkpoints/`.
- Uses a core-node ComfyUI API workflow:
  - `CheckpointLoaderSimple`
  - `CLIPTextEncode`
  - `EmptyLatentImage`
  - `KSampler`
  - `VAEDecode`
  - `SaveImage`
- Exposes an OpenAI-compatible wrapper at `/v1/images/generations`.
- Defaults to the settings recommended by the model card for Photorealistic AIO:
  - steps: `9`
  - CFG: `1.0`
  - sampler: `res_multistep`
  - scheduler: `simple`

## Important honesty note

The Python wrapper and workflow-generation code were syntax/import validated in the packaging environment. The actual GPU inference path requires NVIDIA Docker, CUDA, ComfyUI, and the large model download, so it must be validated on your GPU host.

## Requirements

- Linux server or WSL2 with NVIDIA GPU access
- Docker + Docker Compose
- NVIDIA Container Toolkit
- Enough disk for:
  - Docker images
  - `z-image-turbo-fp8-aio.safetensors`
  - Hugging Face cache in `./data/hf-cache`

The model card claims the FP8 AIO version is the low-VRAM option, but it is still a large image model checkpoint.

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

First boot downloads the checkpoint. Later boots reuse `./data`.

Check health:

```bash
curl http://localhost:8000/health | jq
```

Run test request:

```bash
./scripts/test_curl.sh
```

## OpenAI SDK example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="changeme-local-token",
)

result = client.images.generate(
    model="z-image-turbo-aio-fp8",
    prompt='A photorealistic storefront with a sign reading "LINKSPIX" in large readable letters.',
    size="1024x1024",
    n=1,
    response_format="b64_json",
    extra_body={
        "seed": 123,
        "steps": 9,
        "guidance_scale": 1.0,
        "sampler_name": "res_multistep",
        "scheduler": "simple",
    },
)
```

## API request body

```json
{
  "model": "z-image-turbo-aio-fp8",
  "prompt": "A photorealistic coffee shop sign reading \"COFFEE HOUSE\"",
  "size": "1024x1024",
  "n": 1,
  "response_format": "b64_json",
  "seed": 42,
  "steps": 9,
  "guidance_scale": 1.0,
  "sampler_name": "res_multistep",
  "scheduler": "simple"
}
```

## Files and persistence

```text
./data/       Hugging Face cache and runtime data
./outputs/    ComfyUI output images
```

To remove downloaded weights:

```bash
rm -rf ./data ./outputs
```

To remove Docker build cache/images:

```bash
docker compose down
docker builder prune -af
```

## If `res_multistep` fails

If your ComfyUI build does not expose the `res_multistep` sampler, try editing `.env`:

```env
DEFAULT_SAMPLER=euler
DEFAULT_SCHEDULER=simple
```

But for the AIO Photorealistic workflow, `res_multistep + simple` is the intended setting.

## Security

The wrapper uses a simple bearer token from `.env`:

```env
API_KEY=changeme-local-token
```

Do not expose this directly to the internet without a reverse proxy, TLS, rate limiting, and a stronger key.
