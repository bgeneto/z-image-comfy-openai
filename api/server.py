from __future__ import annotations

import asyncio
import base64
import os
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from .workflow import build_zimage_aio_prompt

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188").rstrip("/")
API_KEY = os.getenv("API_KEY", "changeme-local-token")
CHECKPOINT_NAME = os.getenv("CHECKPOINT_NAME", "z-image-turbo-fp8-aio.safetensors")
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "9"))
DEFAULT_CFG = float(os.getenv("DEFAULT_CFG", "1.0"))
DEFAULT_SAMPLER = os.getenv("DEFAULT_SAMPLER", "res_multistep")
DEFAULT_SCHEDULER = os.getenv("DEFAULT_SCHEDULER", "simple")
DEFAULT_DENOISE = float(os.getenv("DEFAULT_DENOISE", "1.0"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "1.0"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "1"))
DEFAULT_OUTPUT_FORMAT = os.getenv("DEFAULT_OUTPUT_FORMAT", "webp")
WEBP_QUALITY = int(os.getenv("WEBP_QUALITY", "92"))
WEBP_LOSSLESS = os.getenv("WEBP_LOSSLESS", "false").lower() == "true"
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
DEFAULT_SIZE = f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}"

# Warmup configuration
WARMUP_ENABLED = os.getenv("WARMUP_ENABLED", "false").lower() == "true"
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "2"))
WARMUP_SIZE = int(os.getenv("WARMUP_SIZE", "512"))
WARMUP_DELAY = int(os.getenv("WARMUP_DELAY", "10"))

app = FastAPI(title="Z-Image-Turbo-AIO OpenAI-compatible image API", version="1.0.0")


async def _warmup_model():
    """Perform a silent warmup generation on startup to reduce first-request latency."""
    if not WARMUP_ENABLED:
        print("[WARMUP] Disabled via WARMUP_ENABLED=false")
        return

    await asyncio.sleep(WARMUP_DELAY)

    try:
        graph = build_zimage_aio_prompt(
            prompt="",
            checkpoint_name=CHECKPOINT_NAME,
            width=WARMUP_SIZE,
            height=WARMUP_SIZE,
            batch_size=1,
            seed=0,
            steps=WARMUP_STEPS,
            cfg=DEFAULT_CFG,
            sampler_name=DEFAULT_SAMPLER,
            scheduler=DEFAULT_SCHEDULER,
            denoise=DEFAULT_DENOISE,
            filename_prefix="warmup",
        )

        timeout = httpx.Timeout(120, connect=30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            prompt_id = await _submit_prompt(client, graph)
            await _wait_history(client, prompt_id)
        print(
            f"[WARMUP] Model warmed up successfully ({WARMUP_SIZE}x{WARMUP_SIZE}, {WARMUP_STEPS} steps)"
        )
    except Exception as e:
        print(f"[WARMUP] Warning: warmup failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Trigger warmup on server startup."""
    asyncio.create_task(_warmup_model())


class ImageGenerationRequest(BaseModel):
    model: str = Field(default="z-image-turbo-aio-fp8")
    prompt: str = Field(min_length=1)
    n: int = Field(default=1, ge=1, le=MAX_IMAGES)
    size: str = Field(default=DEFAULT_SIZE, pattern=r"^\d+x\d+$")
    response_format: Literal["b64_json", "url"] = "b64_json"
    seed: int | None = None

    # Output format: png or webp (default: from DEFAULT_OUTPUT_FORMAT env)
    output_format: Literal["png", "webp"] = Field(
        default=cast(Literal["png", "webp"], DEFAULT_OUTPUT_FORMAT)
    )
    webp_quality: int = Field(default=WEBP_QUALITY, ge=1, le=100)
    webp_lossless: bool = Field(default=WEBP_LOSSLESS)

    # Non-OpenAI extras. OpenAI SDK can pass these via extra_body={...}.
    steps: int | None = Field(default=None, ge=1, le=50)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    sampler_name: str | None = None
    scheduler: str | None = None
    denoise: float | None = Field(default=None, ge=0.0, le=1.0)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"


def _check_auth(authorization: str | None) -> None:
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def _parse_size(size: str) -> tuple[int, int]:
    width_s, height_s = size.lower().split("x", 1)
    width, height = int(width_s), int(height_s)
    if width < 256 or height < 256 or width > 2048 or height > 2048:
        raise HTTPException(
            status_code=400, detail="size must be between 256x256 and 2048x2048"
        )
    if width % 8 or height % 8:
        raise HTTPException(
            status_code=400, detail="width and height must be divisible by 8"
        )
    return width, height


def convert_image_bytes_to_webp(
    image_bytes: bytes,
    *,
    quality: int = 92,
    lossless: bool = False,
    method: int = 6,
) -> bytes:
    img = Image.open(BytesIO(image_bytes))

    # Preserve alpha if present
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")

    out = BytesIO()
    img.save(
        out,
        format="WEBP",
        quality=quality,
        lossless=lossless,
        method=method,
    )
    return out.getvalue()


def _process_image_from_comfyui(
    image_bytes: bytes,
    output_format: str = "webp",
    webp_quality: int = 92,
    webp_lossless: bool = False,
) -> tuple[bytes, str, str]:
    """Process image bytes according to output format specification.

    Returns:
        Tuple of (final_bytes, extension, mime_type)
    """
    fmt = output_format.lower()
    if fmt == "webp":
        final_bytes = convert_image_bytes_to_webp(
            image_bytes,
            quality=webp_quality,
            lossless=webp_lossless,
        )
        return final_bytes, "webp", "image/webp"
    else:
        # PNG: keep original bytes (ComfyUI already returns PNG)
        return image_bytes, "png", "image/png"


async def _submit_prompt(
    client: httpx.AsyncClient, prompt_graph: dict[str, Any]
) -> str:
    payload = {"prompt": prompt_graph, "client_id": str(uuid.uuid4())}
    r = await client.post(f"{COMFYUI_URL}/prompt", json=payload)
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502, detail={"comfyui_status": r.status_code, "body": r.text}
        )
    data = r.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail=data)
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise HTTPException(
            status_code=502,
            detail={"error": "ComfyUI did not return prompt_id", "response": data},
        )
    return prompt_id


async def _wait_history(client: httpx.AsyncClient, prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
    last: Any = None
    while time.monotonic() < deadline:
        r = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
        if r.status_code < 400:
            data = r.json()
            last = data
            if prompt_id in data:
                item = data[prompt_id]
                # ComfyUI records node errors in status/messages depending on version.
                status = item.get("status", {})
                if status.get("status_str") == "error":
                    raise HTTPException(
                        status_code=502,
                        detail={"error": "ComfyUI workflow failed", "history": item},
                    )
                return item
        await client.get(
            f"{COMFYUI_URL}/queue"
        )  # cheap liveness check; useful for early connection errors
        time.sleep(POLL_INTERVAL_SECONDS)
    raise HTTPException(
        status_code=504, detail={"error": "Timed out waiting for ComfyUI", "last": last}
    )


def _extract_images(history_item: dict[str, Any]) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    outputs = history_item.get("outputs", {})
    for out in outputs.values():
        for img in out.get("images", []) or []:
            filename = img.get("filename")
            if filename:
                images.append(
                    {
                        "filename": filename,
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    }
                )
    return images


async def _save_image_as_output(
    client: httpx.AsyncClient,
    image: dict[str, str],
    output_format: str = "webp",
    webp_quality: int = 92,
    webp_lossless: bool = False,
) -> str:
    """Fetch image from ComfyUI, convert to output format, save to OUTPUT_DIR, and return URL."""
    r = await client.get(f"{COMFYUI_URL}/view", params=image)
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Failed to fetch image",
                "status": r.status_code,
                "body": r.text,
            },
        )
    final_bytes, ext, _ = _process_image_from_comfyui(
        r.content, output_format, webp_quality, webp_lossless
    )
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = OUTPUT_DIR / filename
    with open(path, "wb") as f:
        f.write(final_bytes)
    params = f"filename={filename}&subfolder=&type=output"
    return f"{COMFYUI_URL}/view?{params}"


async def _fetch_image_b64(
    client: httpx.AsyncClient,
    image: dict[str, str],
    output_format: str = "webp",
    webp_quality: int = 92,
    webp_lossless: bool = False,
) -> str:
    r = await client.get(f"{COMFYUI_URL}/view", params=image)
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Failed to fetch image",
                "status": r.status_code,
                "body": r.text,
            },
        )
    final_bytes, _, _ = _process_image_from_comfyui(
        r.content, output_format, webp_quality, webp_lossless
    )
    return base64.b64encode(final_bytes).decode("utf-8")


@app.get("/health")
async def health() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{COMFYUI_URL}/system_stats")
        return {
            "ok": r.status_code < 400,
            "comfyui_status": r.status_code,
            "comfyui_url": COMFYUI_URL,
        }


@app.get("/ready")
async def readiness_check() -> dict[str, Any]:
    """Readiness probe that also warms the model on first call."""
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            ready_size = min(WARMUP_SIZE, 256)
            ready_graph = build_zimage_aio_prompt(
                prompt="",
                checkpoint_name=CHECKPOINT_NAME,
                width=ready_size,
                height=ready_size,
                batch_size=1,
                seed=0,
                steps=1,
                cfg=DEFAULT_CFG,
                sampler_name=DEFAULT_SAMPLER,
                scheduler=DEFAULT_SCHEDULER,
                denoise=DEFAULT_DENOISE,
                filename_prefix="ready",
            )
            prompt_id = await _submit_prompt(client, ready_graph)
            await _wait_history(client, prompt_id)
            return {"status": "ready", "warmed": True}
        except Exception as e:
            return {"status": "not_ready", "error": str(e)}


@app.get("/v1/models")
async def list_models(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [ModelCard(id="z-image-turbo-aio-fp8").model_dump()],
    }


@app.post("/v1/images/generations")
async def image_generations(
    req: ImageGenerationRequest, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    _check_auth(authorization)
    if req.model not in {"z-image-turbo-aio-fp8", "z-image-turbo-aio", "z-image-turbo"}:
        raise HTTPException(status_code=404, detail=f"Unknown model: {req.model}")

    width, height = _parse_size(req.size)
    graph = build_zimage_aio_prompt(
        prompt=req.prompt,
        checkpoint_name=CHECKPOINT_NAME,
        width=width,
        height=height,
        batch_size=req.n,
        seed=req.seed,
        steps=req.steps or DEFAULT_STEPS,
        cfg=req.guidance_scale if req.guidance_scale is not None else DEFAULT_CFG,
        sampler_name=req.sampler_name or DEFAULT_SAMPLER,
        scheduler=req.scheduler or DEFAULT_SCHEDULER,
        denoise=req.denoise if req.denoise is not None else DEFAULT_DENOISE,
        filename_prefix="zimage_openai",
    )

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS + 30, connect=30)
    async with httpx.AsyncClient(timeout=timeout) as client:
        prompt_id = await _submit_prompt(client, graph)
        history_item = await _wait_history(client, prompt_id)
        images = _extract_images(history_item)
        if not images:
            raise HTTPException(
                status_code=502,
                detail={"error": "No image outputs found", "history": history_item},
            )

        data: list[dict[str, str]] = []
        for image in images[: req.n]:
            if req.response_format == "url":
                output_url = await _save_image_as_output(
                    client,
                    image,
                    req.output_format,
                    req.webp_quality,
                    req.webp_lossless,
                )
                data.append({"url": output_url})
            else:
                b64 = await _fetch_image_b64(
                    client,
                    image,
                    req.output_format,
                    req.webp_quality,
                    req.webp_lossless,
                )
                data.append({"b64_json": b64})

    return {"created": int(time.time()), "data": data}
