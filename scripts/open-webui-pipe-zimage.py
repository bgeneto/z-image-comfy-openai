"""
title: Z-Image Turbo AIO OpenAI-Compatible Pipe Function for OpenAI-Compatible Image Generation.
author: bgeneto
author_url: https://github.com/bgeneto/z-image-comfy-openai
funding_url: https://github.com/comfyanonymous/ComfyUI
modified: 2026-05-28
version: 1.0.0
license: MIT
requirements: pydantic, aiohttp
environment_variables: IMAGE_API_URL, IMAGE_API_KEY, MODEL_ID, IMAGE_SIZE, NUM_IMAGES, STEPS, GUIDANCE_SCALE, RESPONSE_FORMAT, SAMPLER_NAME, SCHEDULER, DENOISE, NEGATIVE_PROMPT, OUTPUT_FORMAT, WEBP_QUALITY, WEBP_LOSSLESS, REQUEST_TIMEOUT_SECONDS, USE_UPSCALER, UPSCALE_BY, UPSCALE_STEPS, UPSCALE_CFG, UPSCALE_SAMPLER_NAME, UPSCALE_SCHEDULER, UPSCALE_DENOISE, UPSCALE_METHOD
"""

import base64
import logging
import os
from typing import Any, Dict, List, Union

import aiohttp
from pydantic import BaseModel, Field

try:
    from open_webui.utils.misc import get_last_user_message
except ImportError:
    from chat_webui.utils.misc import get_last_user_message


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipe:
    """OpenWebUI manifold for the Z-Image Turbo AIO image endpoint via ComfyUI."""

    class Valves(BaseModel):
        """Runtime configuration for the configured image endpoint."""

        IMAGE_API_URL: str = Field(
            default="http://localhost:8000/v1/images/generations",
            description="OpenAI-compatible image generation endpoint.",
        )
        IMAGE_API_KEY: str = Field(
            default="changeme-local-token",
            description="Bearer token used to authenticate against the image API.",
        )
        MODEL_ID: str = Field(
            default="z-image-turbo-aio-fp8",
            description="Model identifier sent to the image endpoint.",
        )
        IMAGE_SIZE: str = Field(
            default="1024x1024",
            description="Default image size (width x height, must be divisible by 8).",
        )
        NUM_IMAGES: int = Field(
            default=1,
            ge=1,
            le=8,
            description="Number of images to request.",
        )
        STEPS: int = Field(
            default=9,
            ge=1,
            le=50,
            description="Default number of sampling steps.",
        )
        GUIDANCE_SCALE: float = Field(
            default=1.0,
            ge=0.0,
            le=20.0,
            description="Default CFG/guidance scale.",
        )
        RESPONSE_FORMAT: str = Field(
            default="b64_json",
            description="Default response format: b64_json or url.",
        )
        SAMPLER_NAME: str = Field(
            default="res_multistep",
            description="Sampler name (e.g., res_multistep, euler, dpmpp_2m).",
        )
        SCHEDULER: str = Field(
            default="simple",
            description="Scheduler for the sampler (e.g., simple, normal, legacy_normal).",
        )
        DENOISE: float = Field(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="Denoising strength (0.0-1.0).",
        )
        NEGATIVE_PROMPT: str = Field(
            default="",
            description="Optional default negative prompt.",
        )
        OUTPUT_FORMAT: str = Field(
            default="webp",
            description="Output image format: png or webp.",
        )
        WEBP_QUALITY: int = Field(
            default=92,
            ge=1,
            le=100,
            description="WebP quality (1-100).",
        )
        WEBP_LOSSLESS: bool = Field(
            default=False,
            description="Enable lossless WebP encoding.",
        )
        REQUEST_TIMEOUT_SECONDS: int = Field(
            default=600,
            ge=1,
            le=3600,
            description="HTTP timeout in seconds.",
        )
        USE_UPSCALER: bool = Field(
            default=False,
            description="Enable high-resolution upscaler pass (doubles generation time).",
        )
        UPSCALE_BY: float = Field(
            default=1.5,
            ge=1.0,
            le=4.0,
            description="Upscale scale factor (1.0-4.0).",
        )
        UPSCALE_STEPS: int = Field(
            default=8,
            ge=1,
            le=50,
            description="Steps for the upscaler pass.",
        )
        UPSCALE_CFG: float = Field(
            default=1.0,
            ge=0.0,
            le=20.0,
            description="CFG scale for the upscaler pass.",
        )
        UPSCALE_SAMPLER_NAME: str = Field(
            default="res_multistep",
            description="Sampler for the upscaler pass.",
        )
        UPSCALE_SCHEDULER: str = Field(
            default="beta",
            description="Scheduler for the upscaler pass.",
        )
        UPSCALE_DENOISE: float = Field(
            default=0.3,
            ge=0.0,
            le=1.0,
            description="Denoising strength for the upscaler pass.",
        )
        UPSCALE_METHOD: str = Field(
            default="lanczos",
            description="Upscale method (lanczos, bicubic, bilinear, nearest).",
        )

    def _get_int(self, name: str, default: int) -> int:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    def _get_float(self, name: str, default: float) -> float:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _get_bool(self, name: str, default: bool) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return val.lower() == "true"

    def __init__(self):
        self.type = "manifold"
        self.id = "z_image_turbo_aio"
        self.name = "Z-Image Turbo AIO: "
        self.valves = self.Valves(
            IMAGE_API_URL=os.getenv(
                "IMAGE_API_URL", "http://localhost:8000/v1/images/generations"
            ),
            IMAGE_API_KEY=os.getenv("IMAGE_API_KEY", "changeme-local-token"),
            MODEL_ID=os.getenv("MODEL_ID", "z-image-turbo-aio-fp8"),
            IMAGE_SIZE=os.getenv("IMAGE_SIZE", "1024x1024"),
            NUM_IMAGES=self._get_int("NUM_IMAGES", 1),
            STEPS=self._get_int("STEPS", 9),
            GUIDANCE_SCALE=self._get_float("GUIDANCE_SCALE", 1.0),
            RESPONSE_FORMAT=os.getenv("RESPONSE_FORMAT", "b64_json"),
            SAMPLER_NAME=os.getenv("SAMPLER_NAME", "res_multistep"),
            SCHEDULER=os.getenv("SCHEDULER", "simple"),
            DENOISE=self._get_float("DENOISE", 1.0),
            NEGATIVE_PROMPT=os.getenv("NEGATIVE_PROMPT", ""),
            OUTPUT_FORMAT=os.getenv("OUTPUT_FORMAT", "webp"),
            WEBP_QUALITY=self._get_int("WEBP_QUALITY", 92),
            WEBP_LOSSLESS=self._get_bool("WEBP_LOSSLESS", False),
            REQUEST_TIMEOUT_SECONDS=self._get_int("REQUEST_TIMEOUT_SECONDS", 600),
            USE_UPSCALER=self._get_bool("USE_UPSCALER", False),
            UPSCALE_BY=self._get_float("UPSCALE_BY", 1.5),
            UPSCALE_STEPS=self._get_int("UPSCALE_STEPS", 8),
            UPSCALE_CFG=self._get_float("UPSCALE_CFG", 1.0),
            UPSCALE_SAMPLER_NAME=os.getenv("UPSCALE_SAMPLER_NAME", "res_multistep"),
            UPSCALE_SCHEDULER=os.getenv("UPSCALE_SCHEDULER", "beta"),
            UPSCALE_DENOISE=self._get_float("UPSCALE_DENOISE", 0.3),
            UPSCALE_METHOD=os.getenv("UPSCALE_METHOD", "lanczos"),
        )

    def pipes(self) -> List[Dict[str, str]]:
        name = "Z-Image Turbo AIO"
        if not self.valves.IMAGE_API_URL.strip():
            name = "Z-Image Turbo AIO (endpoint not configured)"
        return [{"id": self.id, "name": name}]

    def build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.valves.IMAGE_API_KEY.strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def normalize_response_format(self, value: Any) -> str:
        if isinstance(value, str) and value.strip().lower() == "url":
            return "url"
        return "b64_json"

    def normalize_output_format(self, value: Any) -> str:
        if isinstance(value, str) and value.strip().lower() == "png":
            return "png"
        return "webp"

    def coerce_int(
        self, value: Any, default: int, min_val: int = 0, max_val: int = 2**63 - 1
    ) -> int:
        try:
            result = int(value)
            return max(min_val, min(max_val, result))
        except (TypeError, ValueError):
            return default

    def coerce_float(
        self, value: Any, default: float, min_val: float = 0.0, max_val: float = 1.0
    ) -> float:
        try:
            result = float(value)
            return max(min_val, min(max_val, result))
        except (TypeError, ValueError):
            return default

    def build_payload(self, body: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "n": self.coerce_int(
                body.get("n", body.get("num_images")), self.valves.NUM_IMAGES, min_val=1
            ),
            "size": body.get("size")
            or body.get("image_size")
            or self.valves.IMAGE_SIZE,
            "response_format": self.normalize_response_format(
                body.get("response_format", self.valves.RESPONSE_FORMAT)
            ),
            "output_format": self.normalize_output_format(
                body.get("output_format", self.valves.OUTPUT_FORMAT)
            ),
            "steps": self.coerce_int(
                body.get("steps", body.get("step")),
                self.valves.STEPS,
                min_val=1,
                max_val=50,
            ),
            "guidance_scale": self.coerce_float(
                body.get("guidance_scale", body.get("cfg_scale")),
                self.valves.GUIDANCE_SCALE,
                min_val=0.0,
                max_val=20.0,
            ),
            "sampler_name": body.get("sampler_name") or self.valves.SAMPLER_NAME,
            "scheduler": body.get("scheduler") or self.valves.SCHEDULER,
            "denoise": self.coerce_float(
                body.get("denoise"),
                self.valves.DENOISE,
                min_val=0.0,
                max_val=1.0,
            ),
        }

        model_id = self.valves.MODEL_ID.strip()
        if model_id:
            payload["model"] = model_id

        negative_prompt = body.get("negative_prompt") or self.valves.NEGATIVE_PROMPT
        if isinstance(negative_prompt, str) and negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt.strip()

        payload["webp_quality"] = self.coerce_int(
            body.get("webp_quality"), self.valves.WEBP_QUALITY, min_val=1, max_val=100
        )
        payload["webp_lossless"] = bool(
            body.get("webp_lossless")
            if body.get("webp_lossless") is not None
            else self.valves.WEBP_LOSSLESS
        )

        if body.get("seed") is not None:
            payload["seed"] = self.coerce_int(body.get("seed"), -1, min_val=-1)

        # Upscaler configuration (passed through to API)
        payload["use_upscaler"] = (
            body.get("use_upscaler")
            if body.get("use_upscaler") is not None
            else self.valves.USE_UPSCALER
        )
        payload["upscale_by"] = self.coerce_float(
            body.get("upscale_by"), self.valves.UPSCALE_BY, min_val=1.0, max_val=4.0
        )
        payload["upscale_steps"] = self.coerce_int(
            body.get("upscale_steps", body.get("upscale_step")),
            self.valves.UPSCALE_STEPS,
            min_val=1,
            max_val=50,
        )
        payload["upscale_cfg"] = self.coerce_float(
            body.get("upscale_cfg", body.get("upscale_cfg_scale")),
            self.valves.UPSCALE_CFG,
            min_val=0.0,
            max_val=20.0,
        )
        payload["upscale_sampler_name"] = (
            body.get("upscale_sampler_name") or self.valves.UPSCALE_SAMPLER_NAME
        )
        payload["upscale_scheduler"] = (
            body.get("upscale_scheduler") or self.valves.UPSCALE_SCHEDULER
        )
        payload["upscale_denoise"] = self.coerce_float(
            body.get("upscale_denoise"),
            self.valves.UPSCALE_DENOISE,
            min_val=0.0,
            max_val=1.0,
        )
        payload["upscale_method"] = (
            body.get("upscale_method") or self.valves.UPSCALE_METHOD
        )

        user = body.get("user")
        if isinstance(user, str) and user.strip():
            payload["user"] = user.strip()

        return payload

    def get_image_media_type(self, image_data: str) -> str:
        """Detect image MIME type from base64-encoded data header."""
        img_header = image_data[:16]

        if img_header.startswith("/9j/"):
            return "image/jpeg"
        if img_header.startswith("iVBOR"):
            return "image/png"
        if img_header.startswith("R0lG"):
            return "image/gif"
        if img_header.startswith("UklGR"):
            return "image/webp"

        return "image/png"

    def build_markdown_image(self, data_uri: str) -> str:
        return f"![Z-Image Turbo AIO]({data_uri})"

    def data_uri_from_b64(self, image_data: str) -> str:
        """Convert base64 image data to a data URI."""
        if image_data.startswith("data:image/"):
            return image_data

        encoded_image = image_data.split(";base64,", 1)[-1]

        try:
            base64.b64decode(encoded_image, validate=True)
        except Exception as exc:
            raise ValueError("Image endpoint returned invalid base64 data.") from exc

        media_type = self.get_image_media_type(encoded_image)
        return f"data:{media_type};base64,{encoded_image}"

    def data_uri_from_bytes(self, image_bytes: bytes, content_type: str) -> str:
        """Convert raw image bytes to a data URI."""
        media_type = content_type.split(";", 1)[0] if content_type else "image/png"
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        if not media_type.startswith("image/"):
            media_type = self.get_image_media_type(encoded_image)
        return f"data:{media_type};base64,{encoded_image}"

    async def markdown_from_url(
        self, session: aiohttp.ClientSession, image_url: str
    ) -> str:
        """Download an image from URL and convert to markdown data URI."""
        async with session.get(image_url, headers=self.build_headers()) as response:
            if response.status >= 400:
                detail = await response.text()
                raise ValueError(
                    f"Failed to download generated image: HTTP {response.status}: {detail}"
                )
            content_type = response.headers.get("Content-Type", "image/png")
            image_bytes = await response.read()

        return self.build_markdown_image(
            self.data_uri_from_bytes(image_bytes, content_type)
        )

    async def handle_json_response(
        self, session: aiohttp.ClientSession, response: aiohttp.ClientResponse
    ) -> str:
        """Parse and render JSON response from the image endpoint."""
        try:
            payload = await response.json()
        except Exception as exc:
            raise ValueError("Image endpoint returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Image endpoint returned an unexpected JSON payload.")

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError("Image endpoint returned no image data.")

        rendered_images: List[str] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            if isinstance(item.get("b64_json"), str) and item["b64_json"].strip():
                rendered_images.append(
                    self.build_markdown_image(
                        self.data_uri_from_b64(item["b64_json"].strip())
                    )
                )
                continue

            if isinstance(item.get("url"), str) and item["url"].strip():
                rendered_images.append(
                    await self.markdown_from_url(session, item["url"].strip())
                )
                continue

        if not rendered_images:
            raise ValueError(
                "Image endpoint returned items without 'b64_json' or 'url'."
            )

        return "\n\n".join(rendered_images)

    async def generate_image(self, payload: Dict[str, Any]) -> str:
        """Send image generation request to the Z-Image endpoint."""
        timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT_SECONDS)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.info(
                "Sending Z-Image Turbo AIO request to %s", self.valves.IMAGE_API_URL
            )
            logger.debug("Request payload: %s", payload)

            async with session.post(
                self.valves.IMAGE_API_URL,
                headers=self.build_headers(),
                json=payload,
            ) as response:
                if response.status == 401:
                    raise ValueError(
                        "Authentication failed for the configured Z-Image endpoint."
                    )

                if response.status >= 400:
                    detail = await response.text()
                    raise ValueError(
                        f"Z-Image Turbo AIO endpoint failed with HTTP {response.status}: {detail}"
                    )

                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    return await self.handle_json_response(session, response)

                if "image/" in content_type:
                    image_bytes = await response.read()
                    return self.build_markdown_image(
                        self.data_uri_from_bytes(image_bytes, content_type)
                    )

                detail = await response.text()
                raise ValueError(
                    f"Unsupported response content type '{content_type}': {detail}"
                )

    async def pipe(self, body: Dict[str, Any]) -> Union[str, List[Dict[str, str]]]:
        """Main entry point for the OpenWebUI pipe function."""
        prompt = body.get("prompt")
        if not prompt:
            prompt = get_last_user_message(body.get("messages", []))

        if not isinstance(prompt, str) or not prompt.strip():
            logger.error("No prompt found in the request body.")
            return "Error: No prompt provided."

        if not self.valves.IMAGE_API_URL.strip():
            logger.error("IMAGE_API_URL is not configured.")
            return "Error: IMAGE_API_URL is not configured."

        payload = self.build_payload(body, prompt.strip())

        try:
            return await self.generate_image(payload)
        except aiohttp.ClientError as exc:
            logger.error("Network error while calling the Z-Image endpoint")
            return f"Error: Request to the Z-Image endpoint failed: {exc}"
        except ValueError as exc:
            logger.error("Z-Image Turbo AIO request failed: %s", exc)
            return f"Error: {exc}"
        except Exception as exc:
            logger.error("Unexpected Z-Image Turbo AIO error")
            return f"Error: Unexpected Z-Image Turbo AIO failure: {exc}"
