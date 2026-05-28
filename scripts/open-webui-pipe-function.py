"""
title: FLUX.2 Klein Manifold Function for OpenAI-Compatible Image Generation.
author: bgeneto
author_url: https://github.com/bgeneto/open-webui-flux-image-gen
funding_url: https://github.com/open-webui
modified: 2026-05-04
version: 0.4.0
license: MIT
requirements: pydantic, aiohttp
environment_variables: IMAGE_API_URL, IMAGE_API_KEY, MODEL_ID, IMAGE_SIZE, NUM_IMAGES, STEPS, CFG_SCALE, RESPONSE_FORMAT, SAMPLING_METHOD, NEGATIVE_PROMPT, REQUEST_TIMEOUT_SECONDS
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
    """OpenWebUI manifold for the FLUX.2 Klein image endpoint."""

    class Valves(BaseModel):
        """Runtime configuration for the configured image endpoint."""

        IMAGE_API_URL: str = Field(
            default="https://imagen.app/v1/images/generations",
            description="OpenAI-compatible image generation endpoint.",
        )
        IMAGE_API_KEY: str = Field(
            default="sk-local",
            description="Bearer token used to authenticate against the image API.",
        )
        MODEL_ID: str = Field(
            default="flux-klein-4b",
            description="Model identifier sent to the image endpoint.",
        )
        IMAGE_SIZE: str = Field(
            default="1024x1024",
            description="Default image size.",
        )
        NUM_IMAGES: int = Field(
            default=1,
            ge=1,
            le=8,
            description="Number of images to request.",
        )
        STEPS: int = Field(
            default=4,
            ge=1,
            le=100,
            description="Default number of sampling steps.",
        )
        CFG_SCALE: float = Field(
            default=1.0,
            ge=0.0,
            le=50.0,
            description="Default CFG scale.",
        )
        RESPONSE_FORMAT: str = Field(
            default="b64_json",
            description="Default response format: b64_json or url.",
        )
        SAMPLING_METHOD: str = Field(
            default="",
            description="Optional sampler name to send with requests.",
        )
        NEGATIVE_PROMPT: str = Field(
            default="",
            description="Optional default negative prompt.",
        )
        REQUEST_TIMEOUT_SECONDS: int = Field(
            default=180,
            ge=1,
            le=600,
            description="HTTP timeout in seconds.",
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "flux.2_klein"
        self.name = "FLUX.2 Klein: "
        self.valves = self.Valves(
            IMAGE_API_URL=os.getenv(
                "IMAGE_API_URL", "https://imagen.webonly.app/v1/images/generations"
            ),
            IMAGE_API_KEY=os.getenv("IMAGE_API_KEY", ""),
            MODEL_ID=os.getenv("MODEL_ID", "flux-klein-4b"),
            IMAGE_SIZE=os.getenv("IMAGE_SIZE", "1024x1024"),
            NUM_IMAGES=os.getenv("NUM_IMAGES", 1),
            STEPS=os.getenv("STEPS", 4),
            CFG_SCALE=os.getenv("CFG_SCALE", 1.0),
            RESPONSE_FORMAT=os.getenv("RESPONSE_FORMAT", "b64_json"),
            SAMPLING_METHOD=os.getenv("SAMPLING_METHOD", ""),
            NEGATIVE_PROMPT=os.getenv("NEGATIVE_PROMPT", ""),
            REQUEST_TIMEOUT_SECONDS=os.getenv("REQUEST_TIMEOUT_SECONDS", 180),
        )

    def pipes(self) -> List[Dict[str, str]]:
        name = "FLUX.2 Klein"
        if not self.valves.IMAGE_API_URL.strip():
            name = "FLUX.2 Klein (endpoint not configured)"
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

    def coerce_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def coerce_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def build_payload(self, body: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "n": self.coerce_int(
                body.get("n", body.get("num_images")), self.valves.NUM_IMAGES
            ),
            "size": body.get("size")
            or body.get("image_size")
            or self.valves.IMAGE_SIZE,
            "response_format": self.normalize_response_format(
                body.get("response_format", self.valves.RESPONSE_FORMAT)
            ),
            "steps": self.coerce_int(
                body.get("steps", body.get("step")), self.valves.STEPS
            ),
            "cfg_scale": self.coerce_float(
                body.get("cfg_scale"), self.valves.CFG_SCALE
            ),
        }

        model_id = self.valves.MODEL_ID.strip()
        if model_id:
            payload["model"] = model_id

        negative_prompt = body.get("negative_prompt") or self.valves.NEGATIVE_PROMPT
        if isinstance(negative_prompt, str) and negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt.strip()

        sampling_method = body.get("sampling_method") or self.valves.SAMPLING_METHOD
        if isinstance(sampling_method, str) and sampling_method.strip():
            payload["sampling_method"] = sampling_method.strip()

        if body.get("seed") is not None:
            payload["seed"] = self.coerce_int(body.get("seed"), -1)

        user = body.get("user")
        if isinstance(user, str) and user.strip():
            payload["user"] = user.strip()

        return payload

    def get_image_media_type(self, image_data: str) -> str:
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
        return f"![FLUX.2 Klein]({data_uri})"

    def data_uri_from_b64(self, image_data: str) -> str:
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
        media_type = content_type.split(";", 1)[0] if content_type else "image/png"
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        if not media_type.startswith("image/"):
            media_type = self.get_image_media_type(encoded_image)
        return f"data:{media_type};base64,{encoded_image}"

    async def markdown_from_url(
        self, session: aiohttp.ClientSession, image_url: str
    ) -> str:
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
        timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT_SECONDS)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.info("Sending FLUX.2 Klein request to %s", self.valves.IMAGE_API_URL)
            logger.debug("Request payload: %s", payload)

            async with session.post(
                self.valves.IMAGE_API_URL,
                headers=self.build_headers(),
                json=payload,
            ) as response:
                if response.status == 401:
                    raise ValueError(
                        "Authentication failed for the configured FLUX.2 Klein endpoint."
                    )

                if response.status >= 400:
                    detail = await response.text()
                    raise ValueError(
                        f"FLUX.2 Klein endpoint failed with HTTP {response.status}: {detail}"
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
            logger.exception("Network error while calling the FLUX.2 Klein endpoint")
            return f"Error: Request to the FLUX.2 Klein endpoint failed: {exc}"
        except ValueError as exc:
            logger.error("FLUX.2 Klein request failed: %s", exc)
            return f"Error: {exc}"
        except Exception as exc:
            logger.exception("Unexpected FLUX.2 Klein error")
            return f"Error: Unexpected FLUX.2 Klein failure: {exc}"
