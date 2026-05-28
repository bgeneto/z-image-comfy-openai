from __future__ import annotations

import random
from typing import Any


def build_zimage_aio_prompt(
    *,
    prompt: str,
    checkpoint_name: str,
    width: int,
    height: int,
    batch_size: int,
    seed: int | None,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    filename_prefix: str,
    use_upscaler: bool = False,
    upscale_by: float = 1.5,
    upscale_steps: int = 8,
    upscale_cfg: float = 1.0,
    upscale_sampler_name: str = "res_multistep",
    upscale_scheduler: str = "beta",
    upscale_denoise: float = 0.3,
    upscale_method: str = "lanczos",
) -> dict[str, Any]:
    """Build a ComfyUI API-format workflow.

    When use_upscaler=False:
        Base txt2img only with ModelSamplingAuraFlow enhancement.

    When use_upscaler=True:
        Approximates the ZIT-AIO-v2.0 improved upscaler path:

        base KSampler
        -> VAEDecode
        -> ImageScaleBy
        -> VAEEncode
        -> second KSampler
        -> VAEDecode
        -> SaveImage

    This avoids UI-only nodes like rgthree bypassers and avoids requiring
    SaveImageWithMetaData. It keeps the workflow API-safe and headless.
    """
    if seed is None or seed < 0:
        seed = random.randint(0, 2**63 - 1)

    graph: dict[str, Any] = {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint_name},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": batch_size,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["4", 1],
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": ["4", 1],
            },
        },
        # ModelSamplingAuraFlow enhances sampling for Aura-style models.
        "2": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "shift": 3.0,
                "model": ["4", 0],
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": ["2", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
        },
    }

    if not use_upscaler:
        graph["9"] = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["8", 0],
            },
        }
        return graph

    # Upscaler path: ImageScaleBy -> VAEEncode -> second KSampler -> VAEDecode
    graph.update(
        {
            "44": {
                "class_type": "ImageScaleBy",
                "inputs": {
                    "upscale_method": upscale_method,
                    "scale_by": upscale_by,
                    "image": ["8", 0],
                },
            },
            "45": {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["44", 0],
                    "vae": ["4", 2],
                },
            },
            "14": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed + 1,
                    "steps": upscale_steps,
                    "cfg": upscale_cfg,
                    "sampler_name": upscale_sampler_name,
                    "scheduler": upscale_scheduler,
                    "denoise": upscale_denoise,
                    "model": ["2", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["45", 0],
                },
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["14", 0],
                    "vae": ["4", 2],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": f"{filename_prefix}_upscaled",
                    "images": ["12", 0],
                },
            },
        }
    )

    return graph
