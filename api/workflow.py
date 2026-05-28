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
) -> dict[str, Any]:
    """Build a ComfyUI API-format workflow using core nodes only.

    This is intentionally not the UI-format workflow from Hugging Face.
    The SeeSee21 AIO checkpoint is meant to be loaded with Load Checkpoint,
    so a standard CheckpointLoaderSimple -> CLIPTextEncode -> KSampler -> VAEDecode
    graph is enough for unattended API use.
    """
    if seed is None or seed < 0:
        seed = random.randint(0, 2**63 - 1)

    return {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint_name},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch_size},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["4", 1]},
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
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]},
        },
    }
