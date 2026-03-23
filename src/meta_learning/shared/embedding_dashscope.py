from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx

from meta_learning.shared.models import DashScopeConfig

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".ico"}
)


def _encode_image_to_data_uri(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower().lstrip(".")
    if suffix == "jpg":
        suffix = "jpeg"
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:image/{suffix};base64,{b64}"


class MultimodalEmbedding:
    def __init__(self, config: DashScopeConfig) -> None:
        self._config = config

    def embed(self, text: str, image_paths: list[str] | None = None) -> list[float]:
        fused_entry: dict[str, str] = {"text": text}

        if image_paths:
            first_image = image_paths[0]
            if first_image.startswith(("http://", "https://", "data:")):
                fused_entry["image"] = first_image
            else:
                fused_entry["image"] = _encode_image_to_data_uri(first_image)

        contents: list[dict[str, str]] = [fused_entry]

        payload = {
            "model": self._config.model,
            "input": {"contents": contents},
            "parameters": {
                "dimension": self._config.dimension,
                "output_type": "dense",
            },
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                self._config.base_url,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        embeddings = data.get("output", {}).get("embeddings", [])
        if not embeddings:
            raise ValueError(f"DashScope returned no embeddings: {data}")

        return embeddings[0]["embedding"]

    def embed_text_only(self, text: str) -> list[float]:
        return self.embed(text, image_paths=None)

    def make_embedding_fn(
        self,
        image_lookup: dict[str, list[str]] | None = None,
    ):
        lookup = image_lookup or {}

        def _fn(text: str) -> list[float]:
            images = lookup.get(text)
            return self.embed(text, image_paths=images)

        return _fn
