from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipRetriever:
    def __init__(self, model_name: str, device: str = "cuda") -> None:
        resolved_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved_device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.inference_mode()
    def encode_images(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        outputs = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            end = min(start + batch_size, len(image_paths))
            print(f"Encoding images {start + 1}-{end}/{len(image_paths)}")
            images = [Image.open(Path(path)).convert("RGB") for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model.get_image_features(**inputs)
            features = _coerce_feature_tensor(features)
            outputs.append(_normalize(features).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            end = min(start + batch_size, len(texts))
            print(f"Encoding texts {start + 1}-{end}/{len(texts)}")
            inputs = self.processor(text=batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model.get_text_features(**inputs)
            features = _coerce_feature_tensor(features)
            outputs.append(_normalize(features).cpu().numpy())
        return np.concatenate(outputs, axis=0)


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _coerce_feature_tensor(features: torch.Tensor) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported feature output type: {type(features)!r}")
