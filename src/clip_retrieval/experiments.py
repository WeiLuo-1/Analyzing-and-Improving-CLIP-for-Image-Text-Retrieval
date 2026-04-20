from __future__ import annotations

import re

import numpy as np


def apply_prompt_templates(captions: list[str], templates: list[str]) -> list[str]:
    if not templates:
        return captions
    prompted: list[str] = []
    for caption in captions:
        for template in templates:
            prompted.append(template.format(caption))
    return prompted


def aggregate_prompt_embeddings(embeddings: np.ndarray, num_templates: int) -> np.ndarray:
    if num_templates <= 1:
        return embeddings
    grouped = embeddings.reshape(-1, num_templates, embeddings.shape[-1])
    averaged = grouped.mean(axis=1)
    norms = np.linalg.norm(averaged, axis=1, keepdims=True)
    return averaged / np.clip(norms, 1e-12, None)


def rerank_similarity(
    similarity: np.ndarray,
    image_to_caption_sets: list[np.ndarray],
    strategy: str,
    top_k: int,
) -> np.ndarray:
    if strategy == "none":
        return similarity
    if strategy != "topk_consensus":
        raise ValueError(f"Unsupported rerank strategy '{strategy}'.")

    reranked = similarity.copy()
    caption_consensus = _caption_consensus(image_to_caption_sets, similarity.shape[1])
    image_consensus = np.asarray([caption_consensus[caption_indices].mean() for caption_indices in image_to_caption_sets])

    for image_idx in range(similarity.shape[0]):
        ranked_caption_indices = np.argsort(-similarity[image_idx])[:top_k]
        reranked[image_idx, ranked_caption_indices] += 0.05 * caption_consensus[ranked_caption_indices]

    for caption_idx in range(similarity.shape[1]):
        ranked_image_indices = np.argsort(-similarity[:, caption_idx])[:top_k]
        reranked[ranked_image_indices, caption_idx] += 0.05 * image_consensus[ranked_image_indices]
    return reranked


def _caption_consensus(image_to_caption_sets: list[np.ndarray], num_captions: int) -> np.ndarray:
    consensus = np.zeros(num_captions, dtype=np.float32)
    for caption_indices in image_to_caption_sets:
        size = len(caption_indices)
        if size == 0:
            continue
        consensus[caption_indices] = 1.0 / size
    return consensus


AMBIGUOUS_TERMS = {"apple", "bank", "bat", "crane", "mouse", "seal", "spring"}
UNUSUAL_SCENE_TERMS = {
    "abstract",
    "surreal",
    "strange",
    "weird",
    "costume",
    "painting",
    "statue",
    "cartoon",
}


def heuristic_failure_tags(caption: str) -> list[str]:
    lowered = caption.lower()
    tokens = set(re.findall(r"[a-z]+", lowered))
    tags: list[str] = []
    if tokens & AMBIGUOUS_TERMS:
        tags.append("ambiguous_query")
    if " and " in lowered or lowered.count(",") >= 2:
        tags.append("multiple_objects")
    if tokens & UNUSUAL_SCENE_TERMS:
        tags.append("unusual_scene")
    if not tags:
        tags.append("other")
    return tags
