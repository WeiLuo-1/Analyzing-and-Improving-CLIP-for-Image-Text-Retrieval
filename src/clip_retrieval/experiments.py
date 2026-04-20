from __future__ import annotations

import re

import numpy as np


FAILURE_CATEGORY_DESCRIPTIONS = {
    "crowded_scene": "Captions describing crowds, groups, or busy public scenes where many similar people/objects compete for attention.",
    "fine_grained_attributes": "Captions that depend on subtle attributes such as clothing, colors, pose, or small object details.",
    "multiple_objects": "Captions involving several entities or relationships, which makes one-to-one image matching harder.",
    "generic_human_activity": "Short person-centric captions with weak visual specificity, such as a person walking, standing, or waiting outside.",
    "ambiguous_query": "Captions containing words with multiple meanings or underspecified object references.",
    "unusual_scene": "Captions describing uncommon, stylized, or visually atypical scenes.",
}

PROMPT_STRATEGIES = {
    "none": [],
    "photo": ["a photo of {}"],
    "image": ["an image of {}"],
    "photo_image": ["a photo of {}", "an image of {}"],
    "detailed_scene": ["a photo of a scene where {}", "an image showing {}"],
}

PERSON_TERMS = {
    "man",
    "woman",
    "person",
    "people",
    "boy",
    "girl",
    "child",
    "children",
    "guy",
    "lady",
    "worker",
    "crowd",
    "group",
}
GROUP_TERMS = {
    "crowd",
    "crowds",
    "group",
    "groups",
    "people",
    "several",
    "many",
    "audience",
    "gathered",
    "metro",
    "station",
}
COLOR_TERMS = {
    "black",
    "white",
    "blue",
    "green",
    "red",
    "yellow",
    "orange",
    "purple",
    "gray",
    "grey",
    "brown",
    "tan",
    "bright",
    "blond",
}
CLOTHING_TERMS = {
    "shirt",
    "pants",
    "jacket",
    "hat",
    "cap",
    "overalls",
    "boots",
    "dress",
    "coat",
    "uniform",
    "helmet",
    "hard",
    "scarf",
}
GENERIC_ACTIVITY_TERMS = {
    "walking",
    "standing",
    "looking",
    "watching",
    "waiting",
    "moving",
    "posing",
    "talking",
}
RELATION_TERMS = {
    "holding",
    "pushing",
    "carrying",
    "watching",
    "driving",
    "next",
    "while",
    "together",
}


def get_prompt_templates(prompt_type: str, extra_templates: list[str] | None = None) -> list[str]:
    if prompt_type not in PROMPT_STRATEGIES:
        raise ValueError(f"Unsupported prompt strategy '{prompt_type}'.")
    templates = list(PROMPT_STRATEGIES[prompt_type])
    if extra_templates:
        templates.extend(extra_templates)
    return templates


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
    if strategy not in {"topk_consensus", "caption_prior"}:
        raise ValueError(f"Unsupported rerank strategy '{strategy}'.")

    reranked = similarity.copy()
    caption_consensus = _caption_consensus(image_to_caption_sets, similarity.shape[1])
    if strategy == "caption_prior":
        for image_idx in range(similarity.shape[0]):
            ranked_caption_indices = np.argsort(-similarity[image_idx])[:top_k]
            reranked[image_idx, ranked_caption_indices] += 0.03 * caption_consensus[ranked_caption_indices]
        return reranked

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
    person_count = sum(1 for token in tokens if token in PERSON_TERMS)

    if tokens & AMBIGUOUS_TERMS:
        tags.append("ambiguous_query")
    if tokens & UNUSUAL_SCENE_TERMS:
        tags.append("unusual_scene")
    if tokens & GROUP_TERMS or ("crowd" in lowered) or ("group" in lowered):
        tags.append("crowded_scene")
    if (tokens & COLOR_TERMS and tokens & CLOTHING_TERMS) or len(tokens & COLOR_TERMS) >= 2:
        tags.append("fine_grained_attributes")
    if " and " in lowered or lowered.count(",") >= 2 or person_count >= 2 or tokens & RELATION_TERMS:
        tags.append("multiple_objects")
    if tokens & PERSON_TERMS and (tokens & GENERIC_ACTIVITY_TERMS or len(tokens) <= 8):
        tags.append("generic_human_activity")
    if not tags:
        tags.append("fine_grained_attributes" if tokens & COLOR_TERMS else "generic_human_activity")
    # Keep order stable while deduplicating.
    return list(dict.fromkeys(tags))
