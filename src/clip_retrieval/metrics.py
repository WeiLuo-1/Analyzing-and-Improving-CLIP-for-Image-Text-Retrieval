from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievalMetrics:
    image_to_text_r1: float
    image_to_text_r5: float
    image_to_text_r10: float
    text_to_image_r1: float
    text_to_image_r5: float
    text_to_image_r10: float

    def to_dict(self) -> dict[str, float]:
        return {
            "image_to_text_r1": self.image_to_text_r1,
            "image_to_text_r5": self.image_to_text_r5,
            "image_to_text_r10": self.image_to_text_r10,
            "text_to_image_r1": self.text_to_image_r1,
            "text_to_image_r5": self.text_to_image_r5,
            "text_to_image_r10": self.text_to_image_r10,
        }


def _recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks < k))


def compute_retrieval_metrics(similarity: np.ndarray, caption_to_image: np.ndarray, image_to_caption_sets: list[np.ndarray]) -> RetrievalMetrics:
    image_to_text_ranks = []
    for image_idx, gt_caption_indices in enumerate(image_to_caption_sets):
        ranked_caption_indices = np.argsort(-similarity[image_idx])
        best_rank = min(int(np.where(ranked_caption_indices == gt_idx)[0][0]) for gt_idx in gt_caption_indices)
        image_to_text_ranks.append(best_rank)

    text_to_image_ranks = []
    for caption_idx, gt_image_idx in enumerate(caption_to_image):
        ranked_image_indices = np.argsort(-similarity[:, caption_idx])
        rank = int(np.where(ranked_image_indices == gt_image_idx)[0][0])
        text_to_image_ranks.append(rank)

    image_to_text_ranks = np.asarray(image_to_text_ranks)
    text_to_image_ranks = np.asarray(text_to_image_ranks)
    return RetrievalMetrics(
        image_to_text_r1=_recall_at_k(image_to_text_ranks, 1),
        image_to_text_r5=_recall_at_k(image_to_text_ranks, 5),
        image_to_text_r10=_recall_at_k(image_to_text_ranks, 10),
        text_to_image_r1=_recall_at_k(text_to_image_ranks, 1),
        text_to_image_r5=_recall_at_k(text_to_image_ranks, 5),
        text_to_image_r10=_recall_at_k(text_to_image_ranks, 10),
    )
