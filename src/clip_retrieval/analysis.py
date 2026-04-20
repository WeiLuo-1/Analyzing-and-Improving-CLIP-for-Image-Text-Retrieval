from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clip_retrieval.experiments import heuristic_failure_tags


def export_failure_cases(
    output_path: str | Path,
    similarity: np.ndarray,
    image_ids: list[str],
    captions: list[str],
    caption_to_image: np.ndarray,
    top_n: int = 50,
) -> None:
    output_path = Path(output_path)
    ranked_rows = []
    for caption_idx, gt_image_idx in enumerate(caption_to_image):
        ranked_image_indices = np.argsort(-similarity[:, caption_idx])
        rank = int(np.where(ranked_image_indices == gt_image_idx)[0][0])
        if rank == 0:
            continue
        ranked_rows.append(
            {
                "caption_index": caption_idx,
                "caption": captions[caption_idx],
                "ground_truth_image_id": image_ids[int(gt_image_idx)],
                "predicted_image_id": image_ids[int(ranked_image_indices[0])],
                "rank": rank + 1,
                "tags": heuristic_failure_tags(captions[caption_idx]),
            }
        )

    ranked_rows.sort(key=lambda row: row["rank"], reverse=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in ranked_rows[:top_n]:
            handle.write(json.dumps(row) + "\n")
