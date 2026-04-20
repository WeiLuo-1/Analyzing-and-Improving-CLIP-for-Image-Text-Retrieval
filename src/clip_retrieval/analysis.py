from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from clip_retrieval.experiments import FAILURE_CATEGORY_DESCRIPTIONS, heuristic_failure_tags


def build_failure_case_rows(
    similarity: np.ndarray,
    image_ids: list[str],
    captions: list[str],
    caption_to_image: np.ndarray,
) -> list[dict[str, Any]]:
    ranked_rows: list[dict[str, Any]] = []
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
    return ranked_rows


def export_failure_cases(
    output_path: str | Path,
    similarity: np.ndarray,
    image_ids: list[str],
    captions: list[str],
    caption_to_image: np.ndarray,
    top_n: int = 50,
) -> list[dict[str, Any]]:
    output_path = Path(output_path)
    ranked_rows = build_failure_case_rows(
        similarity=similarity,
        image_ids=image_ids,
        captions=captions,
        caption_to_image=caption_to_image,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        for row in ranked_rows[:top_n]:
            handle.write(json.dumps(row) + "\n")
    return ranked_rows


def export_failure_summary(
    output_dir: str | Path,
    ranked_rows: list[dict[str, Any]],
    top_categories: int = 3,
    examples_per_category: int = 3,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_counts = Counter(tag for row in ranked_rows for tag in row["tags"])
    ranked_tags = sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))
    selected_tags = [tag for tag, _ in ranked_tags[:top_categories]]

    summary = {
        "total_failures": len(ranked_rows),
        "category_counts": dict(ranked_tags),
        "top_categories": [],
    }

    for tag in selected_tags:
        category_examples = [row for row in ranked_rows if tag in row["tags"]][:examples_per_category]
        summary["top_categories"].append(
            {
                "tag": tag,
                "count": tag_counts[tag],
                "description": FAILURE_CATEGORY_DESCRIPTIONS.get(tag, ""),
                "examples": category_examples,
            }
        )

    with (output_dir / "failure_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "# Failure Case Summary",
        "",
        f"Total failed text-to-image queries analyzed: {len(ranked_rows)}",
        "",
        "## Category Counts",
        "",
    ]
    for tag, count in ranked_tags:
        percentage = 0.0 if not ranked_rows else (100.0 * count / len(ranked_rows))
        description = FAILURE_CATEGORY_DESCRIPTIONS.get(tag, "")
        lines.append(f"- `{tag}`: {count} failures ({percentage:.1f}%). {description}")

    for category in summary["top_categories"]:
        lines.extend(
            [
                "",
                f"## {category['tag']}",
                "",
                category["description"],
                "",
            ]
        )
        for example in category["examples"]:
            lines.append(
                "- "
                f"Rank {example['rank']}: \"{example['caption']}\" "
                f"(gt={example['ground_truth_image_id']}, pred={example['predicted_image_id']})"
            )

    with (output_dir / "failure_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to failure_cases.jsonl")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    ranked_rows = []
    with open(args.input, "r", encoding="utf-8-sig") as f:
        for line in f:
            ranked_rows.append(json.loads(line))

    export_failure_summary(
        output_dir=args.output,
        ranked_rows=ranked_rows,
    )

    print(" Failure summary generated.")
