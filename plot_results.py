from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_COLUMNS = [
    "image_to_text_r1",
    "image_to_text_r5",
    "image_to_text_r10",
    "text_to_image_r1",
    "text_to_image_r5",
    "text_to_image_r10",
]
FAILURE_TAGS = [
    "generic_human_activity",
    "multiple_objects",
    "fine_grained_attributes",
    "crowded_scene",
    "ambiguous_query",
    "unusual_scene",
]
LABEL_OVERRIDES = {
    "baseline": "Baseline",
    "flickr30k_baseline_200": "Baseline (200)",
    "prompt_photo_image": "Photo+Image Prompt",
    "prompt_detailed": "Detailed Prompt",
    "rerank_caption_prior": "Rerank",
    "smoke_test": "Smoke Test",
}
PREFERRED_ORDER = [
    "baseline",
    "flickr30k_baseline_200",
    "baseline_2000",
    "prompt_photo_image",
    "prompt_photo_image_2000",
    "prompt_detailed",
    "prompt_detailed_2000",
    "rerank_caption_prior",
    "rerank_caption_prior_2000",
    "smoke_test",
]
BAR_COLORS = ["#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B", "#84A59D"]


def prettify_label(experiment_name: str) -> str:
    if experiment_name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[experiment_name]
    if experiment_name.endswith("_2000"):
        base_name = experiment_name[: -len("_2000")]
        return f"{prettify_label(base_name)} (2000)"
    return experiment_name.replace("_", " ").title()


def load_experiment_rows(outputs_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for experiment_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        metrics_path = experiment_dir / "metrics.json"
        config_path = experiment_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue

        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        failure_counts = {tag: 0 for tag in FAILURE_TAGS}
        failure_summary_path = experiment_dir / "failure_summary.json"
        total_failures: int | str = ""
        if failure_summary_path.exists():
            with failure_summary_path.open("r", encoding="utf-8") as handle:
                failure_summary = json.load(handle)
            category_counts = failure_summary.get("category_counts", {})
            total_failures = int(failure_summary.get("total_failures", 0))
            for tag in FAILURE_TAGS:
                failure_counts[tag] = int(category_counts.get(tag, 0))

        row: dict[str, object] = {
            "experiment_name": experiment_dir.name,
            "display_name": prettify_label(experiment_dir.name),
            "dataset": config.get("dataset", ""),
            "split": config.get("split", ""),
            "max_examples": config.get("max_examples"),
            "prompt_type": config.get("prompt_type", "none"),
            "prompt_templates": " | ".join(config.get("prompt_templates", [])),
            "rerank_strategy": config.get("rerank_strategy", "none"),
            "rerank_k": config.get("rerank_k", ""),
            "device": config.get("device", ""),
            "total_failures": total_failures,
        }
        for metric_name in METRIC_COLUMNS:
            row[metric_name] = float(metrics.get(metric_name, 0.0))
        row.update(failure_counts)
        rows.append(row)

    order_index = {name: idx for idx, name in enumerate(PREFERRED_ORDER)}
    rows.sort(key=lambda row: (order_index.get(str(row["experiment_name"]), 10_000), str(row["experiment_name"])))
    return rows


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "experiment_name",
        "display_name",
        "dataset",
        "split",
        "max_examples",
        "prompt_type",
        "prompt_templates",
        "rerank_strategy",
        "rerank_k",
        "device",
        *METRIC_COLUMNS,
        "total_failures",
        *FAILURE_TAGS,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    headers = [
        "Method",
        "Split",
        "Max Examples",
        "Prompt",
        "Rerank",
        "I2T R@1",
        "I2T R@5",
        "I2T R@10",
        "T2I R@1",
        "T2I R@5",
        "T2I R@10",
    ]
    lines = [
        "# Results Summary",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["display_name"]),
                    str(row["split"]),
                    str(row["max_examples"]),
                    str(row["prompt_type"]),
                    str(row["rerank_strategy"]),
                    f"{float(row['image_to_text_r1']):.3f}",
                    f"{float(row['image_to_text_r5']):.3f}",
                    f"{float(row['image_to_text_r10']):.3f}",
                    f"{float(row['text_to_image_r1']):.3f}",
                    f"{float(row['text_to_image_r5']):.3f}",
                    f"{float(row['text_to_image_r10']):.3f}",
                ]
            )
            + "|"
        )

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def annotate_bars(ax: plt.Axes, bars, offset: float) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_recall_figure(
    rows: list[dict[str, object]],
    output_path: Path,
    metric_prefix: str,
    title: str,
) -> None:
    labels = [str(row["display_name"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.25

    r1 = [float(row[f"{metric_prefix}_r1"]) for row in rows]
    r5 = [float(row[f"{metric_prefix}_r5"]) for row in rows]
    r10 = [float(row[f"{metric_prefix}_r10"]) for row in rows]
    lower_bound = max(0.0, min(r1 + r5 + r10) - 0.05)
    upper_bound = min(1.0, max(r1 + r5 + r10) + 0.02)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_r1 = ax.bar(x - width, r1, width, label="Recall@1", color="#355070")
    bars_r5 = ax.bar(x, r5, width, label="Recall@5", color="#6D597A")
    bars_r10 = ax.bar(x + width, r10, width, label="Recall@10", color="#B56576")

    ax.set_title(title)
    ax.set_ylabel("Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(lower_bound, upper_bound)
    ax.legend()
    annotate_bars(ax, bars_r1, offset=0.002)
    annotate_bars(ax, bars_r5, offset=0.002)
    annotate_bars(ax, bars_r10, offset=0.002)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_failure_categories(rows: list[dict[str, object]], output_path: Path) -> None:
    labels = [str(row["display_name"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.13

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, tag in enumerate(FAILURE_TAGS):
        heights = [float(row[tag]) for row in rows]
        bars = ax.bar(
            x + (idx - (len(FAILURE_TAGS) - 1) / 2) * width,
            heights,
            width,
            label=tag.replace("_", " "),
            color=BAR_COLORS[idx % len(BAR_COLORS)],
        )
        if max(heights) > 0:
            annotate_bars(ax, bars, offset=max(heights) * 0.01)

    ax.set_title("Failure Category Counts by Method")
    ax.set_ylabel("Number of Tagged Failures")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    outputs_dir = Path("outputs")
    rows = load_experiment_rows(outputs_dir)
    if not rows:
        raise FileNotFoundError("No experiment directories with metrics.json and config.json were found under outputs/.")

    tables_dir = outputs_dir / "final_tables"
    figures_dir = outputs_dir / "final_figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_summary_csv(rows, tables_dir / "results_summary.csv")
    write_summary_markdown(rows, tables_dir / "results_summary.md")
    plot_recall_figure(
        rows=rows,
        output_path=figures_dir / "image_to_text_recall.png",
        metric_prefix="image_to_text",
        title="Image-to-Text Recall Comparison",
    )
    plot_recall_figure(
        rows=rows,
        output_path=figures_dir / "text_to_image_recall.png",
        metric_prefix="text_to_image",
        title="Text-to-Image Recall Comparison",
    )
    plot_failure_categories(rows, figures_dir / "failure_category_counts.png")

    print(f"Wrote summaries to {tables_dir}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
