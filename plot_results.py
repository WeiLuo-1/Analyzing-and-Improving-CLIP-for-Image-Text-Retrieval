from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def annotate_bars(bars, offset: float) -> None:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> None:
    methods = ["Baseline", "Photo Image", "Detailed Scene", "Rerank"]
    recall_at_1 = [0.791, 0.784, 0.777, 0.791]
    recall_at_5 = [0.948, 0.950, 0.953, 0.952]
    recall_at_10 = [0.976, 0.978, 0.977, 0.980]

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, recall_at_1, color=["#4C78A8", "#72B7B2", "#F58518", "#54A24B"])
    plt.title("Text-to-Image Recall@1 Comparison")
    plt.xlabel("Method")
    plt.ylabel("Recall@1")
    plt.ylim(0.75, 0.81)
    annotate_bars(bars, offset=0.001)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_at_1.png", dpi=300)
    plt.close()

    x = np.arange(len(methods))
    width = 0.36

    plt.figure(figsize=(9, 5))
    bars_r5 = plt.bar(x - width / 2, recall_at_5, width, label="Recall@5", color="#4C78A8")
    bars_r10 = plt.bar(x + width / 2, recall_at_10, width, label="Recall@10", color="#F58518")
    plt.title("Text-to-Image Recall@5 and Recall@10 Comparison")
    plt.xlabel("Method")
    plt.ylabel("Recall")
    plt.xticks(x, methods)
    plt.ylim(0.94, 0.985)
    plt.legend()
    annotate_bars(bars_r5, offset=0.0005)
    annotate_bars(bars_r10, offset=0.0005)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_at_5_10.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
