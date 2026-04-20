from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clip_retrieval.analysis import export_failure_cases
from clip_retrieval.config import ExperimentConfig
from clip_retrieval.data import RetrievalExample, load_examples, validate_examples
from clip_retrieval.experiments import aggregate_prompt_embeddings, apply_prompt_templates, rerank_similarity
from clip_retrieval.metrics import compute_retrieval_metrics
from clip_retrieval.modeling import ClipRetriever


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CLIP image-text retrieval experiments.")
    parser.add_argument("--dataset", choices=["local_jsonl", "flickr30k_hf"], default="local_jsonl")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--experiment-name", default="default_run")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-template", action="append", default=[])
    parser.add_argument("--rerank", default="none", choices=["none", "topk_consensus"])
    parser.add_argument("--rerank-k", type=int, default=25)
    return parser


def flatten_examples(examples: list[RetrievalExample]) -> tuple[list[str], list[str], np.ndarray, list[np.ndarray]]:
    image_ids: list[str] = []
    image_paths: list[str] = []
    captions: list[str] = []
    caption_to_image: list[int] = []
    image_to_caption_sets: list[np.ndarray] = []

    caption_cursor = 0
    for image_idx, example in enumerate(examples):
        image_ids.append(example.image_id)
        image_paths.append(example.image_path)
        captions.extend(example.captions)
        caption_indices = np.arange(caption_cursor, caption_cursor + len(example.captions))
        image_to_caption_sets.append(caption_indices)
        caption_to_image.extend([image_idx] * len(example.captions))
        caption_cursor += len(example.captions)
    return image_ids, image_paths, captions, np.asarray(caption_to_image), image_to_caption_sets


def run_experiment(config: ExperimentConfig) -> dict:
    output_dir = config.ensure_output_dir()
    examples = load_examples(config.dataset, config.data_path, config.split, image_root=config.image_root)
    validate_examples(examples)

    image_ids, image_paths, captions, caption_to_image, image_to_caption_sets = flatten_examples(examples)
    prompted_captions = apply_prompt_templates(captions, config.prompt_templates)

    retriever = ClipRetriever(model_name=config.model_name, device=config.device)
    image_embeddings = retriever.encode_images(image_paths, batch_size=config.batch_size)
    text_embeddings = retriever.encode_texts(prompted_captions, batch_size=config.batch_size)

    num_templates = max(1, len(config.prompt_templates))
    text_embeddings = aggregate_prompt_embeddings(text_embeddings, num_templates=num_templates)

    similarity = image_embeddings @ text_embeddings.T
    similarity = rerank_similarity(
        similarity=similarity,
        image_to_caption_sets=image_to_caption_sets,
        strategy=config.rerank_strategy,
        top_k=config.rerank_k,
    )
    metrics = compute_retrieval_metrics(similarity, caption_to_image, image_to_caption_sets)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    export_failure_cases(
        output_path=output_dir / "failure_cases.jsonl",
        similarity=similarity,
        image_ids=image_ids,
        captions=captions,
        caption_to_image=caption_to_image,
    )

    return {
        "metrics": metrics.to_dict(),
        "num_images": len(image_ids),
        "num_captions": len(captions),
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = str(Path("outputs") / args.experiment_name)
    config = ExperimentConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        split=args.split,
        model_name=args.model_name,
        batch_size=args.batch_size,
        image_root=args.image_root,
        prompt_templates=args.prompt_template,
        rerank_strategy=args.rerank,
        rerank_k=args.rerank_k,
        output_dir=output_dir,
        device=args.device,
    )
    result = run_experiment(config)
    print(json.dumps(result, indent=2))
