# Analyzing and Improving CLIP for Image-Text Retrieval

This repository turns the proposal into a runnable project scaffold for CS543.

The current code supports:

- A baseline CLIP retrieval pipeline
- Image-to-text and text-to-image evaluation with `Recall@K`
- Prompt-engineering experiments through caption templates
- Lightweight reranking hooks
- Failure-case export with simple heuristic tags

## Project Goal

We want to measure how a pretrained CLIP model behaves on image-text retrieval, identify where it fails, and test small improvements that do not require full retraining.

## Recommended Dataset Format

The easiest way to start is a local JSONL file where each row looks like:

```json
{"image_id": "1000092795", "image_path": "data/flickr30k/1000092795.jpg", "captions": ["Two dogs run through snow.", "A pair of dogs playing in the snow."]}
```

Each example needs:

- `image_id`: unique string
- `image_path`: path to an image file
- `captions`: list of one or more captions for that image

## Install

```bash
pip install -r requirements.txt
```

## Run A Baseline Experiment

```bash
python run_experiment.py ^
  --dataset local_jsonl ^
  --data-path data/flickr30k/val.jsonl ^
  --experiment-name flickr30k_baseline
```

## Run With Prompt Engineering

```bash
python run_experiment.py ^
  --dataset local_jsonl ^
  --data-path data/flickr30k/val.jsonl ^
  --experiment-name flickr30k_prompt ^
  --prompt-template "a photo of {}" ^
  --prompt-template "an image of {}"
```

## Run With Reranking

```bash
python run_experiment.py ^
  --dataset local_jsonl ^
  --data-path data/flickr30k/val.jsonl ^
  --experiment-name flickr30k_rerank ^
  --rerank topk_consensus ^
  --rerank-k 25
```

## Outputs

Each run writes a directory under `outputs/<experiment-name>/` with:

- `metrics.json`: Recall metrics
- `failure_cases.jsonl`: hard examples with heuristic tags
- `config.json`: run configuration

## Suggested Project Workflow

1. Start with a small split of Flickr30k to verify the pipeline.
2. Run the baseline and record `Recall@1`, `Recall@5`, and `Recall@10`.
3. Inspect exported failure cases and group them into categories from the proposal:
   ambiguous queries, multiple objects, unusual scenes.
4. Compare prompt templates.
5. Try reranking and report whether it improves or hurts retrieval.

## Notes

- The code defaults to `openai/clip-vit-base-patch32`, which matches the proposal's `ViT-B/32` baseline closely.
- The `flickr30k_hf` option uses a Parquet-backed Flickr30k mirror on Hugging Face so it works with current `datasets` releases.
- The scaffold is intentionally lightweight so we can extend it quickly once your dataset split is ready.
