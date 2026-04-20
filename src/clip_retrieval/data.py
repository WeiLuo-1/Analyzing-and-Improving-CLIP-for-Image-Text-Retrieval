from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass
class RetrievalExample:
    image_id: str
    image_path: str
    captions: list[str]


def _normalize_image_path(image_path: str, image_root: str | None) -> str:
    path = Path(image_path)
    if path.is_absolute() or image_root is None:
        return str(path)
    return str(Path(image_root) / path)


def load_local_jsonl(data_path: str, image_root: str | None = None) -> list[RetrievalExample]:
    examples: list[RetrievalExample] = []
    with open(data_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            captions = row.get("captions", [])
            if not isinstance(captions, list) or not captions:
                raise ValueError(f"Line {line_number} is missing a non-empty captions list.")
            examples.append(
                RetrievalExample(
                    image_id=str(row["image_id"]),
                    image_path=_normalize_image_path(str(row["image_path"]), image_root),
                    captions=[str(caption) for caption in captions],
                )
            )
    return examples


def load_hf_flickr30k(split: str = "test", max_examples: int | None = None) -> list[RetrievalExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the datasets package to load Flickr30k from Hugging Face.") from exc

    # `nlphuji/flickr30k` relied on a dataset loading script, which newer
    # versions of `datasets` no longer support. Use a Parquet-backed mirror.
    dataset = load_dataset("lmms-lab/flickr30k", split=split)
    examples: list[RetrievalExample] = []
    for idx, row in enumerate(dataset):
        if max_examples is not None and idx >= max_examples:
            break
        image = row["image"]
        image_id = str(row.get("img_id", idx))
        image_path = _persist_hf_image(image, Path("data/.cache/flickr30k"), f"{split}_{image_id}_{idx:06d}.jpg")
        captions = [str(caption) for caption in row["caption"]]
        examples.append(RetrievalExample(image_id=image_id, image_path=image_path, captions=captions))
    return examples


def _persist_hf_image(image: Image.Image, cache_dir: Path, file_name: str) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / file_name
    if not path.exists():
        image.save(path)
    return str(path)


def load_examples(
    dataset: str,
    data_path: str | None,
    split: str,
    image_root: str | None = None,
    max_examples: int | None = None,
) -> list[RetrievalExample]:
    if dataset == "local_jsonl":
        if data_path is None:
            raise ValueError("--data-path is required when dataset=local_jsonl")
        examples = load_local_jsonl(data_path, image_root=image_root)
        return examples if max_examples is None else examples[:max_examples]
    if dataset == "flickr30k_hf":
        return load_hf_flickr30k(split=split, max_examples=max_examples)
    raise ValueError(f"Unsupported dataset '{dataset}'.")


def validate_examples(examples: Iterable[RetrievalExample]) -> None:
    for example in examples:
        if not Path(example.image_path).exists():
            raise FileNotFoundError(f"Missing image file: {example.image_path}")
        if not example.captions:
            raise ValueError(f"Image {example.image_id} has no captions.")
