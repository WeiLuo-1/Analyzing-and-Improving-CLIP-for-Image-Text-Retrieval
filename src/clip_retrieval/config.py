from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    dataset: str
    data_path: str | None
    split: str
    model_name: str = "openai/clip-vit-base-patch32"
    batch_size: int = 32
    image_root: str | None = None
    prompt_templates: list[str] = field(default_factory=list)
    rerank_strategy: str = "none"
    rerank_k: int = 25
    output_dir: str = "outputs/default_run"
    device: str = "cuda"

    def to_dict(self) -> dict:
        return asdict(self)

    def ensure_output_dir(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
