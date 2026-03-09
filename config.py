"""
Hyperparameter configuration for LayerSkip-LLaDA training.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LayerSkipConfig:
    """New hyperparameters introduced by the LayerSkip-LLaDA recipe."""

    exit_layers: List[int] = field(default_factory=lambda: [4, 8, 16, 24, 32])
    eps_scale: float = 0.2
    c_cap: float = 20.0
    alpha: float = 0.1
    p_max: float = 0.1
    beta: float = 0.3
    rotation_period: int = 16


@dataclass
class CurriculumConfig:
    """Configuration for the multi-phase training curriculum."""

    mode: str = "gradual"  # "rotational", "gradual", or "combined"
    total_tokens: float = 2.3e12
    phase_fractions: List[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.20, 0.20, 0.30]
    )
    timestep_thresholds: List[Optional[float]] = field(
        default_factory=lambda: [None, 0.7, 0.4, 0.2, None]
    )
    phase_exit_layers: List[List[int]] = field(
        default_factory=lambda: [
            [32],
            [24, 32],
            [16, 24, 32],
            [8, 16, 24, 32],
            [4, 8, 16, 24, 32],
        ]
    )


@dataclass
class TrainingConfig:
    """Full training configuration."""

    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"

    mode: str = "pretrain"  # "pretrain" or "sft"
    dataset_name: str = "monology/pile-uncopyrighted"
    dataset_split: str = "train"
    sft_dataset_name: Optional[str] = None

    output_dir: str = "./checkpoints/skipllada"

    # Inherited from LLaDA
    learning_rate: float = 4e-4
    sft_learning_rate: float = 2.5e-5
    weight_decay: float = 0.1
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_ratio: float = 0.01
    lr_scheduler_type: str = "cosine"

    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    mask_token_id: int = 126336
    mask_eps: float = 1e-3
    random_length_fraction: float = 0.01

    # LayerSkip specific
    layerskip: LayerSkipConfig = field(default_factory=LayerSkipConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    seed: int = 42

    @property
    def effective_lr(self) -> float:
        return self.sft_learning_rate if self.mode == "sft" else self.learning_rate
