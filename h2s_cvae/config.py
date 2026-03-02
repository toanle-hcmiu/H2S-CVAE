"""
Centralized configuration for the H2S-CVAE project.

All hyperparameters, paths, and settings are defined here as dataclass fields
with sensible defaults. Override by modifying this file or passing CLI arguments
to the training / preprocessing entry points.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathConfig:
    """File-system paths used throughout the project."""

    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = ""
    mesh_dir: str = ""
    voxel_dir: str = ""
    cross_val_dir: str = ""
    subject_id_file: str = ""
    training_id_file: str = ""
    testing_id_file: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""
    output_dir: str = ""

    def __post_init__(self):
        self.data_dir = self.data_dir or os.path.join(self.project_root, "Data")
        self.mesh_dir = self.mesh_dir or os.path.join(self.data_dir, "HeadAndSkullShapes")
        self.cross_val_dir = self.cross_val_dir or os.path.join(self.data_dir, "CrossValidation")
        self.subject_id_file = self.subject_id_file or os.path.join(
            self.data_dir, "PostProcessedSubjectIDs_PrevPapers.txt"
        )
        self.training_id_file = self.training_id_file or os.path.join(
            self.cross_val_dir, "TrainingIDs.txt"
        )
        self.testing_id_file = self.testing_id_file or os.path.join(
            self.cross_val_dir, "TestingIDs.txt"
        )
        self.checkpoint_dir = self.checkpoint_dir or os.path.join(self.project_root, "checkpoints")
        self.log_dir = self.log_dir or os.path.join(self.project_root, "logs")
        self.output_dir = self.output_dir or os.path.join(self.project_root, "outputs")

    def resolve_voxel_dir(self, resolution: int) -> str:
        """Return (and lazily set) the voxel cache directory for a given resolution."""
        self.voxel_dir = os.path.join(self.data_dir, "Voxelized", str(resolution))
        return self.voxel_dir


@dataclass
class DataConfig:
    """Data representation and augmentation settings."""

    voxel_resolution: int = 64
    representation: str = "binary"  # "binary" | "sdf"
    padding_ratio: float = 0.05  # Extra padding around the bounding box (fraction)
    augment: bool = True
    aug_flip_axes: bool = True  # Random axis flips during training
    aug_translate_voxels: int = 2  # Max random translation in voxels
    num_workers: int = 4


@dataclass
class ModelConfig:
    """CVAE architecture hyper-parameters."""

    latent_dim: int = 256
    encoder_channels: list = field(default_factory=lambda: [32, 64, 128, 256, 512])
    decoder_channels: list = field(default_factory=lambda: [512, 256, 128, 64, 32])
    condition_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    use_learned_prior: bool = True  # Head-conditioned prior p(z|H)
    use_skip_connections: bool = True  # Inject multi-scale head features into decoder
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Training loop hyper-parameters."""

    batch_size: int = 4
    num_epochs: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    # KL annealing
    kl_weight_target: float = 0.001  # β target
    kl_anneal_epochs: int = 100  # Linearly anneal β from 0 → target over this many epochs
    # LR scheduler
    lr_scheduler: str = "cosine"  # "cosine" | "plateau"
    lr_min: float = 1e-6
    plateau_patience: int = 20
    plateau_factor: float = 0.5
    # Checkpointing
    save_every_n_epochs: int = 50
    validate_every_n_epochs: int = 1
    # Reproducibility
    seed: int = 42
    # Device
    device: str = "auto"  # "auto" | "cuda" | "cpu"


@dataclass
class EvalConfig:
    """Evaluation settings."""

    compute_hausdorff: bool = True
    compute_mean_surface_distance: bool = True
    compute_dice: bool = True
    marching_cubes_level: float = 0.5  # Iso-surface threshold for voxel→mesh
    save_predicted_meshes: bool = True
    num_samples_to_visualize: int = 5


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        # Ensure voxel dir is resolved once data config is known
        self.paths.resolve_voxel_dir(self.data.voxel_resolution)


def get_default_config() -> Config:
    """Return a default Config instance."""
    return Config()
