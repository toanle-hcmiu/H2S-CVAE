"""
Training entry-point for the H2S-CVAE model.

Usage
-----
    python train.py                         # defaults
    python train.py --epochs 200 --lr 5e-4  # override hyper-params
    python train.py --resume                # resume from latest checkpoint
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2s_cvae.config import get_default_config
from h2s_cvae.data.dataset import build_dataloaders
from h2s_cvae.models.cvae import HeadToSkullCVAE
from h2s_cvae.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train the H2S-CVAE model.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--kl-target", type=float, default=None)
    parser.add_argument("--kl-anneal", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    cfg = get_default_config()

    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.latent_dim is not None:
        cfg.model.latent_dim = args.latent_dim
    if args.resolution is not None:
        cfg.data.voxel_resolution = args.resolution
        cfg.paths.resolve_voxel_dir(args.resolution)
    if args.kl_target is not None:
        cfg.training.kl_weight_target = args.kl_target
    if args.kl_anneal is not None:
        cfg.training.kl_anneal_epochs = args.kl_anneal
    if args.device is not None:
        cfg.training.device = args.device
    if args.no_augment:
        cfg.data.augment = False
    if args.seed is not None:
        cfg.training.seed = args.seed

    # Reproducibility
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    print("Building data loaders ...")
    train_loader, val_loader = build_dataloaders(cfg, train=True, test=True)
    print(f"  Training samples:   {len(train_loader.dataset)}")
    if val_loader:
        print(f"  Validation samples: {len(val_loader.dataset)}")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = HeadToSkullCVAE(
        latent_dim=cfg.model.latent_dim,
        encoder_channels=cfg.model.encoder_channels,
        decoder_channels=cfg.model.decoder_channels,
        condition_channels=cfg.model.condition_channels,
        use_learned_prior=cfg.model.use_learned_prior,
        use_skip_connections=cfg.model.use_skip_connections,
        spatial_size=cfg.data.voxel_resolution,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: HeadToSkullCVAE")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Latent dim:           {cfg.model.latent_dim}")
    print(f"  Voxel resolution:     {cfg.data.voxel_resolution}^3")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    resume_path = None
    if args.resume:
        resume_path = os.path.join(cfg.paths.checkpoint_dir, "checkpoint_latest.pt")

    trainer.train(resume_from=resume_path)


if __name__ == "__main__":
    main()
