"""
Training loop for the H2S-CVAE model.

Handles:
* Forward / backward passes with the combined CVAE loss
* KL annealing schedule
* Learning-rate scheduling (cosine or reduce-on-plateau)
* Gradient clipping
* Checkpointing (periodic + best validation loss)
* TensorBoard logging
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from h2s_cvae.config import Config
from h2s_cvae.models.cvae import HeadToSkullCVAE
from h2s_cvae.training.losses import cvae_loss, kl_weight_schedule


class Trainer:
    """
    Manages the full training lifecycle of the :class:`HeadToSkullCVAE`.

    Parameters
    ----------
    model : HeadToSkullCVAE
    cfg : Config
    train_loader : DataLoader
    val_loader : DataLoader, optional
    """

    def __init__(
        self,
        model: HeadToSkullCVAE,
        cfg: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.cfg = cfg
        self.tc = cfg.training  # shorthand

        # Device
        if self.tc.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.tc.device)
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimiser
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.tc.learning_rate,
            weight_decay=self.tc.weight_decay,
        )

        # LR scheduler
        if self.tc.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.tc.num_epochs, eta_min=self.tc.lr_min,
            )
        elif self.tc.lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.tc.plateau_factor,
                patience=self.tc.plateau_patience,
                min_lr=self.tc.lr_min,
            )
        else:
            self.scheduler = None

        # Logging
        os.makedirs(cfg.paths.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=cfg.paths.log_dir)

        # Checkpoints
        os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
        self.best_val_loss = float("inf")
        self.start_epoch = 0

    # ------------------------------------------------------------------ #
    #  Checkpoint save / load                                              #
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, epoch: int, tag: str = "latest") -> str:
        path = os.path.join(self.cfg.paths.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {path} at epoch {self.start_epoch}")

    # ------------------------------------------------------------------ #
    #  Single training epoch                                               #
    # ------------------------------------------------------------------ #
    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        kl_w = kl_weight_schedule(epoch, self.tc.kl_weight_target, self.tc.kl_anneal_epochs)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for head, skull in self.train_loader:
            head = head.to(self.device)
            skull = skull.to(self.device)

            out = self.model(head, skull)
            losses = cvae_loss(
                recon=out["recon"],
                target=skull,
                mu_q=out["mu_q"],
                logvar_q=out["logvar_q"],
                mu_p=out["mu_p"],
                logvar_p=out["logvar_p"],
                kl_weight=kl_w,
                representation=self.cfg.data.representation,
            )

            self.optimizer.zero_grad()
            losses["loss"].backward()
            if self.tc.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip_norm)
            self.optimizer.step()

            total_loss += losses["loss"].item()
            total_recon += losses["recon_loss"].item()
            total_kl += losses["kl_loss"].item()
            n_batches += 1

        avg = {
            "loss": total_loss / max(n_batches, 1),
            "recon_loss": total_recon / max(n_batches, 1),
            "kl_loss": total_kl / max(n_batches, 1),
            "kl_weight": kl_w,
        }
        return avg

    # ------------------------------------------------------------------ #
    #  Validation epoch                                                    #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.model.eval()
        kl_w = kl_weight_schedule(epoch, self.tc.kl_weight_target, self.tc.kl_anneal_epochs)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for head, skull in self.val_loader:
            head = head.to(self.device)
            skull = skull.to(self.device)

            out = self.model(head, skull)
            losses = cvae_loss(
                recon=out["recon"],
                target=skull,
                mu_q=out["mu_q"],
                logvar_q=out["logvar_q"],
                mu_p=out["mu_p"],
                logvar_p=out["logvar_p"],
                kl_weight=kl_w,
                representation=self.cfg.data.representation,
            )

            total_loss += losses["loss"].item()
            total_recon += losses["recon_loss"].item()
            total_kl += losses["kl_loss"].item()
            n_batches += 1

        avg = {
            "loss": total_loss / max(n_batches, 1),
            "recon_loss": total_recon / max(n_batches, 1),
            "kl_loss": total_kl / max(n_batches, 1),
        }
        return avg

    # ------------------------------------------------------------------ #
    #  Main training loop                                                  #
    # ------------------------------------------------------------------ #
    def train(self, resume_from: Optional[str] = None) -> None:
        if resume_from and os.path.isfile(resume_from):
            self.load_checkpoint(resume_from)

        print(f"\nStarting training for {self.tc.num_epochs} epochs ...")
        print(f"  Train batches/epoch: {len(self.train_loader)}")
        if self.val_loader:
            print(f"  Val batches/epoch:   {len(self.val_loader)}")
        print()

        for epoch in range(self.start_epoch, self.tc.num_epochs):
            t0 = time.time()

            # --- Train ---
            train_metrics = self._train_one_epoch(epoch)

            # --- Validate ---
            val_metrics = None
            if self.val_loader and (epoch % self.tc.validate_every_n_epochs == 0):
                val_metrics = self._validate(epoch)

            # --- LR scheduler step ---
            if isinstance(self.scheduler, ReduceLROnPlateau):
                metric_for_scheduler = val_metrics["loss"] if val_metrics else train_metrics["loss"]
                self.scheduler.step(metric_for_scheduler)
            elif self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # --- Logging ---
            self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("train/recon_loss", train_metrics["recon_loss"], epoch)
            self.writer.add_scalar("train/kl_loss", train_metrics["kl_loss"], epoch)
            self.writer.add_scalar("train/kl_weight", train_metrics["kl_weight"], epoch)
            self.writer.add_scalar("train/lr", lr, epoch)

            log_str = (
                f"Epoch {epoch:>4d}/{self.tc.num_epochs} | "
                f"Train loss: {train_metrics['loss']:.5f} "
                f"(recon: {train_metrics['recon_loss']:.5f}, "
                f"kl: {train_metrics['kl_loss']:.4f}, "
                f"β: {train_metrics['kl_weight']:.5f})"
            )
            if val_metrics:
                self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("val/recon_loss", val_metrics["recon_loss"], epoch)
                self.writer.add_scalar("val/kl_loss", val_metrics["kl_loss"], epoch)
                log_str += (
                    f" | Val loss: {val_metrics['loss']:.5f} "
                    f"(recon: {val_metrics['recon_loss']:.5f})"
                )

                # Best model checkpoint
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(epoch, tag="best")

            log_str += f" | LR: {lr:.2e} | {elapsed:.1f}s"
            print(log_str)

            # --- Periodic checkpoint ---
            if (epoch + 1) % self.tc.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, tag=f"epoch_{epoch}")

            # Always save 'latest' for easy resume
            self.save_checkpoint(epoch, tag="latest")

        self.writer.close()
        print("\nTraining complete.")
