"""
PyTorch Dataset for paired head / skull voxel volumes.

Loads pre-voxelized .npy files produced by ``preprocess.py`` and returns
``(head_volume, skull_volume)`` tensors of shape ``(1, D, H, W)``.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class H2SDataset(Dataset):
    """
    Dataset of paired head and skull voxel volumes.

    Parameters
    ----------
    subject_ids : list of str
        Subject identifiers (e.g. ``["HN-CHUM-001", ...]``).
    voxel_dir : str
        Directory containing ``{subjectID}-FullHead.npy`` and
        ``{subjectID}-FullSkull.npy`` files.
    augment : bool
        Whether to apply on-the-fly data augmentation (training only).
    flip_axes : bool
        If *augment* is True, randomly flip along spatial axes.
    max_translate : int
        If *augment* is True, maximum random translation in voxels.
    """

    def __init__(
        self,
        subject_ids: List[str],
        voxel_dir: str,
        augment: bool = False,
        flip_axes: bool = True,
        max_translate: int = 2,
    ):
        super().__init__()
        self.subject_ids = subject_ids
        self.voxel_dir = voxel_dir
        self.augment = augment
        self.flip_axes = flip_axes
        self.max_translate = max_translate

        # Validate that all files exist
        missing = []
        for sid in self.subject_ids:
            for suffix in ("-FullHead.npy", "-FullSkull.npy"):
                fp = os.path.join(self.voxel_dir, f"{sid}{suffix}")
                if not os.path.isfile(fp):
                    missing.append(fp)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} voxel files are missing. First few:\n"
                + "\n".join(missing[:5])
            )

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sid = self.subject_ids[idx]
        head_vol = np.load(os.path.join(self.voxel_dir, f"{sid}-FullHead.npy"))
        skull_vol = np.load(os.path.join(self.voxel_dir, f"{sid}-FullSkull.npy"))

        # Apply identical augmentations to both volumes
        if self.augment:
            head_vol, skull_vol = self._augment(head_vol, skull_vol)

        head_tensor = torch.from_numpy(head_vol.copy()).float()
        skull_tensor = torch.from_numpy(skull_vol.copy()).float()
        return head_tensor, skull_tensor

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------
    def _augment(
        self, head: np.ndarray, skull: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random spatial augmentations identically to both volumes."""
        # Random axis flips (along spatial dims 1, 2, 3 — dim 0 is channel)
        if self.flip_axes:
            for axis in (1, 2, 3):
                if np.random.rand() > 0.5:
                    head = np.flip(head, axis=axis)
                    skull = np.flip(skull, axis=axis)

        # Random translation (shift + zero-pad)
        if self.max_translate > 0:
            shifts = np.random.randint(-self.max_translate, self.max_translate + 1, size=3)
            head = self._shift_volume(head, shifts)
            skull = self._shift_volume(skull, shifts)

        return head, skull

    @staticmethod
    def _shift_volume(vol: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        """Shift a (C, D, H, W) volume by integer voxel offsets, zero-padding."""
        shifted = np.zeros_like(vol)
        sd, sh, sw = shifts
        D, H, W = vol.shape[1], vol.shape[2], vol.shape[3]

        # Source and destination slicing ranges
        src_d = slice(max(0, -sd), min(D, D - sd))
        dst_d = slice(max(0, sd), min(D, D + sd))
        src_h = slice(max(0, -sh), min(H, H - sh))
        dst_h = slice(max(0, sh), min(H, H + sh))
        src_w = slice(max(0, -sw), min(W, W - sw))
        dst_w = slice(max(0, sw), min(W, W + sw))

        shifted[:, dst_d, dst_h, dst_w] = vol[:, src_d, src_h, src_w]
        return shifted


def read_subject_ids(filepath: str) -> List[str]:
    """Read a newline-delimited text file of subject IDs."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_dataloaders(
    cfg,
    *,
    train: bool = True,
    test: bool = True,
):
    """
    Convenience factory that returns ``(train_loader, test_loader)`` based on
    a :class:`~h2s_cvae.config.Config` instance.

    Returns ``None`` for any loader that was not requested.
    """
    from torch.utils.data import DataLoader

    voxel_dir = cfg.paths.resolve_voxel_dir(cfg.data.voxel_resolution)

    train_loader = None
    test_loader = None

    if train:
        train_ids = read_subject_ids(cfg.paths.training_id_file)
        train_ds = H2SDataset(
            subject_ids=train_ids,
            voxel_dir=voxel_dir,
            augment=cfg.data.augment,
            flip_axes=cfg.data.aug_flip_axes,
            max_translate=cfg.data.aug_translate_voxels,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    if test:
        test_ids = read_subject_ids(cfg.paths.testing_id_file)
        test_ds = H2SDataset(
            subject_ids=test_ids,
            voxel_dir=voxel_dir,
            augment=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )

    return train_loader, test_loader
