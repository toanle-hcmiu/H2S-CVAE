"""
Visualisation helpers for the H2S-CVAE project.

* Side-by-side axial / coronal / sagittal slices of head, ground-truth skull,
  and predicted skull.
* 3D voxel rendering via matplotlib.
* Latent-space scatter plots (t-SNE / UMAP).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# =========================================================================== #
#  Slice comparison                                                            #
# =========================================================================== #

def plot_slices(
    head: np.ndarray,
    gt_skull: np.ndarray,
    pred_skull: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot mid-plane slices of head, ground-truth skull, and predicted skull.

    Parameters
    ----------
    head, gt_skull, pred_skull : (D, H, W) or (1, D, H, W) arrays
    axis : int
        Spatial axis to slice (0=axial, 1=coronal, 2=sagittal).
    """
    # Remove channel dim if present
    if head.ndim == 4:
        head = head.squeeze(0)
    if gt_skull.ndim == 4:
        gt_skull = gt_skull.squeeze(0)
    if pred_skull.ndim == 4:
        pred_skull = pred_skull.squeeze(0)

    if slice_idx is None:
        slice_idx = head.shape[axis] // 2

    slices = {
        "Head": np.take(head, slice_idx, axis=axis),
        "GT Skull": np.take(gt_skull, slice_idx, axis=axis),
        "Predicted Skull": np.take(pred_skull, slice_idx, axis=axis),
    }

    axis_names = {0: "Axial", 1: "Coronal", 2: "Sagittal"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, slc) in zip(axes, slices.items()):
        ax.imshow(slc.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis("off")

    suptitle = f"{axis_names.get(axis, 'Slice')} at index {slice_idx}"
    if title:
        suptitle = f"{title} — {suptitle}"
    fig.suptitle(suptitle)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_all_axes(
    head: np.ndarray,
    gt_skull: np.ndarray,
    pred_skull: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot slices along all three axes in a single figure."""
    if head.ndim == 4:
        head = head.squeeze(0)
    if gt_skull.ndim == 4:
        gt_skull = gt_skull.squeeze(0)
    if pred_skull.ndim == 4:
        pred_skull = pred_skull.squeeze(0)

    axis_names = {0: "Axial", 1: "Coronal", 2: "Sagittal"}
    volumes = {"Head": head, "GT Skull": gt_skull, "Predicted Skull": pred_skull}

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for row, (axis_idx, axis_name) in enumerate(axis_names.items()):
        slice_idx = head.shape[axis_idx] // 2
        for col, (vol_name, vol) in enumerate(volumes.items()):
            slc = np.take(vol, slice_idx, axis=axis_idx)
            axes[row, col].imshow(slc.T, cmap="gray", origin="lower", vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title(vol_name, fontsize=13)
            if col == 0:
                axes[row, col].set_ylabel(axis_name, fontsize=13)
            axes[row, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# =========================================================================== #
#  3D voxel rendering                                                          #
# =========================================================================== #

def plot_voxels_3d(
    volume: np.ndarray,
    threshold: float = 0.5,
    title: str = "",
    color: str = "skyblue",
    save_path: Optional[str] = None,
) -> None:
    """
    Simple 3D voxel rendering using ``matplotlib.voxels()``.

    Note: this is memory-intensive for resolutions above ~64^3.
    """
    if volume.ndim == 4:
        volume = volume.squeeze(0)

    filled = volume >= threshold

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(filled, facecolors=color, edgecolor="k", linewidth=0.1, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# =========================================================================== #
#  Training curve plotting                                                     #
# =========================================================================== #

def plot_training_curves(
    train_losses: list[float],
    val_losses: Optional[list[float]] = None,
    title: str = "Training Loss",
    save_path: Optional[str] = None,
) -> None:
    """Plot training (and optionally validation) loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train")
    if val_losses:
        ax.plot(val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# =========================================================================== #
#  Latent space scatter                                                        #
# =========================================================================== #

def plot_latent_space(
    latent_codes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    title: str = "Latent Space",
    save_path: Optional[str] = None,
) -> None:
    """
    2D scatter plot of latent codes via t-SNE or PCA.

    Parameters
    ----------
    latent_codes : (N, latent_dim) array
    labels : (N,) optional label array for colouring points
    method : "tsne" | "pca"
    """
    if method == "tsne":
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_codes) - 1)).fit_transform(latent_codes)
    else:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(latent_codes)

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="Set2", s=20, alpha=0.7)
    if labels is not None:
        fig.colorbar(scatter, ax=ax, label="Label")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
