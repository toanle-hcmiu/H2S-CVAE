"""
Evaluation metrics for skull prediction quality.

Supports both **volumetric** metrics (Dice on voxel grids) and **surface**
metrics (Hausdorff distance, mean surface distance) by converting predicted
voxels to meshes via marching cubes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


# =========================================================================== #
#  Result container                                                            #
# =========================================================================== #

@dataclass
class MetricResult:
    """Container for a single subject's evaluation metrics."""

    subject_id: str = ""
    dice: float = 0.0
    # Surface distance metrics (mm if spacing is provided; otherwise voxel units)
    hausdorff: float = 0.0  # symmetric max surface distance
    mean_surface_distance: float = 0.0  # symmetric mean surface distance (MSD)
    rms_surface_distance: float = 0.0  # symmetric RMS surface distance
    max_surface_distance: float = 0.0  # alias of hausdorff (kept for clarity)

    # Volume metrics
    volume_pred: float = 0.0  # occupied volume (voxels^3 or mm^3 depending on downstream use)
    volume_gt: float = 0.0
    volume_deviation_percent: float = 0.0  # 100 * |Vp - Vgt| / Vgt

    # Optionally store per-vertex distances for further analysis
    pred_to_gt_distances: Optional[np.ndarray] = None
    gt_to_pred_distances: Optional[np.ndarray] = None


# =========================================================================== #
#  Volumetric metrics                                                          #
# =========================================================================== #

def dice_coefficient(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    Dice coefficient between two binary volumes.

    Parameters
    ----------
    pred, target : np.ndarray
        Predicted and ground-truth volumes (any shape; thresholded internally).
    threshold : float
        Values >= threshold are considered occupied.
    """
    p = (pred >= threshold).astype(bool).ravel()
    t = (target >= threshold).astype(bool).ravel()
    intersection = np.logical_and(p, t).sum()
    union = p.sum() + t.sum()
    if union == 0:
        return 1.0  # both empty
    return 2.0 * intersection / union


# =========================================================================== #
#  Convert voxel grid → surface mesh                                           #
# =========================================================================== #

def voxel_to_surface(
    volume: np.ndarray,
    level: float = 0.5,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract an iso-surface mesh from a 3D voxel volume using marching cubes.

    Parameters
    ----------
    volume : (D, H, W) ndarray
    level : float
        Iso-surface threshold.
    spacing : tuple of float, optional
        Voxel spacing (dz, dy, dx) in physical units (e.g. mm).
        If None, unit spacing is assumed.

    Returns
    -------
    vertices : (N, 3) array
    faces : (M, 3) int array
    """
    if volume.ndim == 4:
        volume = volume.squeeze(0)  # remove channel dim
    spacing = spacing or (1.0, 1.0, 1.0)
    verts, faces, _, _ = marching_cubes(volume, level=level, spacing=spacing)
    return verts, faces


# =========================================================================== #
#  Surface distance metrics                                                    #
# =========================================================================== #

def surface_distances(
    verts_a: np.ndarray,
    verts_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-vertex closest-point distances between two point clouds.

    Returns
    -------
    a_to_b : (N_a,) array — distance from each vertex in A to nearest in B
    b_to_a : (N_b,) array — distance from each vertex in B to nearest in A
    """
    tree_b = cKDTree(verts_b)
    a_to_b, _ = tree_b.query(verts_a, k=1)

    tree_a = cKDTree(verts_a)
    b_to_a, _ = tree_a.query(verts_b, k=1)

    return a_to_b.astype(np.float64), b_to_a.astype(np.float64)


def hausdorff_distance(verts_a: np.ndarray, verts_b: np.ndarray) -> float:
    """Symmetric Hausdorff distance (max of two directed Hausdorff distances)."""
    a2b, b2a = surface_distances(verts_a, verts_b)
    return float(max(a2b.max(), b2a.max()))


def mean_surface_distance(verts_a: np.ndarray, verts_b: np.ndarray) -> float:
    """Average symmetric surface distance (mean of bidirectional means)."""
    a2b, b2a = surface_distances(verts_a, verts_b)
    return float(0.5 * (a2b.mean() + b2a.mean()))


def rms_surface_distance(verts_a: np.ndarray, verts_b: np.ndarray) -> float:
    """Symmetric RMS surface distance (root mean square of bidirectional distances)."""
    a2b, b2a = surface_distances(verts_a, verts_b)
    return float(np.sqrt(0.5 * (np.mean(a2b ** 2) + np.mean(b2a ** 2))))


# =========================================================================== #
#  Combined evaluation for a single subject                                    #
# =========================================================================== #

def evaluate_subject(
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
    subject_id: str = "",
    level: float = 0.5,
    spacing: Optional[Tuple[float, float, float]] = None,
    compute_surface: bool = True,
) -> MetricResult:
    """
    Compute all metrics for a single predicted vs. ground-truth skull pair.

    Parameters
    ----------
    pred_volume : (1, D, H, W) or (D, H, W) ndarray
    target_volume : same shape
    """
    result = MetricResult(subject_id=subject_id)

    # Dice (volumetric)
    result.dice = dice_coefficient(pred_volume, target_volume, threshold=level)

    # Volume deviation (percentage). Works in voxel counts; percent is invariant to units.
    p = (pred_volume >= level).astype(np.uint8)
    t = (target_volume >= level).astype(np.uint8)
    # Handle optional channel dimension
    if p.ndim == 4:
        p = p.squeeze(0)
    if t.ndim == 4:
        t = t.squeeze(0)
    vp = float(p.sum())
    vg = float(t.sum())
    result.volume_pred = vp
    result.volume_gt = vg
    if vg > 0:
        result.volume_deviation_percent = float(100.0 * abs(vp - vg) / vg)
    else:
        result.volume_deviation_percent = 0.0 if vp == 0 else float("inf")

    if compute_surface:
        try:
            pred_verts, _ = voxel_to_surface(pred_volume, level=level, spacing=spacing)
            gt_verts, _ = voxel_to_surface(target_volume, level=level, spacing=spacing)
        except (ValueError, RuntimeError):
            # Marching cubes may fail if the volume is empty
            result.hausdorff = float("inf")
            result.mean_surface_distance = float("inf")
            result.rms_surface_distance = float("inf")
            result.max_surface_distance = float("inf")
            return result

        if len(pred_verts) == 0 or len(gt_verts) == 0:
            result.hausdorff = float("inf")
            result.mean_surface_distance = float("inf")
            result.rms_surface_distance = float("inf")
            result.max_surface_distance = float("inf")
            return result

        a2b, b2a = surface_distances(pred_verts, gt_verts)
        max_d = float(max(a2b.max(), b2a.max()))
        mean_d = float(0.5 * (a2b.mean() + b2a.mean()))
        rms_d = float(np.sqrt(0.5 * (np.mean(a2b ** 2) + np.mean(b2a ** 2))))
        result.hausdorff = max_d
        result.max_surface_distance = max_d
        result.mean_surface_distance = mean_d
        result.rms_surface_distance = rms_d
        result.pred_to_gt_distances = a2b
        result.gt_to_pred_distances = b2a

    return result


def aggregate_results(results: list[MetricResult]) -> dict:
    """Compute aggregate statistics over a list of per-subject results."""
    dices = [r.dice for r in results]
    hausdorffs = [r.hausdorff for r in results if r.hausdorff != float("inf")]
    msds = [r.mean_surface_distance for r in results if r.mean_surface_distance != float("inf")]
    rmsds = [r.rms_surface_distance for r in results if r.rms_surface_distance != float("inf")]
    vol_devs = [r.volume_deviation_percent for r in results if r.volume_deviation_percent != float("inf")]
    return {
        "n_subjects": len(results),
        "dice_mean": float(np.mean(dices)) if dices else 0.0,
        "dice_std": float(np.std(dices)) if dices else 0.0,
        "dice_median": float(np.median(dices)) if dices else 0.0,
        "hausdorff_mean": float(np.mean(hausdorffs)) if hausdorffs else 0.0,
        "hausdorff_std": float(np.std(hausdorffs)) if hausdorffs else 0.0,
        "hausdorff_median": float(np.median(hausdorffs)) if hausdorffs else 0.0,
        "msd_mean": float(np.mean(msds)) if msds else 0.0,
        "msd_std": float(np.std(msds)) if msds else 0.0,
        "msd_median": float(np.median(msds)) if msds else 0.0,
        "rmsd_mean": float(np.mean(rmsds)) if rmsds else 0.0,
        "rmsd_std": float(np.std(rmsds)) if rmsds else 0.0,
        "rmsd_median": float(np.median(rmsds)) if rmsds else 0.0,
        "volume_dev_percent_mean": float(np.mean(vol_devs)) if vol_devs else 0.0,
        "volume_dev_percent_std": float(np.std(vol_devs)) if vol_devs else 0.0,
        "volume_dev_percent_median": float(np.median(vol_devs)) if vol_devs else 0.0,
    }
