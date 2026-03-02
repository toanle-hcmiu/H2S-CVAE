"""
Mesh-to-voxel conversion utilities.

Converts .ply triangle meshes to 3D binary occupancy grids or signed distance
fields (SDFs) at a configurable resolution, using a shared global bounding box
for spatial consistency across the entire dataset.
"""

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh


def load_mesh(filepath: str) -> trimesh.Trimesh:
    """Load a .ply mesh and ensure it is a single Trimesh object."""
    mesh = trimesh.load(filepath, file_type="ply", force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a single Trimesh, got {type(mesh)} from {filepath}")
    return mesh


def compute_global_bounds(
    mesh_dir: str,
    subject_ids: list,
    padding_ratio: float = 0.05,
) -> Dict[str, list]:
    """
    Compute a shared axis-aligned bounding box that encompasses every head and
    skull mesh in the dataset, with optional fractional padding.

    Returns a dict with keys ``"min"`` and ``"max"`` (each a 3-element list)
    suitable for JSON serialisation.
    """
    global_min = np.full(3, np.inf)
    global_max = np.full(3, -np.inf)

    for sid in subject_ids:
        for suffix in ("-FullHead.ply", "-FullSkull.ply"):
            path = os.path.join(mesh_dir, f"{sid}{suffix}")
            if not os.path.isfile(path):
                print(f"[warn] mesh not found, skipping: {path}")
                continue
            mesh = load_mesh(path)
            global_min = np.minimum(global_min, mesh.vertices.min(axis=0))
            global_max = np.maximum(global_max, mesh.vertices.max(axis=0))

    # Add padding
    extent = global_max - global_min
    pad = extent * padding_ratio
    global_min -= pad
    global_max += pad

    return {"min": global_min.tolist(), "max": global_max.tolist()}


def save_bounds(bounds: Dict[str, list], filepath: str) -> None:
    """Serialise bounding-box parameters to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(bounds, f, indent=2)
    print(f"Saved global bounds to {filepath}")


def load_bounds(filepath: str) -> Dict[str, np.ndarray]:
    """Load previously saved bounding-box parameters."""
    with open(filepath, "r") as f:
        raw = json.load(f)
    return {"min": np.array(raw["min"]), "max": np.array(raw["max"])}


def voxelize_mesh(
    mesh: trimesh.Trimesh,
    resolution: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    representation: str = "binary",
) -> np.ndarray:
    """
    Convert a triangle mesh to a volumetric 3D array.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input surface mesh.
    resolution : int
        Number of voxels along each axis (produces a resolution^3 grid).
    bounds_min, bounds_max : np.ndarray
        Global bounding-box corners (shape (3,)).
    representation : str
        ``"binary"`` for binary occupancy (1 inside / 0 outside),
        ``"sdf"`` for a truncated signed-distance field.

    Returns
    -------
    np.ndarray
        Volume of shape ``(1, resolution, resolution, resolution)`` with a
        leading channel dimension (C=1) ready for PyTorch.
    """
    extent = bounds_max - bounds_min
    # Pitch = voxel edge length (uniform along all 3 axes, determined by the
    # largest extent so the grid stays cubic in index space)
    pitch = extent.max() / resolution

    # Build a regular grid of query points centred in each voxel
    lin = [bounds_min[ax] + pitch * (np.arange(resolution) + 0.5) for ax in range(3)]
    gx, gy, gz = np.meshgrid(lin[0], lin[1], lin[2], indexing="ij")
    query_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    if representation == "binary":
        # Use trimesh's contains to determine inside / outside
        inside = mesh.contains(query_points)
        volume = inside.astype(np.float32).reshape(resolution, resolution, resolution)
    elif representation == "sdf":
        # Signed distance: negative inside, positive outside
        sdf = trimesh.proximity.signed_distance(mesh, query_points)
        # Normalise to [-1, 1] range using a truncation distance
        trunc = pitch * 3  # truncate at 3 voxels
        sdf = np.clip(sdf, -trunc, trunc) / trunc
        volume = sdf.astype(np.float32).reshape(resolution, resolution, resolution)
    else:
        raise ValueError(f"Unknown representation '{representation}'. Use 'binary' or 'sdf'.")

    # Add channel dimension  →  (1, D, H, W)
    return volume[np.newaxis, ...]


def voxelize_and_save(
    mesh_path: str,
    save_path: str,
    resolution: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    representation: str = "binary",
) -> None:
    """Load a mesh, voxelize it, and save the result as a compressed .npy file."""
    mesh = load_mesh(mesh_path)
    vol = voxelize_mesh(mesh, resolution, bounds_min, bounds_max, representation)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, vol)
