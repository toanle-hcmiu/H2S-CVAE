"""
Preprocessing entry-point: voxelise all head / skull .ply meshes and cache them
as .npy files for fast loading during training.

Usage
-----
    python preprocess.py                     # defaults (64^3, binary)
    python preprocess.py --resolution 128    # higher resolution
    python preprocess.py --representation sdf
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2s_cvae.config import get_default_config
from h2s_cvae.data.voxelizer import (
    compute_global_bounds,
    load_bounds,
    save_bounds,
    voxelize_and_save,
)


def read_ids(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Voxelise head/skull meshes.")
    parser.add_argument("--resolution", type=int, default=64, help="Voxel grid resolution (default: 64)")
    parser.add_argument("--representation", type=str, default="binary", choices=["binary", "sdf"],
                        help="Voxel representation type (default: binary)")
    parser.add_argument("--padding", type=float, default=0.05,
                        help="Fractional padding around global bounding box (default: 0.05)")
    parser.add_argument("--force", action="store_true",
                        help="Re-voxelise even if .npy files already exist")
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.data.voxel_resolution = args.resolution
    cfg.data.representation = args.representation
    cfg.data.padding_ratio = args.padding

    voxel_dir = cfg.paths.resolve_voxel_dir(args.resolution)
    os.makedirs(voxel_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Collect all subject IDs (union of train + test)
    # ------------------------------------------------------------------ #
    all_ids = read_ids(cfg.paths.subject_id_file)
    print(f"Total subjects: {len(all_ids)}")

    # ------------------------------------------------------------------ #
    # Step 1: Compute or load global bounding box
    # ------------------------------------------------------------------ #
    bounds_file = os.path.join(voxel_dir, "global_bounds.json")
    if os.path.isfile(bounds_file) and not args.force:
        print(f"Loading existing global bounds from {bounds_file}")
        bounds = load_bounds(bounds_file)
    else:
        print("Computing global bounding box across all meshes ...")
        t0 = time.time()
        raw_bounds = compute_global_bounds(
            mesh_dir=cfg.paths.mesh_dir,
            subject_ids=all_ids,
            padding_ratio=args.padding,
        )
        save_bounds(raw_bounds, bounds_file)
        bounds = {k: np.array(v) for k, v in raw_bounds.items()}
        print(f"  Done in {time.time() - t0:.1f}s")

    bounds_min, bounds_max = bounds["min"], bounds["max"]
    print(f"  Bounds min: {bounds_min}")
    print(f"  Bounds max: {bounds_max}")
    print(f"  Extent:     {bounds_max - bounds_min}")

    # ------------------------------------------------------------------ #
    # Step 2: Voxelise each mesh
    # ------------------------------------------------------------------ #
    suffixes = ["-FullHead.ply", "-FullSkull.ply"]
    total = len(all_ids) * len(suffixes)
    skipped = 0

    print(f"\nVoxelising {total} meshes at resolution {args.resolution}^3 ({args.representation}) ...")
    for sid in tqdm(all_ids, desc="Subjects"):
        for suffix in suffixes:
            mesh_path = os.path.join(cfg.paths.mesh_dir, f"{sid}{suffix}")
            npy_name = f"{sid}{suffix.replace('.ply', '.npy')}"
            save_path = os.path.join(voxel_dir, npy_name)

            if os.path.isfile(save_path) and not args.force:
                skipped += 1
                continue

            if not os.path.isfile(mesh_path):
                print(f"\n[warn] Mesh not found, skipping: {mesh_path}")
                continue

            voxelize_and_save(
                mesh_path=mesh_path,
                save_path=save_path,
                resolution=args.resolution,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                representation=args.representation,
            )

            # Free trimesh caches, R-tree, and intersection buffers
            # between meshes to prevent memory accumulation
            gc.collect()

    print(f"\nDone. Voxelised files saved to: {voxel_dir}")
    if skipped:
        print(f"  ({skipped} files already existed and were skipped; use --force to overwrite)")


if __name__ == "__main__":
    main()
