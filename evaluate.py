"""
Evaluation entry-point — run inference on the test set and report metrics.

Usage
-----
    python evaluate.py                                  # uses best checkpoint
    python evaluate.py --checkpoint checkpoints/checkpoint_epoch_499.pt
    python evaluate.py --save-meshes                    # export predicted .ply
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2s_cvae.config import get_default_config
from h2s_cvae.data.dataset import read_subject_ids
from h2s_cvae.evaluation.metrics import (
    MetricResult,
    aggregate_results,
    evaluate_subject,
    voxel_to_surface,
)
from h2s_cvae.models.cvae import HeadToSkullCVAE


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained H2S-CVAE model.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: best)")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-meshes", action="store_true",
                        help="Export predicted skulls as .ply files")
    parser.add_argument("--use-prior-mean", action="store_true", default=True,
                        help="Use prior mean (deterministic) instead of sampling")
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.data.voxel_resolution = args.resolution
    cfg.paths.resolve_voxel_dir(args.resolution)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    ckpt_path = args.checkpoint or os.path.join(cfg.paths.checkpoint_dir, "checkpoint_best.pt")
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = HeadToSkullCVAE(
        latent_dim=cfg.model.latent_dim,
        encoder_channels=cfg.model.encoder_channels,
        decoder_channels=cfg.model.decoder_channels,
        condition_channels=cfg.model.condition_channels,
        use_learned_prior=cfg.model.use_learned_prior,
        use_skip_connections=cfg.model.use_skip_connections,
        spatial_size=cfg.data.voxel_resolution,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # ------------------------------------------------------------------ #
    # Load test IDs
    # ------------------------------------------------------------------ #
    test_ids = read_subject_ids(cfg.paths.testing_id_file)
    print(f"Evaluating on {len(test_ids)} test subjects ...")

    voxel_dir = cfg.paths.resolve_voxel_dir(args.resolution)
    output_dir = cfg.paths.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if args.save_meshes:
        mesh_out_dir = os.path.join(output_dir, "predicted_meshes")
        os.makedirs(mesh_out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Run inference + evaluation
    # ------------------------------------------------------------------ #
    results: list[MetricResult] = []

    for sid in tqdm(test_ids, desc="Evaluating"):
        head_path = os.path.join(voxel_dir, f"{sid}-FullHead.npy")
        skull_path = os.path.join(voxel_dir, f"{sid}-FullSkull.npy")

        if not os.path.isfile(head_path) or not os.path.isfile(skull_path):
            print(f"  [skip] Missing voxel file for {sid}")
            continue

        head_vol = np.load(head_path)   # (1, D, H, W)
        skull_vol = np.load(skull_path)

        # Inference
        head_tensor = torch.from_numpy(head_vol).unsqueeze(0).float().to(device)  # (1,1,D,H,W)
        pred_tensor = model.predict(head_tensor, use_prior_mean=args.use_prior_mean)
        pred_vol = pred_tensor.squeeze(0).cpu().numpy()  # (1, D, H, W)

        # Metrics
        result = evaluate_subject(
            pred_volume=pred_vol,
            target_volume=skull_vol,
            subject_id=sid,
            level=cfg.evaluation.marching_cubes_level,
        )
        results.append(result)

        # Save predicted mesh
        if args.save_meshes:
            try:
                verts, faces = voxel_to_surface(pred_vol, level=cfg.evaluation.marching_cubes_level)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                mesh.export(os.path.join(mesh_out_dir, f"{sid}-PredSkull.ply"))
            except Exception as e:
                print(f"  [warn] Could not export mesh for {sid}: {e}")

    # ------------------------------------------------------------------ #
    # Aggregate and report
    # ------------------------------------------------------------------ #
    agg = aggregate_results(results)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Subjects evaluated: {agg['n_subjects']}")
    print(f"  Dice coefficient:   {agg['dice_mean']:.4f} ± {agg['dice_std']:.4f} "
          f"(median {agg['dice_median']:.4f})")
    print(f"  Hausdorff distance: {agg['hausdorff_mean']:.3f} ± {agg['hausdorff_std']:.3f} "
          f"(median {agg['hausdorff_median']:.3f})")
    print(f"  Mean Surf. Dist.:   {agg['msd_mean']:.3f} ± {agg['msd_std']:.3f} "
          f"(median {agg['msd_median']:.3f})")
    print("=" * 60)

    # Save results to JSON
    results_json = {
        "aggregate": agg,
        "per_subject": [
            {
                "subject_id": r.subject_id,
                "dice": r.dice,
                "hausdorff": r.hausdorff,
                "mean_surface_distance": r.mean_surface_distance,
            }
            for r in results
        ],
    }
    out_path = os.path.join(output_dir, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
