\# Head-to-Skull 3D Conditional VAE (H2S-CVAE)

Predicting the internal skull surface geometry from external head surface meshes
using a **3D CNN-based Conditional Variational Autoencoder (CVAE)** in PyTorch.

## Project Structure

```
H2S-CVAE/
├── README.md
├── requirements.txt
├── ExampleDataProcessingScript.py    # Original mesh-loading utility
├── preprocess.py                     # Step 1: voxelise all meshes
├── train.py                          # Step 2: train the CVAE
├── evaluate.py                       # Step 3: evaluate on held-out test set
├── Data/
│   ├── PostProcessedSubjectIDs_PrevPapers.txt
│   ├── CrossValidation/
│   │   ├── TrainingIDs.txt           # 80% of subjects
│   │   └── TestingIDs.txt            # 20% of subjects
│   ├── HeadAndSkullShapes/           # 329 paired .ply meshes
│   └── Voxelized/<resolution>/       # cached .npy voxel grids (generated)
├── h2s_cvae/                         # Python package
│   ├── config.py                     # All hyperparameters & paths
│   ├── data/
│   │   ├── voxelizer.py              # Mesh → voxel conversion
│   │   └── dataset.py                # PyTorch Dataset & DataLoader
│   ├── models/
│   │   └── cvae.py                   # 3D CNN CVAE architecture
│   ├── training/
│   │   ├── losses.py                 # CVAE loss (BCE + KL + annealing)
│   │   └── trainer.py                # Training loop
│   ├── evaluation/
│   │   └── metrics.py                # Dice, Hausdorff, MSD metrics
│   └── utils/
│       └── visualization.py          # Slice plots, 3D rendering
├── checkpoints/                      # saved model weights (generated)
├── logs/                             # TensorBoard logs (generated)
└── outputs/                          # evaluation results & meshes (generated)
```

## Architecture

The model is a **3D CNN Conditional VAE** with four sub-networks:

| Component            | Input                  | Output                       |
|----------------------|------------------------|------------------------------|
| Condition Encoder    | Head volume (1,D,H,W)  | Multi-scale feature maps + feature vector |
| Posterior Encoder    | (Skull, Head) concat   | μ_q, log σ²_q               |
| Prior Network        | Head feature vector    | μ_p, log σ²_p (learned)     |
| Decoder              | z + head features      | Predicted skull (1,D,H,W)   |

**Training**: z ~ q(z|S,H) (posterior); **Inference**: z ~ p(z|H) (head-conditioned prior).

Skip connections inject multi-scale head features into the decoder at each spatial level.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess — voxelise meshes

```bash
python preprocess.py                     # 64^3 binary, ~5% padding
python preprocess.py --resolution 128    # higher resolution (more memory)
```

This computes a shared global bounding box across all 329 subjects and converts
each `.ply` mesh to a binary occupancy grid saved as `.npy`.

### 3. Train

```bash
python train.py                          # default settings
python train.py --epochs 200 --lr 5e-4   # override hyperparameters
python train.py --resume                 # resume from latest checkpoint
```

Monitor training via TensorBoard:
```bash
tensorboard --logdir logs/
```

### 4. Evaluate

```bash
python evaluate.py                       # uses best checkpoint
python evaluate.py --save-meshes         # also export predicted .ply files
```

Outputs a JSON report with per-subject Dice, Hausdorff distance, and mean
surface distance, plus aggregate statistics.

## Key Hyperparameters

| Parameter          | Default | Notes                                    |
|--------------------|---------|------------------------------------------|
| `voxel_resolution` | 64      | Grid size per axis (64³ or 128³)         |
| `latent_dim`       | 256     | VAE latent space dimensionality          |
| `batch_size`       | 4       | Limited by GPU memory for 3D data        |
| `learning_rate`    | 1e-4    | Adam optimiser                           |
| `kl_weight_target` | 0.001   | β in β-VAE; annealed from 0             |
| `kl_anneal_epochs` | 100     | Linear warm-up of β                      |
| `num_epochs`       | 500     | Total training epochs                    |

All settings are in `h2s_cvae/config.py`.

## Data

329 paired subjects from 5 hospital cohorts (HN-CHUM, HN-CHUS, HN-HGJ,
HN-HMR, HNSCC). Each subject has:
- `{SubjectID}-FullHead.ply` — external head surface mesh
- `{SubjectID}-FullSkull.ply` — skull bone surface mesh

Train/test split: 80% / 20% (non-overlapping, seed = 42).

## Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{KL}\big(q_\phi(z|S,H) \;\|\; p_\theta(z|H)\big)$$

- **Reconstruction**: Binary Cross-Entropy (binary occupancy) or L1 (SDF)
- **KL divergence**: Analytic, between posterior and learned head-conditioned prior
- **β annealing**: Linear warm-up from 0 → target over first 100 epochs

