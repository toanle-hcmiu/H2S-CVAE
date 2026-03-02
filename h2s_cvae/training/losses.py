"""
Loss functions for the H2S-CVAE.

* **Reconstruction loss** — Binary Cross-Entropy (for binary occupancy) or
  L1 / MSE (for SDF representation).
* **KL divergence** — between the posterior q(z|S,H) and the (optionally
  learned) prior p(z|H).
* **Combined loss** with KL annealing schedule.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# =========================================================================== #
#  KL divergence between two diagonal Gaussians                                #
# =========================================================================== #

def kl_divergence(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """
    Analytic KL divergence  D_KL( q || p )  for two diagonal Gaussians.

    .. math::
        D_{KL} = \\frac{1}{2}\\sum\\left(
            \\frac{\\sigma_q^2}{\\sigma_p^2}
            + \\frac{(\\mu_p - \\mu_q)^2}{\\sigma_p^2}
            - 1
            + \\log\\frac{\\sigma_p^2}{\\sigma_q^2}
        \\right)

    Returns a scalar (mean over batch).
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    kl = 0.5 * (
        var_q / var_p
        + (mu_p - mu_q).pow(2) / var_p
        - 1.0
        + logvar_p - logvar_q
    )
    # Sum over latent dimensions, mean over batch
    return kl.sum(dim=-1).mean()


# =========================================================================== #
#  Reconstruction losses                                                       #
# =========================================================================== #

def reconstruction_loss_bce(
    recon: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Voxel-wise Binary Cross-Entropy (for binary occupancy grids)."""
    return F.binary_cross_entropy(recon, target, reduction="mean")


def reconstruction_loss_l1(
    recon: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Voxel-wise L1 / MAE loss (for SDF representation)."""
    return F.l1_loss(recon, target, reduction="mean")


def reconstruction_loss_mse(
    recon: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Voxel-wise MSE loss."""
    return F.mse_loss(recon, target, reduction="mean")


# =========================================================================== #
#  KL weight (β) annealing schedule                                            #
# =========================================================================== #

def kl_weight_schedule(
    epoch: int,
    target: float,
    anneal_epochs: int,
) -> float:
    """
    Linear warm-up of the KL weight β from 0 to *target* over
    *anneal_epochs* epochs.
    """
    if anneal_epochs <= 0:
        return target
    return min(target, target * epoch / anneal_epochs)


# =========================================================================== #
#  Combined CVAE loss                                                          #
# =========================================================================== #

def cvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    kl_weight: float = 1.0,
    representation: str = "binary",
) -> dict[str, torch.Tensor]:
    """
    Compute the full CVAE loss.

    Returns a dict with ``"loss"`` (total), ``"recon_loss"``, and ``"kl_loss"``.
    """
    # Reconstruction
    if representation == "binary":
        l_recon = reconstruction_loss_bce(recon, target)
    elif representation == "sdf":
        l_recon = reconstruction_loss_l1(recon, target)
    else:
        l_recon = reconstruction_loss_mse(recon, target)

    # KL
    l_kl = kl_divergence(mu_q, logvar_q, mu_p, logvar_p)

    total = l_recon + kl_weight * l_kl

    return {
        "loss": total,
        "recon_loss": l_recon,
        "kl_loss": l_kl,
    }
