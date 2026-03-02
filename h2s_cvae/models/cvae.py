"""
3D CNN-based Conditional Variational Autoencoder (CVAE) for Head-to-Skull
prediction.

Architecture overview
---------------------
* **Condition Encoder** — processes the head volume *H* through a series of 3D
  conv blocks to produce multi-scale feature maps **and** a compact feature
  vector used by the learned prior and the decoder.
* **Posterior Encoder** — takes the concatenation of the skull volume *S* and
  the head volume *H*, downsamples through 3D conv blocks, and outputs the
  posterior distribution parameters (μ, log σ²).
* **Prior Network** — small MLP that maps the head feature vector to prior
  distribution parameters (μ_prior, log σ²_prior), giving a head-conditioned
  prior *p(z|H)* instead of a fixed *N(0, I)*.
* **Decoder** — takes the latent sample *z* (from reparameterisation) together
  with the head condition features, progressively upsamples to reconstruct the
  skull volume *S'*.  Optional skip connections inject multi-scale head features
  at each decoder level.

All spatial operations use 3D convolutions so that the model natively handles
volumetric (voxelised) data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
#  Building blocks                                                             #
# =========================================================================== #

class ConvBlock3D(nn.Module):
    """Conv3d → BatchNorm → LeakyReLU (encoder-style)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConvBlock3D(nn.Module):
    """ConvTranspose3d → BatchNorm → ReLU (decoder-style)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(
                in_ch, out_ch,
                kernel_size=4, stride=stride, padding=1, bias=False,
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =========================================================================== #
#  Condition Encoder — extracts multi-scale features from the head volume      #
# =========================================================================== #

class ConditionEncoder(nn.Module):
    """
    Encodes the head volume *H* into:
    * ``feature_maps`` — a list of intermediate feature volumes at each spatial
      scale (for skip connections into the decoder).
    * ``feature_vector`` — a 1-D vector summarising the whole head shape (used
      by the prior network and concatenated into the decoder input).
    """

    def __init__(self, in_channels: int = 1, channels: List[int] | None = None):
        super().__init__()
        channels = channels or [32, 64, 128, 256]
        layers: list[nn.Module] = []
        ch_in = in_channels
        for ch_out in channels:
            layers.append(ConvBlock3D(ch_in, ch_out, stride=2))
            ch_in = ch_out
        self.blocks = nn.ModuleList(layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        feature_maps: list[torch.Tensor] = []
        h = x
        for block in self.blocks:
            h = block(h)
            feature_maps.append(h)
        # Global feature vector — adaptive pool → flatten
        feat_vec = F.adaptive_avg_pool3d(h, 1).view(h.size(0), -1)
        return feature_maps, feat_vec


# =========================================================================== #
#  Posterior Encoder  q_φ(z | S, H)                                            #
# =========================================================================== #

class PosteriorEncoder(nn.Module):
    """
    Maps concatenated (skull, head) volumes to posterior distribution parameters
    (μ, log σ²).
    """

    def __init__(
        self,
        in_channels: int = 2,
        channels: List[int] | None = None,
        latent_dim: int = 256,
        spatial_size: int = 64,
    ):
        super().__init__()
        channels = channels or [32, 64, 128, 256, 512]
        layers: list[nn.Module] = []
        ch_in = in_channels
        for ch_out in channels:
            layers.append(ConvBlock3D(ch_in, ch_out, stride=2))
            ch_in = ch_out
        self.blocks = nn.Sequential(*layers)

        # Compute flattened spatial size after all down-sampling steps
        ds_factor = 2 ** len(channels)
        final_spatial = spatial_size // ds_factor  # e.g. 64 / 32 = 2
        flat_dim = channels[-1] * (final_spatial ** 3)

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, skull: torch.Tensor, head: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([skull, head], dim=1)  # (B, 2, D, H, W)
        h = self.blocks(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


# =========================================================================== #
#  Prior Network  p_θ(z | H)                                                   #
# =========================================================================== #

class PriorNetwork(nn.Module):
    """
    Maps the head feature vector to prior distribution parameters (μ, log σ²).
    When ``use_learned_prior=False`` this effectively returns N(0, I).
    """

    def __init__(self, feat_dim: int, latent_dim: int, use_learned: bool = True):
        super().__init__()
        self.use_learned = use_learned
        if use_learned:
            self.net = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
            )
            self.fc_mu = nn.Linear(feat_dim, latent_dim)
            self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, feat_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_learned:
            B = feat_vec.size(0)
            device = feat_vec.device
            return (
                torch.zeros(B, self.latent_dim, device=device),
                torch.zeros(B, self.latent_dim, device=device),
            )
        h = self.net(feat_vec)
        return self.fc_mu(h), self.fc_logvar(h)


# =========================================================================== #
#  Decoder  D_θ(z, H) → S'                                                    #
# =========================================================================== #

class Decoder(nn.Module):
    """
    Progressively upsamples a latent code *z* (+ head condition) to produce
    a full-resolution skull volume.

    If ``use_skip_connections`` is True, multi-scale head feature maps from the
    :class:`ConditionEncoder` are concatenated at the matching decoder level.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        condition_feat_dim: int = 256,
        channels: List[int] | None = None,
        condition_channels: List[int] | None = None,
        spatial_size: int = 64,
        use_skip_connections: bool = True,
    ):
        super().__init__()
        channels = channels or [512, 256, 128, 64, 32]
        condition_channels = condition_channels or [32, 64, 128, 256]
        self.use_skip = use_skip_connections
        self.num_ups = len(channels)  # must match number of encoder down-samples

        # Fully-connected projection: z + head_feat → spatial volume
        self.initial_spatial = spatial_size // (2 ** self.num_ups)  # e.g. 2 for 64 w/ 5 ups
        fc_in = latent_dim + condition_feat_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, channels[0] * (self.initial_spatial ** 3)),
            nn.ReLU(inplace=True),
        )
        self.initial_channels = channels[0]

        # Build up-conv blocks
        up_layers: list[nn.Module] = []
        ch_in = channels[0]
        for i, ch_out in enumerate(channels[1:]):
            # If skips: concat condition features at matching scale
            if self.use_skip:
                # condition_channels go from small→large; decoder goes large→small
                # Decoder level 0 is smallest (initial_spatial * 2),
                # decoder level i upsamples from level i to i+1
                # The condition encoder feature_maps[j] has spatial =
                #     spatial_size / 2^(j+1), so we pair decoder level i with
                #     condition_channels[-(i+1)] for matching spatial size after
                #     the up-conv.
                skip_idx = len(condition_channels) - 1 - i
                if 0 <= skip_idx < len(condition_channels):
                    skip_ch = condition_channels[skip_idx]
                else:
                    skip_ch = 0
                up_layers.append(UpConvBlock3D(ch_in + skip_ch, ch_out))
            else:
                up_layers.append(UpConvBlock3D(ch_in, ch_out))
            ch_in = ch_out
        self.up_blocks = nn.ModuleList(up_layers)

        # Final up-conv to reach full resolution (channels[-1] → 1)
        final_in = channels[-1]
        if self.use_skip and len(condition_channels) >= self.num_ups:
            # There may be one more condition level to inject
            skip_idx_final = len(condition_channels) - 1 - (self.num_ups - 1)
            if 0 <= skip_idx_final < len(condition_channels):
                final_in += condition_channels[skip_idx_final]
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(final_in, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        head_feat_vec: torch.Tensor,
        head_feature_maps: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        B = z.size(0)
        # Project to spatial volume
        h = self.fc(torch.cat([z, head_feat_vec], dim=1))
        h = h.view(B, self.initial_channels, self.initial_spatial, self.initial_spatial, self.initial_spatial)

        # Progressively upsample
        cond_maps = head_feature_maps or []
        # Condition feature maps indices: [0] smallest scale … [-1] largest scale
        # We iterate from smallest to largest in the decoder
        # The condition encoder produces maps at scales:
        #   feature_maps[0] → spatial/2, feature_maps[1] → spatial/4, etc.
        # So feature_maps[-1] is the smallest spatial. We reverse for decoder use.
        cond_reversed = list(reversed(cond_maps))

        for i, up_block in enumerate(self.up_blocks):
            if self.use_skip and i < len(cond_reversed):
                # The up_block expects (ch_in + skip_ch) input channels
                # We first up-sample h, then concatenate the matching condition map
                # But up_block already does the up-conv, so we need to concat BEFORE.
                # Actually, the UpConvBlock3D expects the full concat input.
                # So we concat first, then upsample.
                cond_feat = cond_reversed[i]
                # Ensure spatial sizes match before concat
                if cond_feat.shape[2:] != h.shape[2:]:
                    cond_feat = F.interpolate(cond_feat, size=h.shape[2:], mode="trilinear", align_corners=False)
                h = torch.cat([h, cond_feat], dim=1)
            h = up_block(h)

        # Final up-conv
        if self.use_skip and len(cond_reversed) >= self.num_ups:
            last_idx = self.num_ups - 1
            if last_idx < len(cond_reversed):
                cond_feat = cond_reversed[last_idx]
                if cond_feat.shape[2:] != h.shape[2:]:
                    cond_feat = F.interpolate(cond_feat, size=h.shape[2:], mode="trilinear", align_corners=False)
                h = torch.cat([h, cond_feat], dim=1)
        h = self.final_up(h)
        return h


# =========================================================================== #
#  Full CVAE model                                                             #
# =========================================================================== #

class HeadToSkullCVAE(nn.Module):
    """
    3D CNN Conditional VAE for head-to-skull prediction.

    During **training**, both the head *H* and ground-truth skull *S* are
    provided.  The model encodes `(S, H) → q(z|S,H)`, samples *z*, and
    decodes `(z, H) → S'`.

    During **inference**, only *H* is given.  *z* is sampled from the learned
    prior `p(z|H)` and decoded to produce a skull prediction.

    Parameters match the ``ModelConfig`` dataclass fields in ``config.py``.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        encoder_channels: List[int] | None = None,
        decoder_channels: List[int] | None = None,
        condition_channels: List[int] | None = None,
        use_learned_prior: bool = True,
        use_skip_connections: bool = True,
        spatial_size: int = 64,
    ):
        super().__init__()
        encoder_channels = encoder_channels or [32, 64, 128, 256, 512]
        decoder_channels = decoder_channels or [512, 256, 128, 64, 32]
        condition_channels = condition_channels or [32, 64, 128, 256]

        self.latent_dim = latent_dim
        self.spatial_size = spatial_size

        # Sub-networks
        self.condition_encoder = ConditionEncoder(
            in_channels=1, channels=condition_channels,
        )
        self.posterior_encoder = PosteriorEncoder(
            in_channels=2,
            channels=encoder_channels,
            latent_dim=latent_dim,
            spatial_size=spatial_size,
        )
        self.prior_network = PriorNetwork(
            feat_dim=condition_channels[-1],
            latent_dim=latent_dim,
            use_learned=use_learned_prior,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            condition_feat_dim=condition_channels[-1],
            channels=decoder_channels,
            condition_channels=condition_channels,
            spatial_size=spatial_size,
            use_skip_connections=use_skip_connections,
        )

    # ------------------------------------------------------------------ #
    #  Reparameterisation trick                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------ #
    #  Forward pass                                                       #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        head: torch.Tensor,
        skull: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        head : (B, 1, D, H, W)
            Input head voxel volume.
        skull : (B, 1, D, H, W), optional
            Ground-truth skull volume (provided during training).

        Returns
        -------
        dict with keys:
            ``recon`` — reconstructed skull volume (B, 1, D, H, W)
            ``mu_q``  — posterior mean (training only)
            ``logvar_q`` — posterior log-variance (training only)
            ``mu_p``  — prior mean
            ``logvar_p`` — prior log-variance
        """
        # --- Condition encoding ---
        cond_maps, cond_vec = self.condition_encoder(head)

        # --- Prior ---
        mu_p, logvar_p = self.prior_network(cond_vec)

        if skull is not None:
            # --- Training: use posterior ---
            mu_q, logvar_q = self.posterior_encoder(skull, head)
            z = self.reparameterise(mu_q, logvar_q)
        else:
            # --- Inference: sample from prior ---
            mu_q = logvar_q = None
            z = self.reparameterise(mu_p, logvar_p)

        # --- Decoding ---
        recon = self.decoder(z, cond_vec, cond_maps)

        return {
            "recon": recon,
            "mu_q": mu_q,
            "logvar_q": logvar_q,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
        }

    # ------------------------------------------------------------------ #
    #  Inference helper                                                   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        head: torch.Tensor,
        use_prior_mean: bool = True,
    ) -> torch.Tensor:
        """
        Predict skull from head volume at inference time.

        Parameters
        ----------
        head : (B, 1, D, H, W)
        use_prior_mean : bool
            If True, use the prior mean deterministically (no sampling).
            If False, sample from the prior distribution.

        Returns
        -------
        Predicted skull volume (B, 1, D, H, W).
        """
        self.eval()
        cond_maps, cond_vec = self.condition_encoder(head)
        mu_p, logvar_p = self.prior_network(cond_vec)

        if use_prior_mean:
            z = mu_p
        else:
            z = self.reparameterise(mu_p, logvar_p)

        return self.decoder(z, cond_vec, cond_maps)
