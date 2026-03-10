"""
Contrastive Self-Supervised Learning for Time Series Anomaly Detection
=======================================================================
Uses SimCLR-style contrastive learning to learn representations of
normal time-series behavior. Anomalies are far from normal clusters.

Key Idea:
- Two augmented views of the same window → pulled together (positive pairs)
- Two views from different windows → pushed apart (negative pairs)
- At inference: distance from normal cluster = anomaly score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Data Augmentation for Time Series
# ---------------------------------------------------------------------------

class TimeSeriesAugmentor:
    """
    Augmentation strategies for self-supervised contrastive learning.
    Designed to preserve semantic content while creating diverse views.
    """

    def __init__(self, noise_std: float = 0.05, mask_ratio: float = 0.15,
                 scale_range: Tuple[float, float] = (0.8, 1.2),
                 jitter_ratio: float = 0.1):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.scale_range = scale_range
        self.jitter_ratio = jitter_ratio

    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add small Gaussian noise to simulate sensor measurement error."""
        return x + torch.randn_like(x) * self.noise_std

    def temporal_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero out time steps to force context learning."""
        mask = torch.bernoulli(
            torch.full((x.shape[0], x.shape[1], 1), 1 - self.mask_ratio)
        ).to(x.device)
        return x * mask

    def amplitude_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Scale signal amplitude within a range to simulate gain variation."""
        lo, hi = self.scale_range
        scale = torch.FloatTensor(x.shape[0], 1, 1).uniform_(lo, hi).to(x.device)
        return x * scale

    def temporal_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly shift segments for local temporal invariance."""
        jitter = int(x.shape[1] * self.jitter_ratio)
        if jitter == 0:
            return x
        shift = torch.randint(-jitter, jitter + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=1)

    def window_slicing(self, x: torch.Tensor) -> torch.Tensor:
        """Crop and resize a random sub-window back to original length."""
        seq_len = x.shape[1]
        crop_len = int(seq_len * (1 - self.mask_ratio))
        start = torch.randint(0, seq_len - crop_len + 1, (1,)).item()
        cropped = x[:, start:start + crop_len, :]
        # Resize back via interpolation
        return F.interpolate(
            cropped.permute(0, 2, 1),
            size=seq_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a random chain of augmentations to produce one view."""
        # Randomly apply 2 out of 4 augmentations
        augmentations = [
            self.gaussian_noise,
            self.temporal_masking,
            self.amplitude_scaling,
            self.temporal_jitter,
        ]
        selected = np.random.choice(len(augmentations), size=2, replace=False)
        for idx in selected:
            x = augmentations[idx](x)
        return x

    def get_two_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two independently augmented views of the same batch."""
        return self.augment(x.clone()), self.augment(x.clone())


# ---------------------------------------------------------------------------
# Encoder Backbone
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Transformer encoder backbone for contrastive representation learning.
    Captures global temporal dependencies through self-attention.
    """

    def __init__(self, input_dim: int, embed_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        # Global average pooling over time dimension
        return self.pool(x.permute(0, 2, 1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Non-linear projection head maps encoder output to contrastive space.
    Used during training only — features before projection are used for detection.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Contrastive Anomaly Detector
# ---------------------------------------------------------------------------

class ContrastiveAnomalyDetector(nn.Module):
    """
    SimCLR-based anomaly detector for time series.

    Training: learns representations via NT-Xent contrastive loss.
    Inference: anomaly score = distance from normal representation centroid.

    Args:
        input_dim:      Input features per time step
        embed_dim:      Transformer embedding dimension
        proj_dim:       Contrastive projection space dimension
        temperature:    NT-Xent softmax temperature
    """

    def __init__(self, input_dim: int, embed_dim: int = 64, proj_dim: int = 32,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1,
                 temperature: float = 0.07):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, embed_dim, num_heads,
                                           num_layers, dropout)
        self.projector = ProjectionHead(embed_dim, embed_dim, proj_dim)
        self.temperature = temperature
        self.augmentor = TimeSeriesAugmentor()

        # Normal distribution statistics (set during calibration)
        self.register_buffer('normal_centroid', torch.zeros(embed_dim))
        self.register_buffer('normal_std', torch.ones(embed_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized encoder representation (for downstream use)."""
        return F.normalize(self.encoder(x), dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get projected representations of two augmented views."""
        view1, view2 = self.augmentor.get_two_views(x)
        z1 = self.projector(self.encoder(view1))
        z2 = self.projector(self.encoder(view2))
        return z1, z2

    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
        Pulls positive pairs together, pushes negative pairs apart.
        """
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        # Cosine similarity matrix
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # Mask out self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pair indices: (i, i+N) and (i+N, i)
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z.device)

        return F.cross_entropy(sim, labels)

    def calibrate(self, normal_loader) -> None:
        """
        Compute centroid and std of normal data representations.
        Call this after training on normal data only.
        """
        self.eval()
        embeddings = []
        with torch.no_grad():
            for batch in normal_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                embeddings.append(self.encoder(x))
        all_emb = torch.cat(embeddings, dim=0)
        self.normal_centroid = all_emb.mean(dim=0)
        self.normal_std = all_emb.std(dim=0).clamp(min=1e-6)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score as Mahalanobis-like distance from normal centroid.
        Higher score = more anomalous.
        """
        self.eval()
        with torch.no_grad():
            emb = self.encoder(x)
            # Standardized Euclidean distance from normal centroid
            diff = (emb - self.normal_centroid) / self.normal_std
            score = diff.pow(2).sum(dim=-1).sqrt()
        return score


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def train_contrastive(model: ContrastiveAnomalyDetector,
                      train_loader,
                      optimizer: torch.optim.Optimizer,
                      device: torch.device,
                      epochs: int = 50) -> List[float]:
    """
    Train contrastive model on normal (unlabeled) time-series data.

    Returns:
        losses: List of per-epoch average losses
    """
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            optimizer.zero_grad()
            z1, z2 = model(x)
            loss = model.nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Contrastive Loss: {avg_loss:.4f}")

    return losses
