"""
Transformer-based Anomaly Detector for Time Series
====================================================
Uses a masked autoencoding (MAE) pretext task to train a Transformer
without labels. Anomaly score = reconstruction error on masked tokens.

Architecture Inspiration:
- Masked Autoencoders (MAE) by He et al., 2022
- Anomaly Transformer (Xu et al., NeurIPS 2022)
- PatchTST (Nie et al., ICLR 2023)

Self-Supervised Pretext Task:
  Given a time-series window, randomly mask p% of patches.
  Train the Transformer to reconstruct masked patches from context.
  At inference: high reconstruction error on a window => anomaly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    Allows the model to generalize to sequence lengths unseen during training.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.shape[1]])


# ---------------------------------------------------------------------------
# Patch Tokenizer
# ---------------------------------------------------------------------------

class PatchTokenizer(nn.Module):
    """
    Splits a time-series into non-overlapping patches and projects to d_model.

    Patching reduces sequence length (improves attention efficiency) and
    forces the model to learn from local temporal context.

    Args:
        input_dim:  Number of channels (features) per time step
        patch_size: Number of time steps per patch
        d_model:    Output embedding dimension
    """

    def __init__(self, input_dim: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(input_dim * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # Pad sequence to be divisible by patch_size
        if T % self.patch_size != 0:
            pad = self.patch_size - (T % self.patch_size)
            x = F.pad(x, (0, 0, 0, pad))
            T = x.shape[1]
        # Reshape into patches: (B, num_patches, patch_size * C)
        num_patches = T // self.patch_size
        x = x.reshape(B, num_patches, self.patch_size * C)
        return self.proj(x)  # (B, num_patches, d_model)


# ---------------------------------------------------------------------------
# Anomaly Transformer with Masked Autoencoding
# ---------------------------------------------------------------------------

class AnomalyTransformer(nn.Module):
    """
    Masked Autoencoding Transformer for unsupervised anomaly detection.

    Self-supervised training:
      - Randomly mask 40% of patches
      - Reconstruct masked patches using unmasked context
      - Only normal data used during training

    Anomaly detection:
      - High reconstruction error on a window = potential anomaly
      - Point-level scores via patch-to-timestep interpolation

    Args:
        input_dim:      Number of input features
        seq_len:        Input sequence length
        patch_size:     Patch tokenization size (time steps per patch)
        d_model:        Transformer embedding dimension
        num_heads:      Multi-head attention heads
        num_layers:     Number of Transformer encoder layers
        mask_ratio:     Fraction of patches to mask during training
        dropout:        Dropout rate
    """

    def __init__(self, input_dim: int, seq_len: int = 128, patch_size: int = 8,
                 d_model: int = 128, num_heads: int = 4, num_layers: int = 4,
                 mask_ratio: float = 0.40, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        num_patches = math.ceil(seq_len / patch_size)
        self.num_patches = num_patches

        # Patch tokenizer
        self.tokenizer = PatchTokenizer(input_dim, patch_size, d_model)

        # Learned mask token (replaces masked patches during encoding)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=num_patches + 1, dropout=dropout)

        # Transformer encoder (processes visible + mask tokens)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Reconstruction head: project back to original patch space
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, input_dim * patch_size),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches for MAE pretext task.

        Returns:
            x_masked:    Tokens with mask tokens inserted at masked positions
            mask:        Boolean mask (True = masked)
            ids_restore: Indices to restore original patch order
        """
        B, N, D = x.shape
        num_mask = int(self.mask_ratio * N)

        # Random permutation to select which patches to mask
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # First num_mask indices are masked
        ids_keep = ids_shuffle[:, num_mask:]
        ids_mask = ids_shuffle[:, :num_mask]

        # Build masked sequence: visible patches + mask tokens
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        x_masked = torch.cat([x_visible, mask_tokens], dim=1)

        # Restore original order
        x_masked = torch.gather(
            x_masked, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        # Boolean mask: True where patch was masked
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_mask, True)

        return x_masked, mask, ids_restore

    def forward(self, x: torch.Tensor, mask: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x:    Input (B, T, C)
            mask: If True, apply random masking (training). False = full reconstruction.

        Returns:
            recon: Reconstructed patches (B, num_patches, patch_size * input_dim)
            patch_mask: Boolean mask used (None if mask=False)
        """
        # Tokenize into patches
        tokens = self.tokenizer(x)               # (B, N, d_model)
        tokens = self.pos_enc(tokens)

        if mask and self.training:
            tokens_masked, patch_mask, _ = self.random_masking(tokens)
        else:
            tokens_masked = tokens
            patch_mask = None

        # Transformer encoding
        encoded = self.transformer(tokens_masked)  # (B, N, d_model)

        # Reconstruction
        recon = self.decoder(encoded)              # (B, N, patch_size * input_dim)

        return recon, patch_mask

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Masked reconstruction loss for training.
        Only compute loss on masked patches (not the visible ones).
        """
        tokens_orig = self.tokenizer(x)           # Target patches
        recon, patch_mask = self.forward(x, mask=True)

        # Ground truth: reshape x into patches
        B, T, C = x.shape
        if T % self.patch_size != 0:
            pad = self.patch_size - (T % self.patch_size)
            x_padded = F.pad(x, (0, 0, 0, pad))
        else:
            x_padded = x
        N = x_padded.shape[1] // self.patch_size
        target = x_padded.reshape(B, N, self.patch_size * C)

        if patch_mask is None:
            return F.mse_loss(recon, target)

        # MSE only on masked patches
        masked_recon = recon[patch_mask]
        masked_target = target[patch_mask]
        return F.mse_loss(masked_recon, masked_target)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample anomaly scores.

        Uses full reconstruction (no masking) to measure how well the
        model can reconstruct the input. Poor reconstruction = anomaly.
        """
        self.eval()
        with torch.no_grad():
            B, T, C = x.shape

            # Get full reconstruction
            recon_patches, _ = self.forward(x, mask=False)

            # Reconstruct back to time-series shape
            if T % self.patch_size != 0:
                pad = self.patch_size - (T % self.patch_size)
                x_padded = F.pad(x, (0, 0, 0, pad))
            else:
                x_padded = x
            N = x_padded.shape[1] // self.patch_size
            target_patches = x_padded.reshape(B, N, self.patch_size * C)

            # Per-sample MSE (average over patches and features)
            score = F.mse_loss(recon_patches, target_patches, reduction="none")
            score = score.mean(dim=(1, 2))

        return score

    def point_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get point-level anomaly scores by interpolating patch scores.
        Useful for pinpointing exact anomalous time steps.
        """
        self.eval()
        with torch.no_grad():
            B, T, C = x.shape
            recon_patches, _ = self.forward(x, mask=False)

            if T % self.patch_size != 0:
                pad = self.patch_size - (T % self.patch_size)
                x_padded = F.pad(x, (0, 0, 0, pad))
            else:
                x_padded = x

            N = x_padded.shape[1] // self.patch_size
            target_patches = x_padded.reshape(B, N, self.patch_size * C)

            # Per-patch score: (B, N)
            patch_scores = F.mse_loss(
                recon_patches, target_patches, reduction="none"
            ).mean(dim=-1)

            # Expand patch scores back to time steps: (B, T)
            point_scores = patch_scores.unsqueeze(1)
            point_scores = F.interpolate(point_scores, size=T, mode='nearest')
            return point_scores.squeeze(1)


# ---------------------------------------------------------------------------
# Association Discrepancy (Anomaly Transformer variant)
# ---------------------------------------------------------------------------

class AssociationDiscrepancy(nn.Module):
    """
    Computes the association discrepancy from Anomaly Transformer (Xu et al., 2022).

    Normal data has strong prior associations within a window.
    Anomalies break these associations, resulting in high discrepancy.

    This is used as a regularization term during training.
    """

    def __init__(self):
        super().__init__()

    def kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Symmetrized KL divergence between attention distributions."""
        p = p + 1e-8
        q = q + 1e-8
        kl_pq = (p * (p / q).log()).sum(-1)
        kl_qp = (q * (q / p).log()).sum(-1)
        return 0.5 * (kl_pq + kl_qp)

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn_weights: Attention weight matrix (B, H, T, T)
        Returns:
            discrepancy: Per-sample association discrepancy score (B,)
        """
        B, H, T, _ = attn_weights.shape
        # Series association: row-wise softmax (already done by attention)
        series_assoc = attn_weights.mean(dim=1)  # (B, T, T)

        # Prior association: Gaussian kernel centered at each time step
        idx = torch.arange(T, device=attn_weights.device).float()
        prior = torch.exp(-0.5 * (idx.unsqueeze(0) - idx.unsqueeze(1)).pow(2) / (T / 4) ** 2)
        prior = prior / prior.sum(dim=-1, keepdim=True)
        prior = prior.unsqueeze(0).expand(B, -1, -1)

        discrepancy = self.kl_divergence(series_assoc, prior).mean(dim=-1)  # (B,)
        return discrepancy


# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train_transformer(model: AnomalyTransformer,
                      train_loader,
                      optimizer: torch.optim.Optimizer,
                      scheduler,
                      device: torch.device,
                      epochs: int = 100) -> dict:
    """
    Train Transformer MAE on normal time-series data (no labels needed).

    Args:
        model:        AnomalyTransformer model
        train_loader: DataLoader with normal time-series windows
        optimizer:    AdamW recommended
        scheduler:    LR scheduler (CosineAnnealingLR recommended)
        device:       Training device
        epochs:       Number of training epochs

    Returns:
        history: Dict with 'train_loss' list
    """
    model.to(device)
    history = {'train_loss': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            optimizer.zero_grad()
            loss = model.reconstruction_loss(x)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | MAE Loss: {avg_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    return history
