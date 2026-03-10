"""
Autoencoder-based Anomaly Detection for Time Series
=====================================================
Trains a reconstruction-based autoencoder on normal data.
Anomalies are detected when reconstruction error exceeds a threshold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TimeSeriesEncoder(nn.Module):
    """LSTM-based encoder that compresses time-series into a latent representation."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(x)
        h = self.norm(h_n[-1])
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class TimeSeriesDecoder(nn.Module):
    """LSTM-based decoder that reconstructs time-series from latent representation."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 seq_len: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(z)).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(h)
        return self.output_layer(out)


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for time-series anomaly detection.

    The VAE learns a probabilistic latent space from normal data.
    At inference, high reconstruction error flags anomalies.

    Args:
        input_dim:  Number of input features per time step
        hidden_dim: LSTM hidden state size
        latent_dim: Bottleneck dimensionality
        seq_len:    Length of input sequences
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16,
                 seq_len: int = 50, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = TimeSeriesEncoder(input_dim, hidden_dim, latent_dim,
                                         num_layers, dropout)
        self.decoder = TimeSeriesDecoder(latent_dim, hidden_dim, input_dim,
                                         seq_len, num_layers, dropout)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick enables gradient flow through sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval time

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores as weighted sum of reconstruction error + KL.
        Higher scores indicate more anomalous behavior.
        """
        self.eval()
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            recon_loss = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(), dim=1
            )
        return recon_loss + 0.01 * kl_loss

    def fit_threshold(self, normal_scores: torch.Tensor,
                       percentile: float = 95.0) -> float:
        """Determine anomaly threshold from normal data scores."""
        import numpy as np
        return float(np.percentile(normal_scores.cpu().numpy(), percentile))


class ConvAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for time-series anomaly detection.
    Uses dilated convolutions to capture multi-scale temporal patterns.
    Faster than LSTM-based models for long sequences.
    """

    def __init__(self, input_dim: int, seq_len: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(seq_len // 4),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, input_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, features, seq_len) for Conv1d
        x_t = x.permute(0, 2, 1)
        encoded = self.encoder(x_t)
        decoded = self.decoder(encoded)
        return decoded.permute(0, 2, 1)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            min_len = min(recon.shape[1], x.shape[1])
            score = F.mse_loss(
                recon[:, :min_len, :], x[:, :min_len, :], reduction="none"
            )
        return score.mean(dim=(1, 2))


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Beta-VAE ELBO loss.

    Args:
        beta: Weight for KL term. beta > 1 encourages disentanglement.
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
