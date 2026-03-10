"""
Data Utilities for Self-Supervised Time Series Anomaly Detection
=================================================================
Preprocessing, windowing, and dataset creation for time-series data
across cybersecurity, IoT, and financial fraud detection domains.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings


# ---------------------------------------------------------------------------
# Sliding Window Dataset
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """
    Converts a multivariate time series into overlapping sliding windows.

    Used for both training (normal data only) and inference (all data).
    Each window becomes one training sample for the anomaly detection model.

    Args:
        data:       Time series array of shape (T, features) or (T,)
        window_size: Number of time steps per window
        stride:      Step size between consecutive windows
        labels:      Optional anomaly labels (0=normal, 1=anomaly)
        normalize:   Whether to normalize within each window
    """

    def __init__(self, data: np.ndarray, window_size: int, stride: int = 1,
                 labels: Optional[np.ndarray] = None, normalize: bool = False):
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize

        # Create sliding windows
        self.windows, self.window_labels = self._create_windows(data, labels)

    def _create_windows(self, data: np.ndarray,
                         labels: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        T = len(data)
        windows = []
        window_labels = []

        for start in range(0, T - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = data[start:end].astype(np.float32)
            windows.append(window)

            if labels is not None:
                # Window is anomalous if any point in it is anomalous
                win_label = int(labels[start:end].max())
                window_labels.append(win_label)

        return np.array(windows), (np.array(window_labels) if window_labels else None)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        window = self.windows[idx]

        if self.normalize:
            # Per-window normalization (zero mean, unit variance)
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True) + 1e-8
            window = (window - mean) / std

        tensor = torch.FloatTensor(window)

        if self.window_labels is not None:
            return tensor, torch.LongTensor([self.window_labels[idx]])
        return tensor


class NormalOnlyDataset(SlidingWindowDataset):
    """
    Dataset containing ONLY normal (non-anomalous) windows.
    Used for self-supervised training — no labels required for basic use.
    """

    def __init__(self, data: np.ndarray, window_size: int, stride: int = 1,
                 labels: Optional[np.ndarray] = None, **kwargs):
        super().__init__(data, window_size, stride, labels, **kwargs)

        # If labels provided, filter to normal windows only
        if self.window_labels is not None:
            normal_mask = self.window_labels == 0
            self.windows = self.windows[normal_mask]
            self.window_labels = self.window_labels[normal_mask]
            filtered = normal_mask.sum()
            total = len(normal_mask)
            print(f"[NormalOnlyDataset] Kept {filtered}/{total} normal windows "
                  f"({100*filtered/total:.1f}%)")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TimeSeriesPreprocessor:
    """
    Comprehensive preprocessing pipeline for multivariate time series.

    Handles:
    - Missing value imputation
    - Outlier clipping (for training stability, not detection)
    - Normalization (StandardScaler / MinMaxScaler)
    - Train/val/test splitting with no data leakage
    """

    def __init__(self, scaler_type: str = 'standard', clip_quantile: float = 0.999):
        """
        Args:
            scaler_type:    'standard' (z-score) or 'minmax' ([0, 1])
            clip_quantile:  Clip extreme values above this quantile in training data
        """
        self.scaler_type = scaler_type
        self.clip_quantile = clip_quantile
        self.scaler = None
        self.clip_values = None

    def fit(self, train_data: np.ndarray) -> 'TimeSeriesPreprocessor':
        """Fit scaler on training data (normal data only)."""
        if train_data.ndim == 1:
            train_data = train_data.reshape(-1, 1)

        # Compute clip bounds from training data
        self.clip_values = (
            np.percentile(train_data, (1 - self.clip_quantile) * 100, axis=0),
            np.percentile(train_data, self.clip_quantile * 100, axis=0),
        )
        train_clipped = np.clip(train_data, self.clip_values[0], self.clip_values[1])

        # Fit scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

        self.scaler.fit(train_clipped)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted preprocessing to data (train or test)."""
        if self.scaler is None:
            raise RuntimeError("Call fit() before transform()")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data_clipped = np.clip(data, self.clip_values[0], self.clip_values[1])
        return self.scaler.transform(data_clipped)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)


# ---------------------------------------------------------------------------
# Domain-Specific Data Loaders
# ---------------------------------------------------------------------------

def create_dataloaders(
    data: np.ndarray,
    window_size: int = 64,
    stride: int = 1,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    labels: Optional[np.ndarray] = None,
    num_workers: int = 0,
    normalize_windows: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from time-series data.

    For self-supervised training: only normal windows are used.
    If labels=None, ALL windows are used (assumes input is normal data).

    Args:
        data:        Time series (T, features)
        window_size: Sliding window size
        stride:      Window stride (1 = maximum overlap, window_size = no overlap)
        batch_size:  Mini-batch size
        val_ratio:   Fraction of data to use for validation
        labels:      Optional point labels (0=normal, 1=anomaly)

    Returns:
        train_loader, val_loader
    """
    dataset = NormalOnlyDataset(data, window_size, stride, labels, normalize=normalize_windows)
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def create_inference_loader(
    data: np.ndarray,
    window_size: int = 64,
    stride: int = 1,
    batch_size: int = 128,
    labels: Optional[np.ndarray] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create DataLoader for inference (anomaly scoring) on new data.
    Uses all windows regardless of labels.
    """
    dataset = SlidingWindowDataset(data, window_size, stride, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Synthetic Data Generation (for testing/demos)
# ---------------------------------------------------------------------------

class SyntheticAnomalyGenerator:
    """
    Generates synthetic time-series with ground-truth anomaly labels.
    Useful for evaluating detector performance without real datasets.

    Anomaly types:
        - point:   Sudden spike at a single time step
        - contextual: Normal value at abnormal context (pattern break)
        - collective: Subsequence that deviates from normal
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_normal(self, length: int, features: int = 1,
                         noise_std: float = 0.1) -> np.ndarray:
        """Generate normal time series (sinusoidal + noise)."""
        t = np.linspace(0, 4 * np.pi, length)
        data = np.zeros((length, features))
        for i in range(features):
            freq = self.rng.uniform(0.5, 2.0)
            phase = self.rng.uniform(0, 2 * np.pi)
            data[:, i] = np.sin(freq * t + phase) + self.rng.normal(0, noise_std, length)
        return data.astype(np.float32)

    def inject_point_anomalies(self, data: np.ndarray,
                                n_anomalies: int = 10,
                                magnitude: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """Inject point anomalies as sudden spikes."""
        data = data.copy()
        labels = np.zeros(len(data), dtype=np.int32)
        indices = self.rng.choice(len(data), size=n_anomalies, replace=False)
        for idx in indices:
            data[idx] += self.rng.choice([-1, 1]) * magnitude
            labels[idx] = 1
        return data, labels

    def inject_collective_anomalies(self, data: np.ndarray,
                                     n_anomalies: int = 3,
                                     duration: int = 10,
                                     magnitude: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Inject collective anomalies as sustained shifts."""
        data = data.copy()
        labels = np.zeros(len(data), dtype=np.int32)
        starts = self.rng.choice(len(data) - duration, size=n_anomalies, replace=False)
        for start in starts:
            end = start + duration
            data[start:end] += magnitude * self.rng.choice([-1, 1])
            labels[start:end] = 1
        return data, labels

    def generate_dataset(self, length: int = 5000, features: int = 1,
                          anomaly_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete dataset with mixed anomaly types.

        Returns:
            data:   Time series (length, features)
            labels: Point-level labels (0=normal, 1=anomaly)
        """
        data = self.generate_normal(length, features)
        n_point = int(length * anomaly_ratio * 0.6)
        n_collective = max(1, int(length * anomaly_ratio * 0.4) // 10)
        data, labels = self.inject_point_anomalies(data, n_anomalies=n_point)
        data, labels2 = self.inject_collective_anomalies(data, n_anomalies=n_collective)
        labels = np.maximum(labels, labels2)
        return data, labels


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def train_test_split_time_series(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    train_ratio: float = 0.7,
) -> Tuple:
    """
    Temporally split time series (no shuffling to preserve temporal order).

    Train set: assumed normal (for self-supervised learning).
    Test set:  may contain anomalies (for evaluation).
    """
    split = int(len(data) * train_ratio)
    train_data = data[:split]
    test_data = data[split:]
    if labels is not None:
        return train_data, test_data, labels[:split], labels[split:]
    return train_data, test_data


def get_window_labels(point_labels: np.ndarray, window_size: int,
                       stride: int = 1) -> np.ndarray:
    """
    Convert point-level labels to window-level labels.
    A window is anomalous if any point within it is anomalous.
    """
    T = len(point_labels)
    window_labels = []
    for start in range(0, T - window_size + 1, stride):
        window_labels.append(int(point_labels[start:start + window_size].max()))
    return np.array(window_labels)
