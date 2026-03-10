# Self-Supervised Learning for Anomaly Detection in Time Series

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Research Contribution**: Train anomaly detectors on **unlabeled normal data** only.  
> No expensive annotation needed — just feed in normal operational data and the model learns what "normal" looks like.

---

## Overview

This repository implements three families of **self-supervised learning** methods for detecting anomalies in multivariate time-series signals — without requiring any labeled anomaly data during training.

| Model | Mechanism | Best For |
|-------|-----------|----------|
| **Variational Autoencoder (VAE)** | Reconstruction-based | Smooth signals, IoT sensors |
| **Convolutional Autoencoder** | Dilated Conv1D reconstruction | Long sequences, fast inference |
| **Contrastive Learning (SimCLR)** | NT-Xent contrastive loss | Representation learning |
| **Transformer MAE** | Masked patch reconstruction | Complex multi-variate patterns |

### Key Research Contributions

- **Label-free training**: Models train exclusively on normal (unlabeled) data
- **Self-supervised pretext tasks**: MAE masking, contrastive augmentation pairs
- **Unified evaluation**: AUROC, AUPRC, F1 + point-adjust protocol
- **Domain-agnostic**: Tested on cybersecurity, IoT, and financial use cases

---

## Architecture Details

### 1. Variational Autoencoder (VAE)
Uses an LSTM encoder/decoder with reparameterization trick. Learns a compact latent space of normal behavior. Anomaly score = reconstruction MSE + weighted KL divergence.

```
Input → LSTM Encoder → μ, σ → Reparameterize → z → LSTM Decoder → Reconstruction
Anomaly Score = MSE(input, reconstruction) + β × KL(q(z|x) || p(z))
```

### 2. Convolutional Autoencoder  
Uses **dilated 1D convolutions** with dilation rates [1, 2, 4] to capture multi-scale temporal patterns. AdaptiveAvgPool for flexible sequence lengths. Faster than LSTM-based models.

### 3. Contrastive Learning (SimCLR-style)
Applies two random augmentations to each window to create positive pairs. Trains a Transformer encoder with NT-Xent loss to pull augmented views together while pushing different windows apart. At inference, measures Mahalanobis-like distance from the centroid of normal embeddings.

**Augmentation strategies:**
- Gaussian noise injection
- Temporal masking (random time-step zeroing)
- Amplitude scaling
- Temporal jitter (circular shift)

### 4. Transformer Masked Autoencoder (MAE)
Inspired by MAE (He et al., 2022) adapted for time series. Splits sequences into patches, randomly masks 40% of patches, and trains the Transformer to reconstruct masked patches from visible context. High reconstruction error at inference = anomaly.

```
Input → Patch Tokenizer → Mask 40% → Transformer Encoder → Decoder → Reconstruct Masked Patches
```

---

## Project Structure

```
self-supervised-anomaly-detection/
├── models/
│   ├── autoencoder.py           # VAE + Convolutional Autoencoder
│   ├── contrastive.py           # SimCLR contrastive detector
│   └── transformer_detector.py  # Masked Autoencoding Transformer
├── utils/
│   ├── data_utils.py            # Sliding windows, preprocessing, synthetic data
│   └── metrics.py               # AUROC, AUPRC, F1, point-adjust evaluation
├── train.py                     # Unified training + evaluation script
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/PranayMahendrakar/self-supervised-anomaly-detection.git
cd self-supervised-anomaly-detection
pip install -r requirements.txt
```

### Train on Synthetic Data (No dataset needed)

```bash
# Transformer MAE (recommended for complex signals)
python train.py --model transformer --epochs 100 --window_size 64

# VAE with LSTM encoder/decoder
python train.py --model vae --epochs 150 --latent_dim 16 --hidden_dim 64

# Convolutional Autoencoder (fastest)
python train.py --model conv --epochs 100 --window_size 128

# Contrastive Learning (SimCLR-style)
python train.py --model contrastive --epochs 200 --embed_dim 64 --temperature 0.07
```

### Train on Your Own Data

```bash
# Your data should be NumPy arrays: (T, features) for data, (T,) for labels
python train.py \
    --model transformer \
    --data_path data/your_data.npy \
    --labels_path data/your_labels.npy \  # Optional for evaluation
    --window_size 64 \
    --epochs 100 \
    --batch_size 64
```

### Python API

```python
import numpy as np
import torch
from models.transformer_detector import AnomalyTransformer
from utils.data_utils import TimeSeriesPreprocessor, create_inference_loader
from utils.metrics import evaluate

# Load your time series
data = np.load("sensor_data.npy")  # shape: (T, features)

# Preprocess
preprocessor = TimeSeriesPreprocessor(scaler_type='standard')
data_norm = preprocessor.fit_transform(data[:7000])  # fit on normal data
test_data = preprocessor.transform(data[7000:])

# Build and train model
model = AnomalyTransformer(input_dim=5, seq_len=64, d_model=128)
# ... (training loop)

# Compute anomaly scores on test data
loader = create_inference_loader(test_data, window_size=64)
scores = []
model.eval()
with torch.no_grad():
    for x in loader:
        scores.extend(model.anomaly_score(x).numpy())

# Evaluate (if labels available)
test_labels = np.load("test_labels.npy")
results = evaluate(test_labels, np.array(scores))
print(results)
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUROC** | Area Under ROC Curve — threshold-free ranking metric |
| **AUPRC** | Area Under Precision-Recall Curve — better for imbalanced data |
| **Best F1** | Maximum F1 across all threshold candidates |
| **PA-F1** | Point-Adjusted F1 — standard protocol in time-series anomaly detection |

**Point-Adjust (PA) Protocol**: If any prediction in an anomaly segment is detected, the whole segment is counted as detected. This accounts for temporal displacement between the labeled anomaly onset and detector firing.

---

## Applications

### Cybersecurity
Detect unusual network traffic patterns, zero-day intrusions, and insider threats from network flow time series — without needing labeled attack data.

### Industrial IoT Monitoring
Monitor sensor readings from manufacturing equipment, detect bearing failures, overheating, or pressure anomalies in real time using only normal operational data.

### Financial Fraud Detection
Identify unusual transaction patterns, flash crashes, or account takeover behaviors in financial time series without extensive fraud labeling.

---

## Hyperparameter Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--window_size` | 64 | Sliding window length (should cover 1-2 anomaly cycles) |
| `--stride` | 1 | Window stride (1 = max overlap, window_size = no overlap) |
| `--mask_ratio` | 0.4 | Fraction of patches masked in Transformer MAE |
| `--temperature` | 0.07 | NT-Xent temperature (lower = harder negatives) |
| `--latent_dim` | 16 | VAE bottleneck size (smaller = stronger compression) |
| `--beta` | 1.0 | Beta-VAE KL weight (>1 = more disentangled) |
| `--embed_dim` | 64 | Transformer embedding dimension |

---

## References

1. He, K. et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.
2. Chen, T. et al. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.
3. Xu, J. et al. "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." NeurIPS 2022.
4. Nie, Y. et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023.
5. Kingma, D.P. & Welling, M. "Auto-Encoding Variational Bayes." ICLR 2014.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with PyTorch | Self-Supervised Learning | Time Series Analysis*
