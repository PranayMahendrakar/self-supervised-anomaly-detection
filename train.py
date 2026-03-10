"""
Main Training Script
=====================
Train any of the three self-supervised anomaly detection models:
  - vae:         Variational Autoencoder (LSTM-based)
  - conv:        Convolutional Autoencoder (dilated Conv1D)
  - contrastive: SimCLR Contrastive Learning
  - transformer: Masked Autoencoding Transformer (MAE-style)

Usage:
    python train.py --model vae --data_path data/train.npy --epochs 100
    python train.py --model transformer --window_size 128 --batch_size 32
    python train.py --model contrastive --embed_dim 64 --temperature 0.07

No labels required for training! Only normal data is needed.
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.autoencoder import VariationalAutoencoder, ConvAutoencoder, vae_loss
from models.contrastive import ContrastiveAnomalyDetector, train_contrastive
from models.transformer_detector import AnomalyTransformer, train_transformer
from utils.data_utils import (
    TimeSeriesPreprocessor, create_dataloaders, create_inference_loader,
    SyntheticAnomalyGenerator, train_test_split_time_series
)
from utils.metrics import evaluate, print_evaluation_report


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def build_model(args, input_dim: int, device: torch.device):
    """Factory function to build the selected model."""
    model_type = args.model.lower()

    if model_type == 'vae':
        model = VariationalAutoencoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            seq_len=args.window_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        print(f"[VAE] input_dim={input_dim}, hidden_dim={args.hidden_dim}, "
              f"latent_dim={args.latent_dim}")

    elif model_type == 'conv':
        model = ConvAutoencoder(input_dim=input_dim, seq_len=args.window_size)
        print(f"[ConvAE] input_dim={input_dim}, seq_len={args.window_size}")

    elif model_type == 'contrastive':
        model = ContrastiveAnomalyDetector(
            input_dim=input_dim,
            embed_dim=args.embed_dim,
            proj_dim=args.proj_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            temperature=args.temperature,
        )
        print(f"[Contrastive] input_dim={input_dim}, embed_dim={args.embed_dim}, "
              f"temperature={args.temperature}")

    elif model_type == 'transformer':
        model = AnomalyTransformer(
            input_dim=input_dim,
            seq_len=args.window_size,
            patch_size=args.patch_size,
            d_model=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            mask_ratio=args.mask_ratio,
            dropout=args.dropout,
        )
        print(f"[Transformer-MAE] input_dim={input_dim}, seq_len={args.window_size}, "
              f"patch_size={args.patch_size}, mask_ratio={args.mask_ratio}")
    else:
        raise ValueError(f"Unknown model: {model_type}. Choose from: vae, conv, contrastive, transformer")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    return model.to(device)


def load_data(args):
    """Load and preprocess time-series data."""
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading data from {args.data_path}...")
        data = np.load(args.data_path)
        labels = np.load(args.labels_path) if args.labels_path else None
    else:
        print("No data path provided. Generating synthetic dataset...")
        gen = SyntheticAnomalyGenerator(seed=42)
        data, labels = gen.generate_dataset(
            length=10000,
            features=args.input_dim or 5,
            anomaly_ratio=0.05,
        )
        print(f"Generated dataset: {data.shape}, anomaly rate: {labels.mean():.2%}")

    # Temporal train/test split
    if labels is not None:
        train_data, test_data, train_labels, test_labels = train_test_split_time_series(
            data, labels, train_ratio=args.train_ratio
        )
    else:
        train_data, test_data = train_test_split_time_series(
            data, train_ratio=args.train_ratio
        )
        train_labels = test_labels = None

    # Fit preprocessor on training data only (prevents data leakage)
    preprocessor = TimeSeriesPreprocessor(scaler_type=args.scaler)
    train_data = preprocessor.fit_transform(train_data)
    test_data = preprocessor.transform(test_data)

    print(f"Train: {train_data.shape}, Test: {test_data.shape}")
    if test_labels is not None:
        print(f"Test anomaly rate: {test_labels.mean():.2%}")

    return train_data, test_data, train_labels, test_labels, preprocessor


def train_model(model, model_type: str, train_loader, val_loader,
                 args, device: torch.device):
    """Train the model using the appropriate training loop."""

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if model_type == 'transformer':
        history = train_transformer(model, train_loader, optimizer, scheduler,
                                     device, args.epochs)

    elif model_type == 'contrastive':
        history = {'train_loss': train_contrastive(model, train_loader, optimizer,
                                                    device, args.epochs)}
        # Calibrate on validation data after training
        print("Calibrating contrastive model on normal data...")
        model.calibrate(val_loader)

    else:
        # VAE and ConvAE training loops
        history = {'train_loss': [], 'val_loss': []}
        model.train()

        for epoch in range(args.epochs):
            # Training
            total_loss = 0.0
            model.train()
            for batch in train_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                optimizer.zero_grad()

                if model_type == 'vae':
                    recon, mu, logvar = model(x)
                    loss = vae_loss(recon, x, mu, logvar, beta=args.beta)
                else:
                    recon = model(x)
                    loss = torch.nn.functional.mse_loss(recon, x)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            # Validation
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    x = x.to(device)
                    if model_type == 'vae':
                        recon, mu, logvar = model(x)
                        val_loss += vae_loss(recon, x, mu, logvar).item()
                    else:
                        recon = model(x)
                        val_loss += torch.nn.functional.mse_loss(recon, x).item()

            scheduler.step()
            avg_train = total_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] | "
                      f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

    return history


def evaluate_model(model, model_type: str, test_data, test_labels,
                    args, device: torch.device):
    """Run inference and evaluate on test data."""
    if test_labels is None:
        print("No test labels available — skipping evaluation")
        return None

    inference_loader = create_inference_loader(
        test_data,
        window_size=args.window_size,
        stride=args.eval_stride,
        labels=test_labels,
        batch_size=args.batch_size * 2,
    )

    # Collect scores and labels
    all_scores = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in inference_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                all_labels.extend(y.numpy().flatten())
            else:
                x = batch
            x = x.to(device)
            scores = model.anomaly_score(x)
            all_scores.extend(scores.cpu().numpy())

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    results = print_evaluation_report(labels_arr, scores_arr, model_type.upper())
    return results


def save_checkpoint(model, args, history, results=None):
    """Save model checkpoint and training metadata."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Save model weights
    ckpt_path = os.path.join(args.output_dir, f"{args.model}_checkpoint.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'history': history,
    }, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    # Save evaluation results
    if results is not None:
        results_path = os.path.join(args.output_dir, f"{args.model}_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'auroc': results.auroc,
                'auprc': results.auprc,
                'best_f1': results.best_f1,
                'pa_f1': results.pa_f1,
            }, f, indent=2)
        print(f"Results saved: {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-Supervised Anomaly Detection for Time Series"
    )

    # Model selection
    parser.add_argument('--model', type=str, default='transformer',
                         choices=['vae', 'conv', 'contrastive', 'transformer'],
                         help="Model architecture to train")

    # Data
    parser.add_argument('--data_path', type=str, default=None,
                         help="Path to .npy training data (T, features)")
    parser.add_argument('--labels_path', type=str, default=None,
                         help="Path to .npy labels (T,) — optional for training")
    parser.add_argument('--input_dim', type=int, default=None,
                         help="Input feature dimension (auto-detected if not set)")
    parser.add_argument('--train_ratio', type=float, default=0.7,
                         help="Fraction of data for training")
    parser.add_argument('--scaler', type=str, default='standard',
                         choices=['standard', 'minmax'])

    # Windowing
    parser.add_argument('--window_size', type=int, default=64,
                         help="Sliding window size (time steps)")
    parser.add_argument('--stride', type=int, default=1,
                         help="Window stride for training")
    parser.add_argument('--eval_stride', type=int, default=1,
                         help="Window stride for evaluation")

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=64,
                         help="LSTM hidden dimension (VAE)")
    parser.add_argument('--latent_dim', type=int, default=16,
                         help="Latent space dimension (VAE)")
    parser.add_argument('--embed_dim', type=int, default=64,
                         help="Transformer/contrastive embedding dimension")
    parser.add_argument('--proj_dim', type=int, default=32,
                         help="Contrastive projection dimension")
    parser.add_argument('--num_heads', type=int, default=4,
                         help="Attention heads (Transformer/contrastive)")
    parser.add_argument('--num_layers', type=int, default=2,
                         help="Number of encoder layers")
    parser.add_argument('--patch_size', type=int, default=8,
                         help="Patch size for Transformer MAE")
    parser.add_argument('--mask_ratio', type=float, default=0.4,
                         help="Masking ratio for MAE")
    parser.add_argument('--temperature', type=float, default=0.07,
                         help="NT-Xent temperature (contrastive)")
    parser.add_argument('--beta', type=float, default=1.0,
                         help="KL weight for beta-VAE")

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                         help="Directory to save checkpoints and results")
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()

    # Load and preprocess data
    train_data, test_data, train_labels, test_labels, preprocessor = load_data(args)

    input_dim = train_data.shape[1]

    # Create data loaders (normal data only for training)
    train_loader, val_loader = create_dataloaders(
        train_data,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        labels=train_labels,
        num_workers=0,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    model = build_model(args, input_dim, device)

    # Train
    print(f"\nStarting training: {args.model.upper()} for {args.epochs} epochs...")
    history = train_model(model, args.model, train_loader, val_loader, args, device)

    # Evaluate
    print("\nRunning evaluation on test set...")
    results = evaluate_model(model, args.model, test_data, test_labels, args, device)

    # Save
    save_checkpoint(model, args, history, results)
    print("\nDone!")


if __name__ == '__main__':
    main()
