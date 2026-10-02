"""
train.py

Trains the dual-head CrowdFlow LSTM on the windowed sequences produced by
prepare_sequences.py.

Loss = MSE(regression) + CLS_WEIGHT * weighted_CrossEntropy(classification)

Classification uses inverse-frequency class weights because bunching
events (High risk) are the minority class (~8% of stops) but are the
class we most need the model to catch -- unweighted cross-entropy would
let the model get away with mostly predicting "Low" and still score well.

Usage:
    python train.py --epochs 30 --batch-size 128 --lr 1e-3
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lstm_model import CrowdFlowLSTM

CLS_LOSS_WEIGHT = 0.5  # relative weight of classification loss vs regression MSE


class SequenceDataset(Dataset):
    def __init__(self, X, y_reg, y_cls):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_cls[idx]


def compute_feature_scaler(X_train):
    """StandardScaler fit on TRAIN ONLY, per feature (last axis)."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0  # avoid divide-by-zero for constant features
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(X, mean, std):
    return (X - mean) / std


def compute_class_weights(y_cls, n_classes=3):
    counts = np.bincount(y_cls.flatten(), minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, cls_weights):
    model.eval()
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(weight=cls_weights.to(device))
    total_reg_mae, total_correct, total_cls_preds, n_batches = 0.0, 0, 0, 0

    with torch.no_grad():
        for X, y_reg, y_cls in loader:
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
            reg_out, cls_out = model(X)

            total_reg_mae += torch.abs(reg_out - y_reg).mean().item()
            preds = cls_out.argmax(dim=-1)
            total_correct += (preds == y_cls).sum().item()
            total_cls_preds += y_cls.numel()
            n_batches += 1

    return total_reg_mae / n_batches, total_correct / total_cls_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seq-dir", type=str, default="../data/sequences")
    parser.add_argument("--out-dir", type=str, default="../data/model")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data = np.load(os.path.join(args.seq_dir, "train_sequences.npz"))
    val_data = np.load(os.path.join(args.seq_dir, "val_sequences.npz"))
    X_train, y_reg_train, y_cls_train = train_data["X"], train_data["y_reg"], train_data["y_cls"]
    X_val, y_reg_val, y_cls_val = val_data["X"], val_data["y_reg"], val_data["y_cls"]

    mean, std = compute_feature_scaler(X_train)
    X_train = apply_scaler(X_train, mean, std)
    X_val = apply_scaler(X_val, mean, std)

    cls_weights = compute_class_weights(y_cls_train)
    print(f"Class weights (Low/Med/High): {cls_weights.tolist()}")

    train_loader = DataLoader(SequenceDataset(X_train, y_reg_train, y_cls_train),
                               batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_reg_val, y_cls_val),
                             batch_size=args.batch_size, shuffle=False)

    n_features = X_train.shape[-1]
    horizon = y_reg_train.shape[-1]
    model = CrowdFlowLSTM(n_features=n_features, hidden_size=args.hidden_size,
                           horizon=horizon).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(weight=cls_weights.to(device))

    best_val_mae = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y_reg, y_cls in train_loader:
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)

            optimizer.zero_grad()
            reg_out, cls_out = model(X)

            loss_reg = mse_loss(reg_out, y_reg)
            # cls_out: (batch, horizon, n_classes) -> flatten horizon into batch dim
            loss_cls = ce_loss(cls_out.reshape(-1, cls_out.shape[-1]), y_cls.reshape(-1))
            loss = loss_reg + CLS_LOSS_WEIGHT * loss_cls

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_mae, val_acc = evaluate(model, val_loader, device, cls_weights)
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_mae_min": val_mae, "val_cls_acc": val_acc})
        print(f"Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_MAE={val_mae:.3f} min  val_cls_acc={val_acc:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))

    # save scaler + config alongside the model so inference can reproduce preprocessing
    np.savez(os.path.join(args.out_dir, "scaler.npz"), mean=mean, std=std)
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out_dir, "model_config.json"), "w") as f:
        json.dump({"n_features": n_features, "horizon": horizon,
                    "hidden_size": args.hidden_size, "n_classes": 3}, f, indent=2)

    print(f"\nBest val MAE: {best_val_mae:.3f} min -- model saved to {args.out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
