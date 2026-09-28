"""Train SmallCNN on the 45k train split. Never touches the test set.

Model selection uses calibration accuracy. Best checkpoint -> models/cnn_best.pt
(and a copy on Drive if drive_dir is given).
"""
import csv
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data import get_calib_loader, get_train_loader
from src.model import SmallCNN
from src.utils import get_device, set_seed


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass over a loader. Trains if optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def plot_curves(log_path, fig_path):
    import pandas as pd
    df = pd.read_csv(log_path)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(df.epoch, df.train_loss, label="train")
    ax[0].plot(df.epoch, df.calib_loss, label="calibration")
    ax[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax[1].plot(df.epoch, df.train_acc, label="train")
    ax[1].plot(df.epoch, df.calib_acc, label="calibration")
    ax[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def train(epochs=20, lr=1e-3, weight_decay=0.0, data_root="/content/data",
          ckpt_path="models/cnn_best.pt", log_path="results/metrics/training_log.csv",
          fig_path="results/figures/training_curves.png", drive_dir=None):
    set_seed(42)
    device = get_device()
    for p in (ckpt_path, log_path, fig_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    train_loader = get_train_loader(data_root)
    calib_loader = get_calib_loader(data_root)
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "calib_loss", "calib_acc"])
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
            ca_loss, ca_acc = run_epoch(model, calib_loader, device, criterion)
            writer.writerow([epoch, tr_loss, tr_acc, ca_loss, ca_acc])
            f.flush()
            saved = ""
            if ca_acc > best_acc:
                best_acc = ca_acc
                torch.save(model.state_dict(), ckpt_path)
                if drive_dir:
                    os.makedirs(f"{drive_dir}/models", exist_ok=True)
                    shutil.copy(ckpt_path, f"{drive_dir}/models/cnn_best.pt")
                saved = "  <- saved"
            print(f"epoch {epoch:2d} | train {tr_loss:.3f}/{tr_acc:.3f} | "
                  f"calib {ca_loss:.3f}/{ca_acc:.3f}{saved}")

    plot_curves(log_path, fig_path)
    if drive_dir:
        for p in (log_path, fig_path):
            dest = f"{drive_dir}/{os.path.dirname(p)}"
            os.makedirs(dest, exist_ok=True)
            shutil.copy(p, dest)

    # Degenerate-model check: reload best checkpoint, count predicted classes on calib
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in calib_loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
    counts = torch.bincount(torch.cat(preds), minlength=10)
    print("best calib acc:", round(best_acc, 4))
    print("predicted-class counts on calib:", counts.tolist())
    return best_acc
