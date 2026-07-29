#!/usr/bin/env python3
import os
import json
import argparse
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


def build_Xy_from_csv(df: pd.DataFrame, window: int = 100):
    df = df.copy()
    df["epi_delta"] = df["epi_delta"].apply(json.loads)
    df["busy_fraction"] = df["busy_fraction"].apply(json.loads)

    epi_lists = df["epi_delta"].tolist()
    busy_lists = df["busy_fraction"].tolist()
    labels = df["label"].astype(str).to_numpy()

    N = len(df)
    X = np.zeros((N, 2, window), dtype=np.float32)

    for i, (e, b) in enumerate(zip(epi_lists, busy_lists)):
        if not isinstance(e, list) or not isinstance(b, list) or len(e) != window or len(b) != window:
            raise ValueError(f"Bad row {i}: expected lists of length {window}.")
        X[i, 0, :] = np.asarray(e, dtype=np.float32)
        X[i, 1, :] = np.asarray(b, dtype=np.float32)

    return X, labels


class BetterCNN1D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        def block(cin, cout, k):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(cout),
                nn.ReLU(),
            )

        self.net = nn.Sequential(
            block(2, 64, 7),
            block(64, 64, 7),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),

            block(64, 128, 5),
            block(128, 128, 5),
            nn.MaxPool1d(2),
            nn.Dropout(0.20),

            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.fc(x)


def compute_norm_stats(X_train: np.ndarray):
    mu = X_train.mean(axis=(0, 2), keepdims=True)
    sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu, sd


def normalize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    w = counts.sum() / (counts + 1e-9)
    w = w / w.mean()
    return w.astype(np.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tot_loss, tot, correct = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        tot_loss += float(loss.item()) * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        tot += int(y.size(0))

    return tot_loss / max(tot, 1), correct / max(tot, 1)


@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    tot_loss, tot, correct = 0.0, 0, 0
    ys, ps, probs = [], [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)

        tot_loss += float(loss.item()) * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        tot += int(y.size(0))

        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
        probs.append(prob.cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(ps) if ps else np.array([], dtype=int)
    y_prob = np.concatenate(probs) if probs else np.empty((0, 0), dtype=float)

    return tot_loss / max(tot, 1), correct / max(tot, 1), y_true, y_pred, y_prob


def _draw_confmat(
    cm_values: np.ndarray,
    class_names,
    out_path: str,
    *,
    xlabel_fs: int = 22,
    ylabel_fs: int = 22,
    tick_fs: int = 16,
    annot_fs: int = 14,
    title_fs: int = 16,
    cbar_fs: int = 14,
    title: str | None = None,
    add_percent_sign: bool = True,
    hide_zero_text: bool = True,
    figsize=(11, 9),
    vmin: float = 0.0,
    vmax: float = 100.0,
):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    im = ax.imshow(cm_values, interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=cbar_fs)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(class_names, fontsize=tick_fs)

    ax.set_xlabel("Predicted", fontsize=xlabel_fs, fontweight="bold")
    ax.set_ylabel("True", fontsize=ylabel_fs, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=title_fs, fontweight="bold")

    for i in range(cm_values.shape[0]):
        for j in range(cm_values.shape[1]):
            val = cm_values[i, j]
            if hide_zero_text and abs(val) < 1e-12:
                continue
            txt = f"{val:.1f}%"
            if not add_percent_sign:
                txt = f"{val:.1f}"
            ax.text(
                j, i, txt,
                ha="center",
                va="center",
                fontsize=annot_fs,
                fontweight="bold",
                color="white"
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=400)
    plt.close(fig)


def save_confmat_out_of_100(
    cm_counts: np.ndarray,
    class_names,
    out_pdf: str,
    *,
    xlabel_fs: int = 22,
    ylabel_fs: int = 22,
    tick_fs: int = 16,
    annot_fs: int = 14,
    title_fs: int = 16,
    cbar_fs: int = 14,
):
    total = cm_counts.sum()
    cm_pct = (cm_counts / total * 100.0) if total > 0 else np.zeros_like(cm_counts, dtype=float)

    _draw_confmat(
        cm_pct,
        class_names,
        out_pdf,
        xlabel_fs=xlabel_fs,
        ylabel_fs=ylabel_fs,
        tick_fs=tick_fs,
        annot_fs=annot_fs,
        title_fs=title_fs,
        cbar_fs=cbar_fs,
        title="Confusion Matrix (percent of all test samples)",
        add_percent_sign=False,
        hide_zero_text=True,
        figsize=(11, 9),
    )


def save_confmat_percent_overall(
    cm_counts: np.ndarray,
    class_names,
    out_pdf: str,
    *,
    xlabel_fs: int = 22,
    ylabel_fs: int = 22,
    tick_fs: int = 16,
    annot_fs: int = 14,
    title_fs: int = 16,
    cbar_fs: int = 14,
):
    total = cm_counts.sum()
    cm_pct = (cm_counts / total * 100.0) if total > 0 else np.zeros_like(cm_counts, dtype=float)

    _draw_confmat(
        cm_pct,
        class_names,
        out_pdf,
        xlabel_fs=xlabel_fs,
        ylabel_fs=ylabel_fs,
        tick_fs=tick_fs,
        annot_fs=annot_fs,
        title_fs=title_fs,
        cbar_fs=cbar_fs,
        title="Confusion Matrix (overall %, sums to 100%)",
        add_percent_sign=True,
        hide_zero_text=True,
        figsize=(11, 9),
    )


def save_confmat_percent_row(
    cm_counts: np.ndarray,
    class_names,
    out_pdf: str,
    *,
    xlabel_fs: int = 22,
    ylabel_fs: int = 22,
    tick_fs: int = 16,
    annot_fs: int = 14,
    title_fs: int = 16,
    cbar_fs: int = 14,
):
    row_sums = cm_counts.sum(axis=1, keepdims=True).astype(float)
    cm_pct = np.divide(
        cm_counts,
        row_sums,
        out=np.zeros_like(cm_counts, dtype=float),
        where=row_sums > 0
    ) * 100.0

    _draw_confmat(
        cm_pct,
        class_names,
        out_pdf,
        xlabel_fs=xlabel_fs,
        ylabel_fs=ylabel_fs,
        tick_fs=tick_fs,
        annot_fs=annot_fs,
        title_fs=title_fs,
        cbar_fs=cbar_fs,
        title=None,
        add_percent_sign=True,
        hide_zero_text=True,
        figsize=(11, 9),
    )


def print_f1_and_auc(y_true, y_pred, y_prob, class_names):
    n_classes = len(class_names)

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_micro = f1_score(y_true, y_pred, average="micro")

    print(f"[FINAL] F1-score (macro):    {f1_macro:.4f}")
    print(f"[FINAL] F1-score (weighted): {f1_weighted:.4f}")
    print(f"[FINAL] F1-score (micro):    {f1_micro:.4f}")

    if n_classes == 2:
        auc_val = roc_auc_score(y_true, y_prob[:, 1])
        print(f"[FINAL] AUC: {auc_val:.4f}")
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

        auc_macro = roc_auc_score(
            y_true_bin, y_prob,
            average="macro",
            multi_class="ovr"
        )
        auc_weighted = roc_auc_score(
            y_true_bin, y_prob,
            average="weighted",
            multi_class="ovr"
        )
        auc_micro = roc_auc_score(
            y_true_bin, y_prob,
            average="micro"
        )

        print(f"[FINAL] AUC (macro, OvR):    {auc_macro:.4f}")
        print(f"[FINAL] AUC (weighted, OvR): {auc_weighted:.4f}")
        print(f"[FINAL] AUC (micro):         {auc_micro:.4f}")

        print("[FINAL] Per-class AUC:")
        for i, cls in enumerate(class_names):
            auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            print(f"  {cls}: {auc_i:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", default="dataset_seq100.csv")
    ap.add_argument("--out_dir", default="cnn_out_better")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--window", type=int, default=100)

    # Plot sizes
    ap.add_argument("--xlabel_fs", type=int, default=22)
    ap.add_argument("--ylabel_fs", type=int, default=22)
    ap.add_argument("--tick_fs", type=int, default=16)
    ap.add_argument("--annot_fs", type=int, default=14)
    ap.add_argument("--title_fs", type=int, default=16)
    ap.add_argument("--cbar_fs", type=int, default=14)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.dataset_csv)
    required = {"epi_delta", "busy_fraction", "label"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Dataset missing columns. Need {required}, got {set(df.columns)}")

    X, y_str = build_Xy_from_csv(df, window=args.window)

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_names = list(le.classes_)
    C = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    mu, sd = compute_norm_stats(X_train)
    X_train = normalize(X_train, mu, sd)
    X_test = normalize(X_test, mu, sd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetterCNN1D(num_classes=C).to(device)

    w = class_weights(y_train, C)
    w_t = torch.tensor(w, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w_t)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(
        SeqDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        SeqDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False
    )

    best_acc = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc, _, _, _ = eval_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{args.epochs} | train acc {tr_acc:.4f} | test acc {te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": class_names,
                    "mu": mu,
                    "sd": sd,
                },
                best_path
            )

    print(f"[OK] Best test accuracy: {best_acc:.4f}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    _, _, y_true, y_pred, y_prob = eval_model(model, test_loader, criterion, device)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n[FINAL] Test Accuracy: {acc:.4f}\n")

    print_f1_and_auc(y_true, y_pred, y_prob, class_names)

    print("\n[FINAL] Classification report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm_counts = confusion_matrix(y_true, y_pred, labels=np.arange(C))
    print("[FINAL] Confusion matrix (counts):\n", cm_counts)

    # Save raw confusion matrix
    cm_counts_df = pd.DataFrame(cm_counts, index=class_names, columns=class_names)
    cm_counts_csv = os.path.join(args.out_dir, "confusion_matrix_counts.csv")
    cm_counts_df.to_csv(cm_counts_csv)

    # Save normalized csv
    row_sums = cm_counts.sum(axis=1, keepdims=True).astype(float)
    cm_row_pct = np.divide(
        cm_counts,
        row_sums,
        out=np.zeros_like(cm_counts, dtype=float),
        where=row_sums > 0
    ) * 100.0
    cm_row_pct_df = pd.DataFrame(cm_row_pct, index=class_names, columns=class_names)
    cm_row_pct_csv = os.path.join(args.out_dir, "confmat_row_100pct.csv")
    cm_row_pct_df.to_csv(cm_row_pct_csv)

    out_pdf_1 = os.path.join(args.out_dir, "confusion_matrix_out_of_100.pdf")
    out_pdf_2 = os.path.join(args.out_dir, "confmat_overall_100pct.pdf")
    out_pdf_3 = os.path.join(args.out_dir, "confmat_row_100pct.pdf")
    out_png_3 = os.path.join(args.out_dir, "confmat_row_100pct.png")

    save_confmat_out_of_100(
        cm_counts,
        class_names,
        out_pdf_1,
        xlabel_fs=args.xlabel_fs,
        ylabel_fs=args.ylabel_fs,
        tick_fs=args.tick_fs,
        annot_fs=args.annot_fs,
        title_fs=args.title_fs,
        cbar_fs=args.cbar_fs,
    )

    save_confmat_percent_overall(
        cm_counts,
        class_names,
        out_pdf_2,
        xlabel_fs=args.xlabel_fs,
        ylabel_fs=args.ylabel_fs,
        tick_fs=args.tick_fs,
        annot_fs=args.annot_fs,
        title_fs=args.title_fs,
        cbar_fs=args.cbar_fs,
    )

    save_confmat_percent_row(
        cm_counts,
        class_names,
        out_pdf_3,
        xlabel_fs=args.xlabel_fs,
        ylabel_fs=args.ylabel_fs,
        tick_fs=args.tick_fs,
        annot_fs=args.annot_fs,
        title_fs=args.title_fs,
        cbar_fs=args.cbar_fs,
    )

    save_confmat_percent_row(
        cm_counts,
        class_names,
        out_png_3,
        xlabel_fs=args.xlabel_fs,
        ylabel_fs=args.ylabel_fs,
        tick_fs=args.tick_fs,
        annot_fs=args.annot_fs,
        title_fs=args.title_fs,
        cbar_fs=args.cbar_fs,
    )

    print(f"[OK] Saved best model: {best_path}")
    print(f"[OK] Saved counts CSV: {cm_counts_csv}")
    print(f"[OK] Saved row-% CSV: {cm_row_pct_csv}")
    print(f"[OK] Saved percent-of-all PDF: {out_pdf_1}")
    print(f"[OK] Saved overall-% PDF: {out_pdf_2}")
    print(f"[OK] Saved row-% PDF: {out_pdf_3}")
    print(f"[OK] Saved row-% PNG: {out_png_3}")


if __name__ == "__main__":
    main()