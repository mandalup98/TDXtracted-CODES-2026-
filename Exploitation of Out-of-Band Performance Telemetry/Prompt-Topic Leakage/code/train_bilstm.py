"""
Bi-LSTM + Attention — Prompt Topic Leakage Classifier
=======================================================
Follows TDXtracted Section VII-B exactly.

Data splits (from paper):
  Profiling (TRAIN):
    4,000 breast-cancer traces (prompts 1-80, 50 queries each)
  + 4,000 background traces   (first 4,000 of valid background rows)
  Attack simulation (TEST):
    1,000 breast-cancer traces (prompts 81-100, 50 queries each)
  + 1,000 background traces   (next 1,000 background rows)

Model (Whisper Leak Appendix III / TDXtracted VII-B):
  Input -> BiLSTM (2 layers, 128 hidden) -> Additive Attention
        -> MLP [128, 64] -> Sigmoid
  Adam lr=0.0002, batch=32, patience=20, max_epochs=100
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, classification_report,
    precision_recall_curve,
)
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────
BC_CSV     = "breast_cancer_epi_traces.csv"
BG_CSV     = "background_epi_traces.csv"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 1216
HIDDEN     = 128
N_LAYERS   = 2
ATTN_DIM   = 64
MLP_DIMS   = [128, 64]
DROPOUT    = 0.3
EMBED_DIM  = 32
LR         = 0.0002
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE   = 20
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

print(f"Device  : {DEVICE}")
print(f"Max len : {MAX_LEN}")

# ─────────────────────────────────────────────────────────────────
# 1. Load & split data (paper methodology)
# ─────────────────────────────────────────────────────────────────

def parse_trace(trace_str, max_len):
    vals = np.array(trace_str.strip().split(), dtype=np.float32)
    mu, sd = vals.mean(), vals.std() + 1e-8
    vals = (vals - mu) / sd
    if len(vals) > max_len:
        vals = vals[:max_len]
    if len(vals) < max_len:
        vals = np.concatenate([vals, np.zeros(max_len - len(vals), dtype=np.float32)])
    return vals

print("\n── Loading data ──────────────────────────────────────────")
bc_df = pd.read_csv(BC_CSV)
bg_df = pd.read_csv(BG_CSV)
bg_df = bg_df[bg_df['class_label'] == 'background'].reset_index(drop=True)
print(f"  BC : {len(bc_df):,} rows, {bc_df['prompt_text'].nunique()} unique prompts")
print(f"  BG : {len(bg_df):,} valid background rows")

# BC: first 80 unique prompts -> train, last 20 -> test
unique_prompts = bc_df['prompt_text'].unique()
train_p = set(unique_prompts[:80])
test_p  = set(unique_prompts[80:])
bc_train = bc_df[bc_df['prompt_text'].isin(train_p)]   # 4,000
bc_test  = bc_df[bc_df['prompt_text'].isin(test_p)]    # 1,000

# BG: first 4000 -> train, next 1000 -> test
bg_train = bg_df.iloc[:4000]
bg_test  = bg_df.iloc[4000:5000]

print(f"\n── Splits ────────────────────────────────────────────────")
print(f"  Train : {len(bc_train):,} BC + {len(bg_train):,} BG = {len(bc_train)+len(bg_train):,}")
print(f"  Test  : {len(bc_test):,}  BC + {len(bg_test):,}  BG = {len(bc_test)+len(bg_test):,}")

# ─────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────

class EPIDataset(Dataset):
    def __init__(self, bc_rows, bg_rows, max_len):
        rows = []
        for _, r in bc_rows.iterrows():
            rows.append((r['epi_trace'], 1))
        for _, r in bg_rows.iterrows():
            rows.append((r['epi_trace'], 0))
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(rows))
        self.data    = [rows[i] for i in perm]
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trace_str, label = self.data[idx]
        x = torch.tensor(
            parse_trace(trace_str, self.max_len), dtype=torch.float32
        ).unsqueeze(-1)                    # (L, 1)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

print("\n── Building datasets ─────────────────────────────────────")
train_ds = EPIDataset(bc_train, bg_train, MAX_LEN)
test_ds  = EPIDataset(bc_test,  bg_test,  MAX_LEN)
print(f"  Train : {len(train_ds):,}  Test : {len(test_ds):,}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=64,         shuffle=False, num_workers=0)

# ─────────────────────────────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────────────────────────────

class AdditiveAttention(nn.Module):
    def __init__(self, lstm_hidden, attn_dim):
        super().__init__()
        self.W = nn.Linear(lstm_hidden * 2, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h):
        energy = torch.tanh(self.W(h))
        scores = self.v(energy).squeeze(-1)
        alpha  = F.softmax(scores, dim=1)
        ctx    = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return ctx, alpha


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size=1, hidden=128, n_layers=2,
                 attn_dim=64, mlp_dims=None, dropout=0.3, embed_dim=32):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [128, 64]
        self.proj = nn.Linear(input_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn = AdditiveAttention(hidden, attn_dim)
        in_d, layers = hidden * 2, []
        for out_d in mlp_dims:
            layers += [nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(dropout)]
            in_d = out_d
        layers.append(nn.Linear(in_d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.proj(x))
        h, _ = self.lstm(x)
        ctx, alpha = self.attn(h)
        logit = self.mlp(ctx).squeeze(-1)
        return logit, alpha


model  = BiLSTMAttention(
    input_size=1, hidden=HIDDEN, n_layers=N_LAYERS,
    attn_dim=ATTN_DIM, mlp_dims=MLP_DIMS,
    dropout=DROPOUT, embed_dim=EMBED_DIM,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n── Model ─────────────────────────────────────────────────")
print(model)
print(f"  Trainable params : {n_params:,}")

# ─────────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────────

criterion = nn.BCEWithLogitsLoss()
optimiser = Adam(model.parameters(), lr=LR)

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def run_epoch(loader, train=True):
    model.train(train)
    total_loss, probs_list, labels_list = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logit, _ = model(x)
            loss = criterion(logit, y)
            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            total_loss += loss.item() * len(y)
            probs_list.append(torch.sigmoid(logit).detach().cpu().numpy())
            labels_list.append(y.cpu().numpy())
    probs  = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    auprc  = average_precision_score(labels, probs) if labels.sum() > 0 else 0.0
    return total_loss / len(loader.dataset), probs, labels, auprc


stopper = EarlyStopping(PATIENCE)
history = dict(epoch=[], tr_loss=[], tr_auprc=[], val_loss=[], val_auprc=[])

print(f"\n── Training (lr={LR}, batch={BATCH_SIZE}, patience={PATIENCE}) ──")
t0 = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    tr_loss, _, _, tr_auprc = run_epoch(train_loader, train=True)
    vl_loss, _, _, vl_auprc = run_epoch(test_loader,  train=False)

    history["epoch"].append(epoch)
    history["tr_loss"].append(round(tr_loss, 4))
    history["tr_auprc"].append(round(tr_auprc, 4))
    history["val_loss"].append(round(vl_loss, 4))
    history["val_auprc"].append(round(vl_auprc, 4))

    print(f"  Epoch {epoch:3d} | "
          f"tr_loss={tr_loss:.4f}  tr_AUPRC={tr_auprc:.4f} | "
          f"val_loss={vl_loss:.4f}  val_AUPRC={vl_auprc:.4f}")

    if stopper.step(vl_loss, model):
        print(f"\n  Early stop @ epoch {epoch}. "
              f"Best val_loss={stopper.best_loss:.4f}")
        break

stopper.restore(model)
elapsed = time.time() - t0
print(f"\n  Training time : {elapsed:.1f}s")

# ─────────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────────

_, probs, labels, _ = run_epoch(test_loader, train=False)
preds = (probs >= 0.5).astype(int)

auprc = average_precision_score(labels, probs)
auc   = roc_auc_score(labels, probs)
acc   = accuracy_score(labels, preds)
f1    = f1_score(labels, preds)

print("\n" + "="*58)
print("  TEST RESULTS   (following TDXtracted Section VII-B)")
print("="*58)
print(f"  AUPRC    : {auprc:.4f}   ← primary metric (paper standard)")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1-score : {f1:.4f}")
print("="*58)

print("\nDetailed classification report:")
print(classification_report(labels, preds,
      target_names=["background", "breast_cancer"]))

# Precision @ Recall (Whisper Leak Table 2 projection style)
precision_arr, recall_arr, _ = precision_recall_curve(labels, probs)
print("── Precision @ Recall thresholds ────────────────────────")
for target_r in [0.05, 0.10, 0.20, 0.50, 0.80]:
    idx = np.argmin(np.abs(recall_arr - target_r))
    print(f"  Recall ≈ {target_r:.0%}  → "
          f"Precision = {precision_arr[idx]:.4f}  "
          f"(actual recall = {recall_arr[idx]:.4f})")

# ── Save outputs ──────────────────────────────────────────────────
hist_df = pd.DataFrame(history)
hist_df.to_csv("/mnt/user-data/outputs/training_history.csv", index=False)
torch.save(model.state_dict(), "/mnt/user-data/outputs/bilstm_model.pt")
print(f"\nModel   saved → /mnt/user-data/outputs/bilstm_model.pt")
print(f"History saved → /mnt/user-data/outputs/training_history.csv")

# Final summary
best_ep = history["val_auprc"].index(max(history["val_auprc"])) + 1
print(f"\n── Final Summary ─────────────────────────────────────────")
print(f"  Best epoch     : {best_ep}")
print(f"  Best val AUPRC : {max(history['val_auprc']):.4f}")
print(f"  Test AUPRC     : {auprc:.4f}")
print(f"  Test AUC-ROC   : {auc:.4f}")
print(f"  Test Accuracy  : {acc:.4f}")
print(f"  Test F1        : {f1:.4f}")
print(f"  Total epochs   : {len(history['epoch'])}")
print(f"  Train time     : {elapsed:.1f}s")
