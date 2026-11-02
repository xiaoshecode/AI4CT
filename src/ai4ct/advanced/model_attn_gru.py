# ai4ct/advanced/model_attn_gru.py
"""Attention-GRU + Focal Loss — trainer uses 362 CP × 124 input values."""
import time, os, math, random, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, roc_curve, auc)
from .config import SEED, OUTPUT_DIR

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.5):
        super().__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        w = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (w * (1 - pt) ** self.gamma * ce).mean()

# --- Model ---
class AttnGRU(nn.Module):
    def __init__(self, input_dim=122, hidden=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv1d(input_dim, 64, 1), nn.BatchNorm1d(64), nn.GELU())
        self.gru = nn.GRU(64, hidden, num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers>1 else 0)
        self.attn_q = nn.Linear(hidden*2, 64)
        self.attn_k = nn.Linear(hidden*2, 64)
        self.attn_v = nn.Linear(hidden*2, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2))
        self._attn_weights = None

    def forward(self, x):
        B, T, D = x.shape
        x = self.proj(x.permute(0,2,1)).permute(0,2,1)  # Conv1d along feature axis
        h, _ = self.gru(x)                               # (B, T, 128)
        q = self.attn_q(h); k = self.attn_k(h); v = self.attn_v(h)
        scores = torch.softmax((q @ k.transpose(-2,-1)) / math.sqrt(64), dim=-1)
        self._attn_weights = scores.mean(dim=1)           # (B, T) ← interpretability
        ctx = torch.sum(scores.unsqueeze(-1) * v.unsqueeze(1), dim=2)  # (B, T, 64)
        pooled = (self._attn_weights.unsqueeze(-1) * ctx).sum(dim=1)    # (B, 64)
        return self.classifier(pooled)

    @property
    def attention_weights(self): return self._attn_weights

# --- Trainer ---
def _pad_batch(seqs, max_len=362):
    B = len(seqs); D = seqs[0].shape[1]
    out = torch.zeros(B, max_len, D)
    for i, s in enumerate(seqs):
        L = min(s.shape[0], max_len); out[i,:L] = torch.from_numpy(s[:L])
    return out

def train_and_eval(X_raw_train, y_train, X_raw_val, y_val, X_raw_test, y_test,
                   epochs=150, lr=1e-3, batch_size=32, device="cpu"):
    model_dir = os.path.join(OUTPUT_DIR, "attn_gru"); os.makedirs(model_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model = AttnGRU(input_dim=124).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0; patience = 30; wait = 0
    hist = {'train_loss':[], 'val_auc':[]}

    t0 = time.time()
    for ep in range(epochs):
        model.train(); idx = torch.randperm(len(X_raw_train))
        total_loss = 0.0
        for i in range(0, len(idx), batch_size):
            bi = idx[i:i+batch_size]
            xb = _pad_batch([X_raw_train[j] for j in bi]).to(device)
            yb = torch.tensor(y_train[bi], dtype=torch.long).to(device)
            optimizer.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step(); total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            xv = _pad_batch(X_raw_val).to(device)
            prob = torch.softmax(model(xv),1)[:,1].cpu().numpy()
            val_auc = roc_auc_score(y_val, prob)
        hist['train_loss'].append(total_loss); hist['val_auc'].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc; wait = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
        else:
            wait += 1
            if wait >= patience: break

        if (ep+1) % 30 == 0:
            print(f"[GRU] E{ep+1:>3d} loss={total_loss:.2f} val_auc={val_auc:.4f}")
    train_time = time.time() - t0

    # Load best & evaluate
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt")))
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        xt = _pad_batch(X_raw_test).to(device)
        prob = torch.softmax(model(xt),1)[:,1].cpu().numpy()
        pred = (prob >= 0.5).astype(int)
        attn = model.attention_weights.cpu().numpy()[0]  # first test sample
    infer_time = time.time() - t1

    acc  = accuracy_score(y_test, pred)
    sens = recall_score(y_test, pred)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    prec = precision_score(y_test, pred, zero_division=0)
    bal  = (sens+spec)/2.0
    auc_v = roc_auc_score(y_test, prob)
    fpr, tpr, _ = roc_curve(y_test, prob)

    print(f"[GRU] Acc={acc:.4f} Sens={sens:.4f} Spec={spec:.4f} "
          f"Prec={prec:.4f} Bal={bal:.4f} AUC={auc_v:.4f} Time={train_time:.1f}s")

    # Attention heatmap
    fig, ax = plt.subplots(figsize=(12,3), dpi=150)
    ax.plot(attn, 'b-', lw=0.5); ax.fill_between(range(len(attn)), 0, attn, alpha=0.3)
    ax.set_xlabel("Control Point"); ax.set_ylabel("Attention"); ax.set_title("Attention-GRU CP-level Focus")
    plt.tight_layout(); plt.savefig(os.path.join(model_dir,"attention_heatmap.png"),dpi=150); plt.close()

    # ROC
    fig, ax = plt.subplots(figsize=(6,6), dpi=150)
    ax.plot(fpr,tpr,lw=2,label=f'GRU (AUC={auc_v:.4f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=.3); ax.legend(); ax.grid(alpha=.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Attention-GRU ROC")
    plt.tight_layout(); plt.savefig(os.path.join(model_dir,"roc.png"),dpi=150); plt.close()

    return {'model':'Attention-GRU','accuracy':acc,'sensitivity':sens,'specificity':spec,
            'precision':prec,'balanced_accuracy':bal,'auc':auc_v,
            'train_time_s':train_time,'inference_time_s':infer_time,
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'prob':prob,'pred':pred}
