# ai4ct/advanced/model_inception.py
"""InceptionTime + Tversky Loss — raw sequence, multi-scale 1D convolutions."""
import time, os, math, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, roc_curve, auc)
from .config import SEED, OUTPUT_DIR

# --- Tversky Loss ---
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__(); self.alpha=alpha; self.beta=beta; self.smooth=smooth
    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)[:,1]
        tp = (probs * targets.float()).sum()
        fp = (probs * (1-targets.float())).sum()
        fn = ((1-probs) * targets.float()).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha*fn + self.beta*fp + self.smooth)
        return 1.0 - tversky

# --- Inception Block ---
class InceptionBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        bottleneck = max(1, out_c // 4)
        self.bottleneck = nn.Sequential(nn.Conv1d(in_c, bottleneck, 1), nn.BatchNorm1d(bottleneck), nn.ReLU())
        self.b1 = nn.Conv1d(bottleneck, bottleneck, 1)
        self.b2 = nn.Conv1d(bottleneck, bottleneck, 3, padding=1)
        self.b3 = nn.Conv1d(bottleneck, bottleneck, 5, padding=2)
        self.b4 = nn.Conv1d(bottleneck, bottleneck, 7, padding=3)
        self.fuse = nn.Sequential(nn.Conv1d(bottleneck*4, out_c, 1), nn.BatchNorm1d(out_c), nn.ReLU())
        self.res = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x):
        y = self.bottleneck(x)
        y = torch.cat([self.b1(y), self.b2(y), self.b3(y), self.b4(y)], dim=1)
        return self.bn(F.relu(self.fuse(y) + self.res(x)))

# --- Full Model ---
class InceptionTimeModel(nn.Module):
    def __init__(self, in_channels=122, hidden=64):
        super().__init__()
        self.block1 = InceptionBlock(in_channels, hidden)
        self.block2 = InceptionBlock(hidden, hidden)
        self.block3 = InceptionBlock(hidden, hidden)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2))

    def forward(self, x):
        x = x.permute(0,2,1)          # (B, input channels, sequence length)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)    # (B, hidden)
        return self.classifier(x)

    def cam_heatmap(self, x):
        """Score-CAM: return (B, T) activation map from last block."""
        x = x.permute(0,2,1)
        x = self.block1(x); x = self.block2(x)
        x3 = self.block3(x)            # (B, hidden, T)
        w = self.classifier[0].weight[1]  # positive class weights
        cam = torch.einsum('bct,c->bt', x3, w)
        return cam.cpu().detach()

# --- Trainer ---
def _pad_batch(seqs, max_len=362):
    B = len(seqs); D = seqs[0].shape[1]
    out = torch.zeros(B, max_len, D)
    for i, s in enumerate(seqs):
        L = min(s.shape[0], max_len); out[i,:L] = torch.from_numpy(s[:L])
    return out

def train_and_eval(X_raw_train, y_train, X_raw_val, y_val, X_raw_test, y_test,
                   epochs=200, lr=1e-3, batch_size=16, device="cpu"):
    model_dir = os.path.join(OUTPUT_DIR, "inception"); os.makedirs(model_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # raw sequence has 124 channels: 120 MLC + 2 JawY + Gantry + CumWeight
    model = InceptionTimeModel(in_channels=124, hidden=64).to(device)
    criterion = TverskyLoss(alpha=0.7, beta=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    best_val_auc = 0; patience = 40; wait = 0
    hist = {'train_loss':[], 'val_auc':[]}

    t0 = time.time()
    for ep in range(epochs):
        model.train(); idx = torch.randperm(len(X_raw_train)); total_loss = 0.0
        for i in range(0, len(idx), batch_size):
            bi = idx[i:i+batch_size]
            xb = _pad_batch([X_raw_train[j] for j in bi]).to(device)
            yb = torch.tensor(y_train[bi], dtype=torch.long).to(device)
            optimizer.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step(); total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            xv = _pad_batch(X_raw_val).to(device)
            prob = torch.softmax(model(xv),1)[:,1].cpu().numpy()
            val_auc = roc_auc_score(y_val, prob)
        hist['train_loss'].append(total_loss); hist['val_auc'].append(val_auc)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc; wait = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
        else:
            wait += 1
            if wait >= patience: break
        if (ep+1) % 40 == 0:
            print(f"[INC] E{ep+1:>3d} loss={total_loss:.2f} val_auc={val_auc:.4f}")
    train_time = time.time() - t0

    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt")))
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        xt = _pad_batch(X_raw_test).to(device)
        prob = torch.softmax(model(xt),1)[:,1].cpu().numpy()
        pred = (prob >= 0.5).astype(int)
        cam = model.cam_heatmap(xt[0:1]).numpy()[0]  # first test sample
    infer_time = time.time() - t1

    acc  = accuracy_score(y_test, pred)
    sens = recall_score(y_test, pred)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    prec = precision_score(y_test, pred, zero_division=0)
    bal  = (sens+spec)/2.0
    auc_v = roc_auc_score(y_test, prob)
    fpr, tpr, _ = roc_curve(y_test, prob)

    print(f"[INC] Acc={acc:.4f} Sens={sens:.4f} Spec={spec:.4f} "
          f"Prec={prec:.4f} Bal={bal:.4f} AUC={auc_v:.4f} Time={train_time:.1f}s")

    # Score-CAM heatmap
    fig, ax = plt.subplots(figsize=(12,3), dpi=150)
    ax.plot(cam,'g-',lw=.5); ax.fill_between(range(len(cam)),0,cam,alpha=.3,color='green')
    ax.set_xlabel("Control Point"); ax.set_ylabel("Score-CAM"); ax.set_title("InceptionTime Score-CAM")
    plt.tight_layout(); plt.savefig(os.path.join(model_dir,"scorecam_heatmap.png"),dpi=150); plt.close()

    # ROC
    fig, ax = plt.subplots(figsize=(6,6), dpi=150)
    ax.plot(fpr,tpr,lw=2,label=f'IncTime (AUC={auc_v:.4f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=.3); ax.legend(); ax.grid(alpha=.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("InceptionTime ROC")
    plt.tight_layout(); plt.savefig(os.path.join(model_dir,"roc.png"),dpi=150); plt.close()

    return {'model':'InceptionTime','accuracy':acc,'sensitivity':sens,'specificity':spec,
            'precision':prec,'balanced_accuracy':bal,'auc':auc_v,
            'train_time_s':train_time,'inference_time_s':infer_time,
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'prob':prob,'pred':pred}
