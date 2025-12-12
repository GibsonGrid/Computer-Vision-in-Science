import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from pyts.approximation import PiecewiseAggregateApproximation
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# Synthetic dataset
# ============================================================
def generate_dataset(n_subjects=5, signals_per_subject=20, length=500):
    X, y, subjects = [], [], []
    for sid in range(n_subjects):
        for _ in range(signals_per_subject):
            t = np.linspace(0, 10 * np.pi, length)
            signal = np.sin(t) + 0.2 * np.random.randn(length) + sid * 0.1
            X.append(signal)
            y.append(sid % 2)      # dummy binary labels
            subjects.append(sid)
    return np.array(X), np.array(y), np.array(subjects)

# ============================================================
# Windowing utilities
# ============================================================
def sliding_windows(signal, N=128, O=64):
    """Windows of length N with overlap O for one 1D signal."""
    return np.array([
        signal[start:start + N]
        for start in range(0, len(signal) - N + 1, N - O)
    ])

def make_windowed_dataset(X, y, N=128, O=64):
    """
    Apply sliding window segmentation to all signals X and replicate labels y.
    Returns windowed signals and corresponding labels.
    """
    X_w, y_w = [], []
    for s, lab in zip(X, y):
        ws = sliding_windows(s, N=N, O=O)
        X_w.extend(ws)
        y_w.extend([lab] * len(ws))
    return np.array(X_w), np.array(y_w)

# ============================================================
def apply_paa(window, window_size=8):
    paa = PiecewiseAggregateApproximation(window_size=window_size)
    return paa.transform(window.reshape(1, -1))[0]

# ============================================================
# Time-series imaging transforms
gasf = GramianAngularField(method="summation")
gadf = GramianAngularField(method="difference")
mtf  = MarkovTransitionField()
rp   = RecurrencePlot(threshold="point", percentage=10)

def window_to_images(win, use_paa=False, paa_ws=8):
    """Convert one window to four 2D images (GASF, GADF, MTF, RP)."""
    w = apply_paa(win, paa_ws) if use_paa else win
    X1d = w.reshape(1, -1)
    img_gasf = gasf.fit_transform(X1d)[0]
    img_gadf = gadf.fit_transform(X1d)[0]
    img_mtf  = mtf.fit_transform(X1d)[0]
    img_rp   = rp.fit_transform(X1d)[0]
    return img_gasf, img_gadf, img_mtf, img_rp

# ============================================================
def loso_split(X, y, subjects, test_subject):
    """Leave-one-subject-out split."""
    train_idx = subjects != test_subject
    test_idx  = subjects == test_subject
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

# ============================================================
class TSIDataset(Dataset):
    def __init__(self, X, y, use_paa=False):
        self.X = X
        self.y = y
        self.use_paa = use_paa

    def __getitem__(self, idx):
        win = self.X[idx]
        imgs = window_to_images(win, use_paa=self.use_paa)
        imgs = np.stack(imgs, axis=0).astype(np.float32)  # (4, H, W)
        return torch.tensor(imgs), torch.tensor(self.y[idx]).long()

    def __len__(self):
        return len(self.X)

# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ============================================================
# LOSO + metrics + saving
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, y, subjects = generate_dataset()
results_loso = {}

# window parameters
WIN_LEN = 128
WIN_OVL = 64

for test_sid in np.unique(subjects):
    # 1) LOSO split
    Xtr, ytr, Xte, yte = loso_split(X, y, subjects, test_sid)

    # 2) Windowing (segmentation) via dedicated function
    Xtr_w, ytr_w = make_windowed_dataset(Xtr, ytr, N=WIN_LEN, O=WIN_OVL)
    Xte_w, yte_w = make_windowed_dataset(Xte, yte, N=WIN_LEN, O=WIN_OVL)

    train_ds = TSIDataset(Xtr_w, ytr_w, use_paa=True)
    test_ds  = TSIDataset(Xte_w, yte_w, use_paa=True)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 3) Model and training
    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):  # short demo training
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 4) Evaluation on test subject
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            y_true.extend(yb.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # 5) Metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    specificity = tn / (tn + fp + 1e-8)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred, zero_division=0)

    f1_micro    = f1_score(y_true, y_pred, average="micro",    zero_division=0)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_binary   = f1_score(y_true, y_pred, average="binary",   zero_division=0)

    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        roc_auc = float("nan")

    sid_key = int(test_sid)
    results_loso[sid_key] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": cm,
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_binary": float(f1_binary),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
    }

    print(f"Finished LOSO fold: subject {test_sid}")

# 6) Save all LOSO results to JSON
with open("loso_results.json", "w") as f:
    json.dump(
        {
            str(k): {
                m: (v.tolist() if isinstance(v, np.ndarray) else v)
                for m, v in res.items()
            }
            for k, res in results_loso.items()
        },
        f,
        indent=2,
    )

print("All LOSO results saved to loso_results.json")
