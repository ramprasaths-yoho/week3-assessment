"""Task 2 - PyTorch training pipeline on the breast-cancer dataset."""
import copy
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
MODEL_PATH = os.path.join(ROOT, "models", "task2_mlp.pt")

torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------------ data
data = load_breast_cancer()
X_train, X_val, y_train, y_val = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
scaler = StandardScaler().fit(X_train)             # fit on train only -> no leakage
X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)


class CsvDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(CsvDataset(X_train_s, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(CsvDataset(X_val_s, y_val), batch_size=32, shuffle=False)


# ------------------------------------------------------------------ model
class MLP(nn.Module):
    def __init__(self, in_features=30, hidden=(64, 32), dropout=0.3):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))            # raw logit; sigmoid is inside the loss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        pred = (torch.sigmoid(model(xb)) >= 0.5).float()
        correct += (pred == yb).sum().item()
        total += len(yb)
    return correct / total


# ------------------------------------------------------------------ training with early stopping
model = MLP()
print(f"Params: {count_params(model):,}")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS, PATIENCE = 40, 8
best_acc, best_epoch, best_state, wait = 0.0, 0, None, 0
history = {"train_loss": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()            # 1. clear old gradients
        loss = criterion(model(xb), yb)  # 2. forward + loss
        loss.backward()                  # 3. backward
        optimizer.step()                 # 4. update weights
        running += loss.item() * len(yb)
    train_loss = running / len(train_loader.dataset)
    val_acc = evaluate(model, val_loader)
    history["train_loss"].append(train_loss); history["val_acc"].append(val_acc)
    print(f"epoch {epoch:02d} | train loss {train_loss:.4f} | val acc {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc, best_epoch, best_state, wait = val_acc, epoch, copy.deepcopy(model.state_dict()), 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

model.load_state_dict(best_state)
print(f"\nParams: {count_params(model):,} | Best val accuracy {best_acc:.2f} at epoch {best_epoch}")

# ------------------------------------------------------------------ save / reload / prove identical
torch.save(model.state_dict(), MODEL_PATH)
new_model = MLP()
new_model.load_state_dict(torch.load(MODEL_PATH))
new_model.eval()
reloaded_acc = evaluate(new_model, val_loader)
print(f"Original model accuracy: {evaluate(model, val_loader):.4f}")
print(f"Reloaded model accuracy: {reloaded_acc:.4f} ({'identical' if abs(reloaded_acc - best_acc) < 1e-9 else 'DIFFERENT!'})")

# ------------------------------------------------------------------ scikit-learn comparison
lr = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
sk_acc = accuracy_score(y_val, lr.predict(X_val_s))
print(f"\nComparison -> PyTorch MLP: {reloaded_acc:.4f} | sklearn LogisticRegression: {sk_acc:.4f}")
