import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. GENERATE NOISY DATASET (Perfect for Overfitting)
torch.manual_seed(42)
np.random.seed(42)

X = torch.randn(100, 1) * 2  # 100 samples, 1 feature
# True relationship is a simple line, but we add heavy random noise
Y = 3.5 * X + 2.0 + torch.randn(100, 1) * 4.5

# Split into 20 Train samples (tiny training set forces overfitting) and 80 Val samples
X_train, Y_train = X[:20], Y[:20]
X_val, Y_val = X[20:], Y[20:]

# ---------------------------------------------------------------------
# MODEL A: DELIBERATELY OVERFITTED NETWORK
# ---------------------------------------------------------------------
# A massive network with 1000 hidden units processing only 20 samples will memorize the data easily.
class OverfittedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
        )
    def forward(self, x): return self.net(x)

# ---------------------------------------------------------------------
# MODEL B: THE REGULARIZED & NORMALIZED FIXED NETWORK
# ---------------------------------------------------------------------
class FixedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 1000),
            nn.BatchNorm1d(1000),  # Stabilizes the activation distributions
            nn.ReLU(),
            nn.Dropout(p=0.5),     # Deactivates 50% of neurons randomly
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 1)
        )
    def forward(self, x): return self.net(x)

# TRAINING FUNCTION ENGINE
def train_model(model, weight_decay=0.0, epochs=300):
    criterion = nn.MSELoss()
    # weight_decay param handles L2 Regularization automatically inside the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train() # Enable training state (turns Dropout/BatchNorm ON)
        optimizer.zero_grad()
        pred_train = model(X_train)
        loss_train = criterion(pred_train, Y_train)
        loss_train.backward()
        optimizer.step()
        
        model.eval() # Enable evaluation state (turns Dropout/BatchNorm OFF)
        with torch.no_grad():
            pred_val = model(X_val)
            loss_val = criterion(pred_val, Y_val)
            
        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        
    return train_losses, val_losses

# Run Both Simulations
train_loss_overfit, val_loss_overfit = train_model(OverfittedNet(), weight_decay=0.0)
train_loss_fixed, val_loss_fixed = train_model(FixedNet(), weight_decay=0.1) # L2 lambda set to 0.1

# ---------------------------------------------------------------------
# 3. PLOTTING THE DIAGNOSTIC CURVES
# ---------------------------------------------------------------------
plt.figure(figsize=(14, 6))

# Plot 1: The Overfitted Disaster
plt.subplot(1, 2, 1)
plt.plot(train_loss_overfit, label='Train Loss', color='blue', lw=2)
plt.plot(val_loss_overfit, label='Val Loss', color='red', lw=2)
plt.title("The Overfitting Split (Unregularized Model)")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.yscale('log')
plt.legend()
plt.grid(True, linestyle='--')

# Plot 2: The Generalizing Fixed Model
plt.subplot(1, 2, 2)
plt.plot(train_loss_fixed, label='Train Loss', color='blue', lw=2)
plt.plot(val_loss_fixed, label='Val Loss', color='green', lw=2)
plt.title("The Generalized Convergence (Fixed Model)")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.yscale('log')
plt.legend()
plt.grid(True, linestyle='--')

plt.tight_layout()
plt.show()