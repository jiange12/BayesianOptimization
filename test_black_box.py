import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any

# This is the "black box" that bayesian_optimize will call
def run_model(*args, opt=None, scheduler=None, loss_fn=None, **params) -> float:
    """
    A general black-box model that:
      - defines a PyTorch model internally
      - constructs optimizer, scheduler, and loss function based on given params
      - trains briefly on a synthetic dataset (for demo)
      - returns a scalar loss (to minimize)
    
    Arguments:
      *args / **params: arbitrary hyperparameters (lr, hidden_size, dropout, etc.)
      opt, scheduler, loss_fn: optional fixed components or factories passed externally
    """

    # ======== Step 1: Prepare synthetic data (example dataset) ========
    torch.manual_seed(0)
    N = 256
    X = torch.linspace(-2 * np.pi, 2 * np.pi, N).reshape(-1, 1)
    y = torch.sin(X) + 0.1 * torch.randn_like(X)

    # ======== Step 2: Build model ========
    hidden_size = int(params.get("hidden_size", 64))
    dropout = float(params.get("dropout", 0.2))

    class SimpleRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleRegressor()

    # ======== Step 3: Define optimizer, scheduler, and loss ========
    lr = float(params.get("lr", 1e-3))
    weight_decay = float(params.get("weight_decay", 0.0))
    gamma = float(params.get("gamma", 0.96))
    optimizer_choice = params.get("optimizer", "adamw").lower()

    # choose optimizer dynamically
    if optimizer_choice == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_choice == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    # loss functions
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    # combine them for demo (weighted sum)
    def combined_loss(pred, target):
        return mse_loss(pred, target) + 0.1 * mae_loss(pred, target)

    # ======== Step 4: Training loop (short, just for demonstration) ========
    n_epochs = 30
    model.train()
    for epoch in range(n_epochs):
        opt.zero_grad()
        y_pred = model(X)
        loss = combined_loss(y_pred, y)
        loss.backward()
        opt.step()
        scheduler.step()

    # ======== Step 5: Evaluation loss ========
    model.eval()
    with torch.no_grad():
        y_hat = model(X)
        final_loss = float(combined_loss(y_hat, y).item())

    # This final loss is what Bayesian optimization will minimize
    return final_loss
