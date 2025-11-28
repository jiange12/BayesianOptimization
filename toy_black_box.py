# csv file is param_space.csv

import math

def run_model(*args, lr=0.01, weight_decay=0.001, hidden_size=64, dropout=0.2, optimizer='adam', gamma=0.96, **kwargs):
    """
    Deterministic toy black-box function.
    Loss is minimal at:
        lr = 0.005
        weight_decay = 0.0005
        hidden_size = 128
        dropout = 0.1
        optimizer = 'adamw'
        gamma = 0.95
    """
    
    # Define “optimal” values
    opt_vals = {
        'lr': 0.005,
        'weight_decay': 0.0005,
        'hidden_size': 128,
        'dropout': 0.1,
        'optimizer': 'adamw',
        'gamma': 0.95
    }

    # Continuous parameters: use squared difference
    loss = 0
    loss += (lr - opt_vals['lr'])**2
    loss += (weight_decay - opt_vals['weight_decay'])**2
    loss += ((hidden_size - opt_vals['hidden_size'])/100)**2   # scale hidden_size
    loss += (dropout - opt_vals['dropout'])**2
    loss += (gamma - opt_vals['gamma'])**2

    # Categorical parameter: 0 if match, 1 if mismatch
    loss += 0 if optimizer == opt_vals['optimizer'] else 1

    return loss
