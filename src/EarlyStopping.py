import numpy as np
import torch


# Thanks to https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
# Changed to fit our requirements

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=None):
        """
        Args:
            patience (int): How long to wait after last time metric improved. Set negative to deactivate
                            Default: 7
            verbose (bool): If True, prints a message for each metric improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.deactivated = patience < 0
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, loss):
        if self.deactivated:
            return

        score = -loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace_func is not None:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
