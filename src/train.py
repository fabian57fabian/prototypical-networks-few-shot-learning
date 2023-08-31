import os
from src.prototypical_net import PrototypicalNetwork
from src.DatasetMiniImagenet import DatasetMiniImagenet

import numpy as np
import torch
print(torch.__version__)

# example trian from https://github.com/pytorch/examples/blob/main/mnist/main.py


def train(epochs):
    data_loader = DatasetMiniImagenet("../datasets/mini_imagenet", (28, 28))
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    model = PrototypicalNetwork(1, 32, 32).to(device)

    for epoch in range(epochs):
        print(f"Ep {epoch}")
        model.train()
        # Train
        for batch in next(data_loader):
            pass
