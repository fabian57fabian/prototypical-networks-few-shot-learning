import os

from prototypical_net import PrototypicalNetwork

import numpy as np
import torch
print(torch.__version__)


def train():

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    model = PrototypicalNetwork(1, 32, 32).to(device)
