from unittest import TestCase
from datetime import datetime

import torch

from src.prototypical_net import PrototypicalNetwork


class TestPrototypicalNetwork(TestCase):
    def test_forward_torch(self):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        model = PrototypicalNetwork().to(device)
        x = torch.rand(size=(600, 3, 84, 84)).to(device)
        def get_time():
            start_time = datetime.now()
            out = model(x)
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return duration
        urations = [get_time() for _ in range(5)]
        assert True
