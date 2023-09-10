from unittest import TestCase
from datetime import datetime

import torch

from src.prototypical_net import PrototypicalNetwork


class TestPrototypicalNetwork(TestCase):
    def test_forward_cpu(self):
        device = torch.device("cpu")
        model = PrototypicalNetwork().to(device)
        x = torch.rand(size=(600, 3, 84, 84)).to(device)
        def get_time():
            start_time = datetime.now()
            out = model(x)
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return duration
        durations = [get_time() for _ in range(5)]
        mean = sum(durations) / len(durations)
        print(f"GetSample avg CPU ms: {mean}, [{durations}]")

    def test_forward_gpu(self):
        device = torch.device("cuda:0")
        model = PrototypicalNetwork().to(device)
        x = torch.rand(size=(600, 3, 84, 84)).to(device)
        def get_time():
            start_time = datetime.now()
            out = model(x)
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return duration
        durations = [get_time() for _ in range(5)]
        mean = sum(durations) / len(durations)
        print(f"GetSample avg GPU ms: {mean}, [{durations}]")

