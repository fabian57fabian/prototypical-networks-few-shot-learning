from unittest import TestCase
import torch

from src.prototypical_loss import euclidean_dist, cosine_dist


class Test(TestCase):
    def test_euclidean_dist(self):
        t1 = torch.rand(size=(45, 64)) # query_samples
        t2 = torch.rand(size=(1, 64)) # prototype
        res = euclidean_dist(t1, t2).squeeze()
        assert res.shape[0] == t1.shape[0]

    def test_cosine_dist(self):
        t1 = torch.rand(size=(45, 64)) # query_samples
        t2 = torch.rand(size=(1, 64)) # prototype
        res = torch.nn.functional.cosine_similarity(t1, t2, dim=1).squeeze()
        assert res.shape[0] == t1.shape[0]
