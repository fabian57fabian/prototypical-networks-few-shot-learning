from unittest import TestCase
import torch

from src.prototypical_loss import euclidean_dist, cosine_dist


class Test(TestCase):
    def test_euclidean_dist(self):
        t1 = torch.rand(size=(45, 64)) # query_samples
        t2 = torch.rand(size=(3, 64)) # prototype
        res = euclidean_dist(t1, t2)
        assert res.shape[0] == t1.shape[0] and res.shape[1] == t2.shape[0]

    def test_euclidean_dist_different_shape(self):
        t1 = torch.rand(size=(45, 61)) # query_samples
        t2 = torch.rand(size=(3, 64)) # prototype
        with self.assertRaises(Exception) as context:
            euclidean_dist(t1, t2)

    def test_cosine_dist(self):
        t1 = torch.rand(size=(45, 64)) # query_samples
        t2 = torch.rand(size=(3, 64)) # prototype
        res = cosine_dist(t1, t2)
        assert res.shape[0] == t1.shape[0] and res.shape[1] == t2.shape[0]

    def test_cosine_dist_different_shape(self):
        t1 = torch.rand(size=(45, 61)) # query_samples
        t2 = torch.rand(size=(3, 64)) # prototype
        with self.assertRaises(Exception) as context:
            cosine_dist(t1, t2)