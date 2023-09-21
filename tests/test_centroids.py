import os
import torch
from unittest import TestCase
from src.data.centroids import save_centroids


class TestCore(TestCase):

    def setUp(self) -> None:
        self.path_save = "test_save.npy"

    def tearDown(self) -> None:
        if os.path.exists(self.path_save):
            os.remove(self.path_save)

    def test_save_centroids(self):
        centroids = torch.ones(size=(64,2))
        save_centroids(self.path_save, centroids)
        assert os.path.exists(self.path_save), "centroids not saved!"
