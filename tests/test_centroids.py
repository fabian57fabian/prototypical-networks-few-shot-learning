import os
import shutil
import uuid

import numpy as np
import torch
from PIL import Image
from unittest import TestCase
from src.data.centroids import save_centroids, load_centroids


class TestCentroids(TestCase):

    def setUp(self) -> None:
        self.path_save = "test_save.npy"
        self.path_load = "test_dataset_centroids"
        if os.path.exists(self.path_load): shutil.rmtree(self.path_load)
        os.mkdir(self.path_load)
        self.classes_files = 3
        self.centroids_size = 64
        for i in range(self.classes_files):
            cl_path = os.path.join(self.path_load, str(i) + ".npy")
            centroids = np.random.randint(0, 10, self.centroids_size) / 10
            np.save(cl_path, centroids)

    def tearDown(self) -> None:
        if os.path.exists(self.path_save):
            os.remove(self.path_save)
        if os.path.exists(self.path_load):
            shutil.rmtree(self.path_load)

    def test_save_centroids(self):
        centroids = torch.ones(size=(64, 2))
        save_centroids(self.path_save, centroids)
        assert os.path.exists(self.path_save), "centroids not saved!"

    def test_load_centroids(self):
        c = load_centroids(self.path_load)
        assert type(c) is tuple and len(c) == 2
        centroids, names = c
        assert centroids.shape == (self.classes_files, self.centroids_size)
        assert len(names) == self.classes_files
