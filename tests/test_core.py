import os
import uuid
import shutil
import numpy as np
from PIL import Image
from unittest import TestCase
from src.core import cosine_dist, euclidean_dist, init_savemodel, build_distance_function, build_dataloaders, build_dataloaders_test
from src import entrypoint
from src.data import ALLOWED_BASE_DATASETS
from src.data.Flowers102Dataset import Flowers102Dataset
from src.data.MiniImagenetDataset import MiniImagenetDataset
from src.data.OmniglotDataset import OmniglotDataset
from src.data.StanfordCarsDataset import StanfordCarsDataset
from src.data.CustomDataset import CustomDataset


class TestCore(TestCase):

    def setUp(self) -> None:
        self.model_to_use = "omniglot_28_5shot.pt"
        link_download = "https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/download/v0.0-unit-tests-dataset-mini_imagenet/omniglot_28_5shot.pt"
        os.system(f"wget {link_download} -P ./")
        self.model_outputs = 64

        # Create centroids for learn()
        self.path_learn = "test_dataset_test_to_load"
        os.mkdir(self.path_learn)
        self.centroids_images = 5
        self.classes_num = 3
        self.classes_of_learn = []
        for i in range(self.classes_num):
            class_name = str(i)
            cl_path = os.path.join(self.path_learn, class_name)
            os.mkdir(cl_path)
            self.classes_of_learn.append(class_name)
            for i_image in range(self.centroids_images):
                im_path = os.path.join(cl_path, str(uuid.uuid4()) + ".jpg")
                Image.new('RGB', (28, 28)).save(im_path)

        # Create data for meta_train()
        self.base_datasets_meta_train = "datasets"
        os.mkdir(self.base_datasets_meta_train)
        custom_dts_name = "custom_1"
        self.datasets_to_meta_learn = [custom_dts_name, "mini_imagenet", "omniglot", "flowers102", "stanford_cars"]
        self.channels_to_meta_learn = [3, 3, 1, 3, 3]
        self.meta_train_images = 10
        self.meta_train_classes_num = 5
        for dataset, channels in zip(self.datasets_to_meta_learn, self.channels_to_meta_learn):
            path_meta_train = os.path.join(self.base_datasets_meta_train, dataset)
            os.mkdir(path_meta_train)
            for phase in ["train", "test", "val"]:
                path_phase = os.path.join(path_meta_train, phase)
                os.mkdir(path_phase)
                for i in range(self.meta_train_classes_num):
                    class_name = str(i)
                    cl_path = os.path.join(path_phase, class_name)
                    os.mkdir(cl_path)
                    for i_image in range(self.meta_train_images):
                        im_path = os.path.join(cl_path, str(uuid.uuid4()) + ".jpg")
                        img = Image.new('RGB', (28, 28))
                        if channels == 1:
                            img = img.convert('L')
                        img.save(im_path)
        # custom dataset should be path to a dataset to train
        self.datasets_to_meta_learn[0] = os.path.join(self.base_datasets_meta_train, custom_dts_name)

        # Create centroids data for predict()
        self.path_load_centroids = "test_dataset_centroids"
        if os.path.exists(self.path_load_centroids): shutil.rmtree(self.path_load_centroids)
        os.mkdir(self.path_load_centroids)
        self.classes_files = 3
        self.classes_of_predict = []
        for i in range(self.classes_files):
            class_name = str(i)
            cl_path = os.path.join(self.path_load_centroids, class_name + ".npy")
            centroids = np.random.randint(0, 10, self.model_outputs) / 10
            np.save(cl_path, centroids)
            self.classes_of_predict.append(class_name)
        # Create images for predict()
        self.path_images_predict = "images_predict"
        os.mkdir(self.path_images_predict)
        self.num_images_predict = 2
        for i in range(self.num_images_predict):
            im_path = os.path.join(self.path_images_predict, str(uuid.uuid4()) + ".jpg")
            Image.new('RGB', (28, 28)).save(im_path)

    def tearDown(self) -> None:
        if os.path.exists(self.model_to_use):
            os.remove(self.model_to_use)

        if os.path.exists(self.base_datasets_meta_train):
            shutil.rmtree(self.base_datasets_meta_train)

        if os.path.exists(self.path_load_centroids):
            shutil.rmtree(self.path_load_centroids)

        if os.path.exists(self.path_images_predict):
            shutil.rmtree(self.path_images_predict)

        if os.path.exists(self.path_learn):
            shutil.rmtree(self.path_learn)

        if os.path.exists("runs"):
            shutil.rmtree("runs")

    def test_ALLOWED_BASE_DATASETS(self):
        ad = ALLOWED_BASE_DATASETS
        assert type(ad) is list
        assert "mini_imagenet" in ad
        assert "omniglot" in ad
        assert "flowers102" in ad
        assert "stanford_cars" in ad


    def test_meta_train(self):
        for distance_fn in ["cosine", "euclidean"]:
            for dts, ch in zip(self.datasets_to_meta_learn, self.channels_to_meta_learn):
                usable_supp_query =  self.meta_train_images - 2
                cfg = {"data": dts, "episodes": 2, "device": "cpu", "adam_lr": 0.1,
                       "num_way": self.meta_train_classes_num - 1, "val_num_way": int(self.meta_train_classes_num / 2),
                       "query": int(usable_supp_query * .4), "shot": int(usable_supp_query * .6), "iterations": 10,
                       "adam_step": 20, "adam_gamma": .5, "metric": distance_fn, "imgsz": 16, "channels": ch,
                       "save_period": 1, "eval_each": 2, "patience": 50, "patience_delta": 0, "mode": "train"}
                entrypoint(cfg)
                assert os.path.exists("runs")
                path_run = f"runs/train_0"
                assert os.path.exists(path_run), f"Run not found in {path_run}"
                files = list(os.listdir(path_run))
                assert "model_best.pt" in files
                for i in range(cfg["episodes"] + 1):
                    assert f"model_{i}.pt" in files
                assert "config.yaml" in files
                shutil.rmtree(path_run)

    def test_loading_model(self):
        cfg = {"data": "mini_imagenet", "model": self.model_to_use, "episodes": 2, "device": "cpu", "adam_lr": 0.1,
               "num_way": 1, "val_num_way": 1, "query": 1, "shot": 1, "iterations": 10, "imgsz": 28, "channels": 1,
               "save_period": 1, "eval_each": 1, "patience": 50, "patience_delta": 0, "mode": "train"}
        train_dir = entrypoint(cfg)
        assert os.path.exists(train_dir)
        assert os.path.exists(os.path.join(train_dir, "model_best.pt"))

    def test_meta_test(self):
        dataset = self.datasets_to_meta_learn[1]
        image_ch = self.channels_to_meta_learn[1]
        usable_supp_query = self.meta_train_images - 2
        cfg = {"model": self.model_to_use, "data": dataset, "episodes": 2, "device": "cpu", "adam_lr": 0.1,
               "val_num_way": int(self.meta_train_classes_num / 2),
               "query": int(usable_supp_query * .4), "shot": int(usable_supp_query * .6), "iterations": 10,
               "adam_step": 20, "adam_gamma": .5, "metric": "euclidean", "imgsz": 16, "channels": image_ch,
               "save_period": 1, "eval_each": 2, "patience": 50, "patience_delta": 0, "mode": "eval"}
        acc = entrypoint(cfg)
        assert acc is not None

    def test_learn(self):
        cfg = {"data": self.path_learn, "model": self.model_to_use, "device": "cpu",
               "imgsz": 28, "channels": 3, "mode": "learn"}
        entrypoint(cfg)
        assert os.path.exists("runs")
        path_centroids = "runs/centroids_0"
        assert os.path.exists(path_centroids)
        for file_npy in os.listdir(path_centroids):
            assert file_npy.endswith("npy")
            class_name = file_npy[:-4]
            assert class_name in self.classes_of_learn
            path = os.path.join(path_centroids, file_npy)
            np_file = np.load(path)
            assert np_file.shape[0] == self.model_outputs

    def test_predict(self):
        cfg = {"data": self.path_images_predict, "model": self.model_to_use, "device": "cpu",
               "centroids": self.path_load_centroids,
               "imgsz": 28, "channels": 3, "mode": "predict"}
        res = entrypoint(cfg)
        assert len(res) == self.num_images_predict
        for img_path, classification in res:
            assert classification in self.classes_of_predict

    def test_build_dataloaders_unknown(self):
        with self.assertRaises(Exception) as context:
            build_dataloaders(str(uuid.uuid4()), 28, 3)
        assert "dataset unknown" in context.exception.args

    def test_build_dataloaders_test_unknown(self):
        with self.assertRaises(Exception) as context:
            build_dataloaders_test(str(uuid.uuid4()), 28, 3)
        assert "dataset unknown" in context.exception.args

    def test_build_dataloaders_test_omniglot(self):
        OmniglotDataset.URL = "https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/download/v0.0-unit-tests-dataset-mini_imagenet/omniglot.zip"
        dl = build_dataloaders_test("omniglot", 28, 1)
        assert dl is not None
        assert type(dl) == OmniglotDataset

    def test_build_dataloaders_test_mini_imagenet(self):
        MiniImagenetDataset.URL = "https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/download/v0.0-unit-tests-dataset-mini_imagenet/mini_imagenet.zip"
        dl = build_dataloaders_test("mini_imagenet", 28, 3)
        assert dl is not None
        assert type(dl) == MiniImagenetDataset

    def test_build_dataloaders_test_flowers102(self):
        Flowers102Dataset.URL = "https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/download/v0.0-unit-tests-dataset-mini_imagenet/flowers102.zip"
        dl = build_dataloaders_test("flowers102", 10, 3)
        assert dl is not None
        assert type(dl) == Flowers102Dataset

    def test_build_dataloaders_test_stanford_cars(self):
        StanfordCarsDataset.URL = "https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/download/v0.0-unit-tests-dataset-mini_imagenet/stanford_cars.tar.xz"
        dl = build_dataloaders_test("stanford_cars", 10, 3)
        assert dl is not None
        assert type(dl) == StanfordCarsDataset

    def test_init_savemodel(self):
        prefix = "wow"
        init_savemodel(prefix)
        assert os.path.exists(os.path.join("runs", f"{prefix}_0"))
        init_savemodel(prefix)
        assert os.path.exists(os.path.join("runs", f"{prefix}_1"))

    def test_build_distance_function(self):
        func1 = build_distance_function("cosine")
        assert func1 == cosine_dist
        func2 = build_distance_function("euclidean")
        assert func2 == euclidean_dist
        with self.assertRaises(Exception) as context:
            _ = build_distance_function("aaaaaaaaaaaaaaaaaaaaaaa")
        assert "Wrong distance function supplied" in context.exception.args
