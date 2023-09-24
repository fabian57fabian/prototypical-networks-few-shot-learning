import os
import shutil
from datetime import datetime

from unittest import TestCase
from src.data.MiniImagenetDataset import MiniImagenetDataset, download_dataset_miniimagenet


class TestMiniImagenetDataset(TestCase):

    def setUp(self) -> None:
        self.tmp_downlaod_dir = "tmp_download_miniimagenet"
        self.dts_dir = "datasetstest_miniimagenet"
        if os.path.exists(self.dts_dir): shutil.rmtree(self.dts_dir)

        MiniImagenetDataset.URL = "https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/download/v0.0-unit-tests-dataset-mini_imagenet/mini_imagenet.zip"
        self.dl = MiniImagenetDataset(load_on_ram=True, download=True, tmp_dir=self.dts_dir, images_size=10)

    def tearDown(self) -> None:
        if os.path.exists(self.dts_dir): shutil.rmtree(self.dts_dir)
        if os.path.exists(self.tmp_downlaod_dir): shutil.rmtree(self.tmp_downlaod_dir)

    def test_downloaded_files(self):
        assert os.path.exists(self.dl.dts_dir)
        assert len(os.listdir(self.dl.dts_dir)) == 3
        for d in ["test", "val", "train"]:
            p = os.path.join(self.dl.dts_dir, d)
            n_classes = len(os.listdir(p))
            assert n_classes > 0
            for ch in os.listdir(p):
                assert len(os.listdir(os.path.join(p, ch))) > 0

    def test_getsample(self):
        NC, NS, NQ = 2, 2, 1
        start_time = datetime.now()
        sample = self.dl.GetSample(NC, NS, NQ)
        duration = (datetime.now() - start_time).total_seconds() * 1000
        print(f"GetSample ms: {duration}")
        x, y = sample
        shx = x.shape
        shy = y.shape
        assert shx[0] == NC * (NS + NQ) and shx[1] == 3 and shx[2] == self.dl.IMAGE_SIZE[0] and shx[3] == self.dl.IMAGE_SIZE[1]
        assert shy[0] == NC * (NS + NQ)

    def test_download_dataset(self):
        _dir = self.tmp_downlaod_dir
        if os.path.exists(_dir): shutil.rmtree(_dir)
        download_dataset_miniimagenet(_dir, "https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/download/v0.0-unit-tests-dataset-mini_imagenet/mini_imagenet.zip")
        assert os.path.exists(_dir)
        train_dir = os.path.join(_dir, "train")
        val_dir = os.path.join(_dir, "val")
        test_dir = os.path.join(_dir, "test")

        assert os.path.exists(train_dir)
        assert len(os.listdir(train_dir)) > 0

        assert os.path.exists(val_dir)
        assert len(os.listdir(val_dir)) > 0

        assert os.path.exists(test_dir)
        assert len(os.listdir(test_dir)) > 0

