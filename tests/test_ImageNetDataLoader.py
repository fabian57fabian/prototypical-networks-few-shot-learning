import os
import shutil
from datetime import datetime

from unittest import TestCase
from src.data.MiniImagenetDataset import MiniImagenetDataset, download_dataset


class TestMiniImagenetDataset(TestCase):
    def test_constructor_download(self):
        _dts_dir = "datasets"
        if os.path.exists(_dts_dir): shutil.rmtree(_dts_dir)
        dl = MiniImagenetDataset(load_on_ram=False, download=True)
        assert os.path.exists(dl.dts_dir)
        assert len(os.listdir(dl.dts_dir)) == 3

    def test_constructor_onram(self):
        _dts_dir = "datasets_ram"
        if os.path.exists(_dts_dir): shutil.rmtree(_dts_dir)
        dl = MiniImagenetDataset(load_on_ram=True, download=True, tmp_dir=_dts_dir)
        assert os.path.exists(dl.dts_dir)
        assert len(os.listdir(dl.dts_dir)) == 3

    def test_getsample(self):
        _dts_dir = "datasets_sample"
        if os.path.exists(_dts_dir): shutil.rmtree(_dts_dir)
        dl = MiniImagenetDataset(load_on_ram=True, download=True, tmp_dir=_dts_dir)
        NC, NS, NQ = 20, 5, 6
        start_time = datetime.now()
        sample = dl.GetSample(NC, NS, NQ)
        duration = (datetime.now() - start_time).total_seconds() * 1000
        print(f"GetSample ms: {duration}")
        x, y = sample
        shx = x.shape
        shy = y.shape
        assert shx[0] == NC * (NS + NQ) and shx[1] == 3 and shx[2] == dl.IMAGE_SIZE[0] and shx[3] == dl.IMAGE_SIZE[1]
        assert shy[0] == NC * (NS + NQ)

    def test_download_dataset(self):
        _dir = "tmp_dts"
        if os.path.exists(_dir): shutil.rmtree(_dir)
        download_dataset(_dir)
        assert os.path.exists(_dir)
        train_dir = os.path.join(_dir, "train")
        val_dir = os.path.join(_dir, "val")
        test_dir = os.path.join(_dir, "test")

        assert os.path.exists(train_dir)
        assert len(os.listdir(train_dir)) == 64

        assert os.path.exists(val_dir)
        assert len(os.listdir(val_dir)) == 16

        assert os.path.exists(test_dir)
        assert len(os.listdir(test_dir)) == 20

