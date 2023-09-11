import os
import shutil

from unittest import TestCase
from src.data.Flowers102Dataset import Flowers102Dataset

class TestFlowers102Dataset(TestCase):

    def test_constructor_download(self):
        _dts_dir = "datasets_fl"
        if os.path.exists(_dts_dir): shutil.rmtree(_dts_dir)
        dl = Flowers102Dataset(load_on_ram=False, download=True, tmp_dir=_dts_dir, seed=647473)
        assert os.path.exists(dl.dts_dir)
        assert len(os.listdir(dl.dts_dir)) == 3
        for d in ["test", "val", "train"]:
            p = os.path.join(dl.dts_dir, d)
            n_classes = len(os.listdir(p))
            assert n_classes > 0
            for ch in os.listdir(p):
                assert len(os.listdir(os.path.join(p, ch))) > 0
