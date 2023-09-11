import os
import shutil
from datetime import datetime

from unittest import TestCase
from src.OmniglotDataset import OmniglotDataset

class TestMiniImagenetDataset(TestCase):
    def test_constructor_download(self):
        _dts_dir = "datasets_og"
        if os.path.exists(_dts_dir): shutil.rmtree(_dts_dir)
        dl = OmniglotDataset(batch_size=16, load_on_ram=False, download=True, tmp_dir=_dts_dir)
        assert os.path.exists(dl.dts_dir)
        assert len(os.listdir(dl.dts_dir)) == 3
        for d in ["test", "val", "train"]:
            p = os.path.join(dl.dts_dir, d)
            n_classes = len(os.listdir(p))
            assert n_classes > 0
            for ch in os.listdir(p):
                assert len(os.listdir(os.path.join(p, ch))) > 0
