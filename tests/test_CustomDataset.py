import os
import uuid
import shutil
from PIL import Image
from unittest import TestCase
from src.data.CustomDataset import CustomDataset

class TestFlowCustomDataset(TestCase):

    def setUp(self) -> None:
        # Create dataset with random images
        _dts_dir = "./dataset_custom"
        if os.path.exists(_dts_dir): shutil.rmtree(_dts_dir)
        os.mkdir(_dts_dir)
        train_dir = os.path.join(_dts_dir, "train")
        val_dir = os.path.join(_dts_dir, "val")
        test_dir = os.path.join(_dts_dir, "test")
        os.mkdir(train_dir)
        os.mkdir(val_dir)
        os.mkdir(test_dir)
        self.create_random_images(train_dir, 6, 32, size=64)
        self.create_random_images(val_dir, 6, 20, size=72)
        self.create_random_images(test_dir, 6, 26, size=84)

        self.dataset_dir = _dts_dir

    def tearDown(self) -> None:
        if os.path.exists(self.dataset_dir): shutil.rmtree(self.dataset_dir)

    def create_random_images(self, path, num_classes, num_images, size=64):
        for _ in range(num_classes):
            classname = str(uuid.uuid4())
            path_cls = os.path.join(path, classname)
            os.mkdir(path_cls)
            for _ in range(num_images):
                im = Image.new(mode="RGB", size=(size, size))
                filename = str(uuid.uuid4()) + ".jpg"
                im.save(os.path.join(path_cls, filename))

    def test_constructor_sample(self):
        size = 64
        ch = 3
        dl = CustomDataset(mode='test', load_on_ram=True, images_size=size, image_ch=ch, dataset_path=self.dataset_dir)
        NC, NS, NQ = 2,3,4
        s, y = dl.GetSample(NC, NS, NQ)
        assert s.shape[0] == NC * (NS + NQ), "Wrong sample size"
        assert s.shape[1] == ch and s.shape[2] == size and s.shape[3] == size
        assert y.shape[0] == NC * (NS + NQ), "Wrong target sample size"
