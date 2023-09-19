import os

from src.data.AbstractClassificationDataset import AbstractDataset


class CustomDataset(AbstractDataset):
    def __init__(self, mode='train', load_on_ram=True, images_size=None, image_ch=None, dataset_path="custom_dataset_path"):
        images_size = 64 if images_size is None else images_size
        image_ch = 3 if image_ch is None else image_ch
        assert image_ch in [1, 2, 3, 4], "Image channel size not accepted"
        assert os.path.exists(dataset_path), f"Dataset path does not exist: {dataset_path}"
        assert os.path.isdir(dataset_path), f"Dataset path is not a dir: {dataset_path}"
        tmp_dir = os.path.dirname(dataset_path)
        dataset_name = os.path.basename(dataset_path)
        super().__init__(mode, (images_size, images_size, image_ch),
                         load_on_ram, download=False, tmp_dir=tmp_dir, dataset_name=dataset_name)