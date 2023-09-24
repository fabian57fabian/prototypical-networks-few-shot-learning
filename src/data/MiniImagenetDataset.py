import os

from src.utils import download_file_from_url
from src.data.AbstractClassificationDataset import AbstractDataset

def download_dataset_miniimagenet(dest_dir, url):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    download_file_from_url(url, dest_dir)
    zip_file = os.path.join(dest_dir,'mini_imagenet.zip')
    os.system(f"unzip -q {zip_file} -d {dest_dir}")
    os.remove(zip_file)
    path_dataset_train = os.path.join(dest_dir, 'train')
    path_dataset_val = os.path.join(dest_dir, 'val')
    path_dataset_test = os.path.join(dest_dir, 'test')
    num_cl_train = len(os.listdir(path_dataset_train))
    num_cl_val = len(os.listdir(path_dataset_val))
    num_cl_test = len(os.listdir(path_dataset_test))
    print(f"Dataset MiniImageNet: {num_cl_train} train, {num_cl_val} val, {num_cl_test} test")


class MiniImagenetDataset(AbstractDataset):
    URL = "https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/download/v0.1-dataset-mini_imagenet/mini_imagenet.zip"
    def __init__(self, mode='train', load_on_ram=True, download=True, images_size=None, tmp_dir="datasets"):
        images_size = 84 if images_size is None else images_size
        super().__init__(mode, (images_size, images_size, 3), load_on_ram, download, tmp_dir, "mini_imagenet", download_dataset_miniimagenet, MiniImagenetDataset.URL)