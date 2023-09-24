import logging
import os
import shutil
import random

from src.utils import download_file_from_url
from src.data.AbstractClassificationDataset import AbstractDataset


def create_random_splits(train_num=64, val_num=16):
    dataset_num = 102
    assert train_num + val_num <= dataset_num
    all_classes = [i+1 for i in range(dataset_num)]
    train_cls = random.sample(all_classes, train_num)
    remaining_fl102 = [i for i in all_classes if i not in train_cls]
    val_cls = random.sample(remaining_fl102, val_num)
    test_cls = [i for i in remaining_fl102 if i not in val_cls]
    return train_cls, val_cls, test_cls

def postprocess_dataset(src_dir, dest_dir):
    train_cls, val_cls, test_cls = create_random_splits(64, 16)

    # Create destination dirs
    path_dataset_train = os.path.join(dest_dir, 'train')
    path_dataset_val = os.path.join(dest_dir, 'val')
    path_dataset_test = os.path.join(dest_dir, 'test')
    if not os.path.exists(path_dataset_train): os.mkdir(path_dataset_train)
    if not os.path.exists(path_dataset_val): os.mkdir(path_dataset_val)
    if not os.path.exists(path_dataset_test): os.mkdir(path_dataset_test)

    def move_this(move_classes, src,  dest):
        for cls_int in move_classes:
            cls = str(cls_int)
            _from = os.path.join(src, cls)
            _to = os.path.join(dest, cls)
            if not os.path.exists(_from):
                logging.warning(f"Class {cls} not in flowers102 dataset. Skipping")
            else:
                shutil.move(_from, dest)

    move_this(train_cls, src_dir, path_dataset_train)
    move_this(val_cls, src_dir, path_dataset_val)
    move_this(test_cls, src_dir, path_dataset_test)


def download_dataset_flowers102(dest_dir, url):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    tmp_dest_dir = os.path.join(dest_dir, "tmp")
    if not os.path.exists(tmp_dest_dir):
        os.mkdir(tmp_dest_dir)
    download_file_from_url(url, dest_dir)
    zip_file = os.path.join(dest_dir,'flowers102.zip')
    os.system(f"unzip -q {zip_file} -d {tmp_dest_dir}")
    os.remove(zip_file)
    postprocess_dataset(tmp_dest_dir, dest_dir)
    shutil.rmtree(tmp_dest_dir)


class Flowers102Dataset(AbstractDataset):
    URL = "https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/download/v0.3-dataset-flowers102/flowers102.zip"
    def __init__(self, mode='train', load_on_ram=True, download=True,images_size=None, tmp_dir="datasets", seed=3840):
        if seed >= 0: random.seed(seed)
        images_size = 74 if images_size is None else images_size
        super().__init__(mode, (images_size, images_size, 3), load_on_ram, download, tmp_dir,"flowers102", download_dataset_flowers102, Flowers102Dataset.URL)