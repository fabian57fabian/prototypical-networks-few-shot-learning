import logging
import os
import shutil
import random
import math

from src.utils import download_file_from_url
from src.data.AbstractClassificationDataset import AbstractDataset


def create_random_splits(all_classes, train_perc=.6, val_perc=.2):
    if not train_perc + val_perc < 1:
        raise Exception("train and validation should be under 1 (100%)")
    train_num = int(len(all_classes) * train_perc)
    train_cls = random.sample(all_classes, train_num)
    remaining_cars = [i for i in all_classes if i not in train_cls]
    val_num = math.ceil(len(all_classes) * val_perc)
    val_cls = random.sample(remaining_cars, val_num)
    test_cls = [i for i in remaining_cars if i not in val_cls]
    return train_cls, val_cls, test_cls

def postprocess_dataset(src_dir, dest_dir):
    files = os.listdir(src_dir)
    train_cls, val_cls, test_cls = create_random_splits(files,.6, .2)

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
                logging.warning(f"Class {cls} not in StanfordCars dataset. Skipping")
            else:
                shutil.move(_from, dest)

    move_this(train_cls, src_dir, path_dataset_train)
    move_this(val_cls, src_dir, path_dataset_val)
    move_this(test_cls, src_dir, path_dataset_test)


def download_dataset_stanford_cars(dest_dir, url):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    tmp_dest_dir = os.path.join(dest_dir, "tmp")
    if not os.path.exists(tmp_dest_dir):
        os.mkdir(tmp_dest_dir)
    download_file_from_url(url, dest_dir)
    tar_file = os.path.join(dest_dir,'stanford_cars.tar.xz')
    os.system(f"tar xf {tar_file} -C {tmp_dest_dir}")
    os.remove(tar_file)
    postprocess_dataset(tmp_dest_dir, dest_dir)
    shutil.rmtree(tmp_dest_dir)


class StanfordCarsDataset(AbstractDataset):
    URL = "https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/download/v0.4-dataset-stanford-cars/stanford_cars.tar.xz"
    def __init__(self, mode='train', load_on_ram=True, download=True,images_size=None, tmp_dir="datasets", seed=3840):
        if seed >= 0: random.seed(seed)
        images_size = 74 if images_size is None else images_size
        super().__init__(mode, (images_size, images_size, 3), load_on_ram, download, tmp_dir,"stanford_cars", download_dataset_stanford_cars, StanfordCarsDataset.URL)