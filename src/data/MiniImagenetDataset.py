import os

from src.data.AbstractClassificationDataset import AbstractDataset

def download_dataset_miniimagenet(dest_dir):
    url = "https://github.com/fabian57fabian/fewshot-learning-prototypical-networks/releases/download/v0.1/mini_imagenet.zip"
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    os.system(f'wget -q "{url}" -P {dest_dir}')
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
    def __init__(self, mode='train', load_on_ram=True, download=True, tmp_dir="datasets"):
        super().__init__(mode, (84, 84, 3), load_on_ram, download, tmp_dir, download_dataset_miniimagenet)