import os

import numpy as np
from PIL import Image

def download_dataset(dest_dir):
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



class MiniImagenetDataset:
    IMAGE_SIZE = (84, 84)
    def __init__(self, mode='train', batch_size=16, load_on_ram=True, download=True, tmp_dir="datasets"):
        assert mode in ['train', 'val', 'test'], "given mode should be train, val or test."
        self.batch_size = batch_size
        self.load_on_ram = load_on_ram
        base_dts = tmp_dir
        dataset_exists = False
        if not os.path.exists(base_dts):
            os.mkdir(base_dts)
        self.dts_dir = os.path.join(base_dts, "mini_imagenet")
        if not os.path.exists(self.dts_dir):
            os.mkdir(self.dts_dir)

        self.curr_dataset_folder = os.path.join(self.dts_dir, mode)
        if os.path.exists(self.curr_dataset_folder):
            self.classes = os.listdir(self.curr_dataset_folder)
            if len(self.classes) > 0:
                dataset_exists = True
        if download and not dataset_exists:
            print("Downloading dataset")
            download_dataset(self.dts_dir)
        self.classes = os.listdir(self.curr_dataset_folder)
        self.cache = {}
        if self.load_on_ram:
            self._load_on_memory()

    def GetSample(self, NC: int, NS: int, NQ: int):
        """
        Gets ransom samples from dataset
        :param NC: Number of classes to sample
        :param NS: Number of support samples
        :param NQ: Number of query samples
        :return: ndarray with NC X NS+NQ, classes encoded
        """
        return np.zeros((NC, NS + NQ)), np.array([i for i in range(NC)])

    def _load_on_memory(self):
        print("Loading dataset on cache")
        self.classes = os.listdir(self.curr_dataset_folder)
        for cl in self.classes:
            path = os.path.join(self.curr_dataset_folder, cl)
            images = os.listdir(path)
            images_num = len(images)
            tens = np.ndarray(shape=(images_num, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3))
            for i, img_file in enumerate(images):
                img_path = os.path.join(path, img_file)
                pil_img = Image.open(img_path)
                if pil_img.size != self.IMAGE_SIZE:
                    pil_img = pil_img.resize(self.IMAGE_SIZE)
                np_img = np.asarray(pil_img)
                tens[i, :, :, :] = np_img
            self.cache[cl] = tens
        print("cache loaded")



    def get_train_dataloader(self):
        return None
