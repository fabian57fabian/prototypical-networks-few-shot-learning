import os
import random
import shutil

from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def read_alphabets_in_splits(filename):
    with open(filename, "r") as file:
        alphabets = set([line.rstrip().split('/')[0] for line in file.readlines()])
    if "" in alphabets:
        alphabets.remove("")
    return alphabets


def postprocess_dataset(src_dir, dest_dir):
    """
    Omniglot has lots of languages, characters per language and finally images.
    Those have to be rotated 0, 90, 180, 270 degrees before being used.
    We'll do it to simplify all
    :param src_dir: downloaded dir
    :param dest_dir: destination dir
    :return:
    """
    path_dataset_train = os.path.join(dest_dir, 'train')
    path_dataset_val = os.path.join(dest_dir, 'val')
    path_dataset_test = os.path.join(dest_dir, 'test')
    if not os.path.exists(path_dataset_train): os.mkdir(path_dataset_train)
    if not os.path.exists(path_dataset_val): os.mkdir(path_dataset_val)
    if not os.path.exists(path_dataset_test): os.mkdir(path_dataset_test)
    train_alp = read_alphabets_in_splits(os.path.join(src_dir, "splits", "train.txt"))
    val_alp = read_alphabets_in_splits(os.path.join(src_dir, "splits", "val.txt"))
    test_alp = read_alphabets_in_splits(os.path.join(src_dir, "splits", "test.txt"))
    def postprocess_this_dir(alphabet, path_dts):
        for lang in tqdm(alphabet, total=len(alphabet)):
            path_src = os.path.join(os.path.join(src_dir, "data", lang))
            for character in os.listdir(path_src):
                path_chr = os.path.join(path_src, character)
                dest_dir_ch = os.path.join(path_dts, f"{lang}___{character}")
                if not os.path.exists(dest_dir_ch): os.mkdir(dest_dir_ch)
                for file in os.listdir(path_chr):
                    img_path = os.path.join(path_chr, file)
                    img = Image.open(img_path)
                    for rot in [0, 90, 180, 270]:
                        x = img.rotate(float(rot))
                        x = x.resize((28, 28))
                        x.save(os.path.join(dest_dir_ch, f"{file[:-4]}_{rot}.png"))

    postprocess_this_dir(train_alp, path_dataset_train)
    postprocess_this_dir(val_alp, path_dataset_val)
    postprocess_this_dir(test_alp, path_dataset_test)

def download_dataset(dest_dir):
    print("Downloading omniglot dataset")
    url = "https://github.com/fabian57fabian/fewshot-learning-prototypical-networks/releases/download/v0.2/omniglot.zip"
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    tmp_dest_dir = os.path.join(dest_dir, "tmp")
    if not os.path.exists(tmp_dest_dir):
        os.mkdir(tmp_dest_dir)
    os.system(f'wget -q "{url}" -P {dest_dir}')
    zip_file = os.path.join(dest_dir,'omniglot.zip')
    os.system(f"unzip -q {zip_file} -d {tmp_dest_dir}")
    os.remove(zip_file)
    postprocess_dataset(tmp_dest_dir, dest_dir)
    shutil.rmtree(tmp_dest_dir)



class OmniglotDataset:
    def __init__(self, mode='train', data_shape=(84, 84, 3), load_on_ram=True, download=True, tmp_dir="datasets"):
        assert mode in ['train', 'val', 'test'], "given mode should be train, val or test."
        self.IMAGE_SIZE = (data_shape[0], data_shape[1])
        self.IMAGE_CHANNELS = data_shape[2]
        self.load_on_ram = load_on_ram
        base_dts = tmp_dir
        dataset_exists = False
        if not os.path.exists(base_dts):
            os.mkdir(base_dts)
        self.dts_dir = os.path.join(base_dts, "omniglot")
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
        self.cache = None
        if self.load_on_ram:
            self._load_on_memory()

    def GetSample(self, NC: int, NS: int, NQ: int):
        """
        Gets ransom samples from dataset
        :param NC: Number of classes to sample
        :param NS: Number of support samples
        :param NQ: Number of query samples
        :return: ndarray with NC X (NS+NQ) X C X H X W, classes encoded
        """
        assert NC <= len(self.classes)

        indexes = []
        y = []

        # choose NC classes
        classes_choosed = random.sample(self.classes, NC)

        # For each class, randomly choose (NS + NQ) examples
        for cl in classes_choosed:
            num_examples_of_dts, start, stop = self.classes_to_indexes[cl]
            indexes_samples = random.sample(range(start, stop, 1), (NS + NQ))
            indexes += indexes_samples
            cls_y = self.classes.index(cl)
            y += [cls_y for _ in range(NS + NQ)]

        # select samples
        samples = self.cache[indexes, ...]# torch.index_select(self.cache, 0, torch.LongTensor(indexes))
        return samples, torch.tensor(y)

    def _load_on_memory(self):
        self.classes = os.listdir(self.curr_dataset_folder)
        # count all images
        # number, start, stop
        self.classes_to_indexes = {c: [-1, 0, 0] for c in self.classes}
        self.index_to_class = []
        images_count = 0
        for i, cl in enumerate(self.classes):
            path = os.path.join(self.curr_dataset_folder, cl)
            tot_imgs = len(os.listdir(path))
            self.classes_to_indexes[cl] = [tot_imgs, images_count, images_count + tot_imgs]
            self.index_to_class += [i for _ in range(tot_imgs)]
            images_count += tot_imgs

        # Create architecture and load
        self.cache = torch.rand(size=(images_count, 3, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1]))
        cache_index = 0
        for cl in self.classes:
            path = os.path.join(self.curr_dataset_folder, cl)
            images = os.listdir(path)
            images_num = len(images)
            for i, img_file in enumerate(images):
                img_path = os.path.join(path, img_file)
                pil_img = Image.open(img_path)
                if pil_img.size != self.IMAGE_SIZE:
                    pil_img = pil_img.resize(self.IMAGE_SIZE)
                t_img = transforms.PILToTensor()(pil_img)
                self.cache[cache_index, ...] = t_img
                cache_index += 1
