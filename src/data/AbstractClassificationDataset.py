import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image


class AbstractDataset:
    def __init__(self, mode='train', data_shape=(84, 84, 3), load_on_ram=True, download=True, tmp_dir="datasets", dataset_name="dataset_YYY", download_function = None, download_url = ""):
        assert mode in ['train', 'val', 'test'], "given mode should be train, val or test."
        self.IMAGE_SIZE = (data_shape[0], data_shape[1])
        self.IMAGE_CHANNELS = data_shape[2]
        self.load_on_ram = load_on_ram
        assert self.load_on_ram, "Currently accepting only RAM loading"
        base_dts = tmp_dir
        dataset_exists = False
        if not os.path.exists(base_dts) and base_dts != '':
            os.mkdir(base_dts)
        self.dts_dir = os.path.join(base_dts, dataset_name)
        if not os.path.exists(self.dts_dir):
            os.mkdir(self.dts_dir)

        self.curr_dataset_folder = os.path.join(self.dts_dir, mode)
        if os.path.exists(self.curr_dataset_folder):
            self.classes = os.listdir(self.curr_dataset_folder)
            if len(self.classes) > 0:
                dataset_exists = True
        if download and not dataset_exists:
            print("Downloading dataset")
            download_function(self.dts_dir, download_url)
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
        assert NC <= len(self.classes), "There are less classes than required for this batch. Try lowering number of classes."

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
            cache_class = load_class_images(path, self.IMAGE_SIZE, self.IMAGE_CHANNELS)
            self.cache[cache_index:cache_index+cache_class.shape[0], ...] = cache_class
            cache_index += cache_class.shape[0]

def load_class_images(path, IMAGE_SIZE, CHANNELS) -> torch.tensor:
    """
    Loads all images in a folder as a tensor with size.
    :param path: Images dir
    :param IMAGE_SIZE: Requested image size e.g. (64, 64)
    :param CHANNELS: Requested image channnels e.g. 1 or 3
    :return: tensor IAMGES_COUNT X C X H X W
    """
    if CHANNELS is None: CHANNELS = 3
    images = os.listdir(path)
    cache = torch.rand(size=(len(images), CHANNELS, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    cache_index = 0
    for i, img_file in enumerate(images):
        img_path = os.path.join(path, img_file)
        t_img = load_image(img_path, IMAGE_SIZE)
        cache[cache_index, ...] = t_img
        cache_index += 1
    return cache

def load_image(img_path: str, requested_size: tuple = None) -> torch.tensor:
    """
    Loads an image into a tensor
    :param img_path: Image path
    :param requested_size: Requested image size e.g. (64, 64)
    :return: tensor C X H X W
    """
    pil_img = Image.open(img_path)
    if requested_size is not None and pil_img.size != requested_size:
        pil_img = pil_img.resize(requested_size)
    t_img = transforms.PILToTensor()(pil_img)
    return t_img