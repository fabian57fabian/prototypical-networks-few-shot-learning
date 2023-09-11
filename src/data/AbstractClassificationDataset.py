import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image


class AbstractDataset:
    def __init__(self, mode='train', data_shape=(84, 84, 3), load_on_ram=True, download=True, tmp_dir="datasets", download_function = None):
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
            download_function(self.dts_dir)
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
