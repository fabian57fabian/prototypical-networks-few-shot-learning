import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import dataset as torch_dataset

# https://discuss.pytorch.org/t/loss-problem-in-net-finetuning/18311/23?page=2

class DatasetMiniImagenet(torch_dataset):
    def __init__(self, dts_path: str, img_resize=(28,28)):
        super(DatasetMiniImagenet, self).__init__()
        self.dts_path = dts_path
        if not os.path.exists(self.dts_path):
            raise Exception("Dataset path does not exist!")
        # load classes
        self.classes = os.listdir(self.dts_path)
        if len(self.classes) == 0:
            raise Exception("Dataset is empty (no classes)")
        # load dataset
        # TODO: maybe update with a caching system
        self.items = []
        self.items_by_class = {cls: [] for cls in self.classes}
        for cls in self.classes:
            for file in os.listdir(os.path.join(self.dts_path, cls)):
                path = os.path.join(self.dts_path, cls, file)
                # Open img file to grayscale
                img = Image.open(path).convert('L')
                x = img.resize(img_resize)
                shape = 1, x.size[0], x.size[1]
                x = np.array(x, np.float32, copy=False)
                x = 1.0 - torch.from_numpy(x)
                x = x.transpose(0, 1).contiguous().view(shape)
                self.items.append([x, cls, file])

    def __getitem__(self, idx):
        it = self.items[idx]
        return it[0], it[1]

    def __len__(self):
        return len(self.items)