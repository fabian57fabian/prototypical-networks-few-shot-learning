import os
import shutil

from tqdm import tqdm

from PIL import Image
from src.utils import download_file_from_url
from src.data.AbstractClassificationDataset import AbstractDataset

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

def download_dataset_omniglot(dest_dir, url):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    tmp_dest_dir = os.path.join(dest_dir, "tmp")
    if not os.path.exists(tmp_dest_dir):
        os.mkdir(tmp_dest_dir)
    download_file_from_url(url, dest_dir)
    zip_file = os.path.join(dest_dir,'omniglot.zip')
    os.system(f"unzip -q {zip_file} -d {tmp_dest_dir}")
    os.remove(zip_file)
    postprocess_dataset(tmp_dest_dir, dest_dir)
    shutil.rmtree(tmp_dest_dir)


class OmniglotDataset(AbstractDataset):
    URL = "https://github.com/fabian57fabian/prototypical-networks-few-shot-learning/releases/download/v0.2-dataset-omniglot/omniglot.zip"
    def __init__(self, mode='train', load_on_ram=True, download=True, images_size=None, tmp_dir="datasets"):
        images_size = 28 if images_size is None else images_size
        super().__init__(mode, (images_size, images_size, 1), load_on_ram, download, tmp_dir,"omniglot", download_dataset_omniglot, OmniglotDataset.URL)
