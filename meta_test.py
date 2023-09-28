import argparse
import os
from pathlib import Path

from src.data import ALLOWED_BASE_DATASETS
from src import entrypoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Dataset to train in ["mini_imagenet", "omniglot", "flowers102", "path/to/my_custom_dataset"]')
    parser.add_argument('--model', type=str, default=None, help='Encoder trained model')
    parser.add_argument('--iterations', type=int, default=None, help='Number of episodes (iterations) per epoch (training/validation).')
    parser.add_argument('--device', type=str, default=None,  help='Device to use')
    parser.add_argument('--query', type=int, default=None, help='Number of queries samples per class during training.')
    parser.add_argument('--val-num-way', type=int, default=None, help='Number of classes in batch during validation.')
    parser.add_argument('--shot', type=int, default=None, help='Number of support samples per class during training.')
    parser.add_argument('--metric', type=str, default=None, choices=["euclidean", "cosine"], help='Distance function.')
    parser.add_argument('--imgsz', type=int, default=None, help='Convert in a different square size the image from dataset.')
    parser.add_argument('--channels', type=int, default=None, help='Convert in a different channel size the image from dataset.')
    args = parser.parse_args()

    cfg = vars(args)
    cfg["mode"] = "eval"
    if cfg['data'] not in ALLOWED_BASE_DATASETS:
        # Custom dir, build full path
        if not os.path.isabs(cfg['data']):
            cfg['data'] = os.path.join(os.path.dirname(__file__), cfg['data'])

    entrypoint(cfg)