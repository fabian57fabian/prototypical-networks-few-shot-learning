import argparse
import os
from pathlib import Path

from src.data import ALLOWED_BASE_DATASETS
from src import entrypoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Dataset to train in ["mini_imagenet", "omniglot", "flowers102", "path/to/my_custom_dataset"]')
    parser.add_argument('--episodes', type=int, default=None, help='Epochs to train.')
    parser.add_argument('--device', type=str, default=None,  help='Device to use')
    parser.add_argument('--num-way', type=int, default=None, help='Number of classes in batch during training.')
    parser.add_argument('--val-num-way', type=int, default=None, help='Number of classes in batch during validation.')
    parser.add_argument('--shot', type=int, default=None, help='Number of support samples per class during training.')
    parser.add_argument('--query', type=int, default=None, help='Number of queries samples per class during training.')
    parser.add_argument('--iterations', type=int, default=None, help='Number of episodes (iterations) per epoch (training/validation).')
    parser.add_argument('--adam-lr', type=float, default=None, help='Base learning rate for Adam.')
    parser.add_argument('--adam-step', type=int, default=None, help='Optimization scheduler step size (defualt 20)')
    parser.add_argument('--adam-gamma', type=float, default=None, help='Optimization scheduler gamma (defualt 0.5)')
    parser.add_argument('--imgsz', type=int, default=None, help='Convert in a different square size the image from dataset.')
    parser.add_argument('--channels', type=int, default=None, help='Convert in a different channel size the image from dataset.')
    parser.add_argument('--metric', type=str, default=None, choices=["euclidean", "cosine"], help='Distance function.')
    parser.add_argument('--save-period', type=int, default=None, help='Save model each N epochs, default 5')
    parser.add_argument('--eval-each', type=int, default=None, help='Evaluate each X epochs, default 1.')
    parser.add_argument('--patience', type=int, default=None, help='Execute early stopping after COUNT epochs. default -1.')
    parser.add_argument('--patience-delta', type=float, default=None, help='Early stopping delta value for score change, default 0.')
    parser.add_argument('--model', type=str, default=None, help='Start training from existing model')
    args = parser.parse_args()

    cfg = vars(args)
    cfg["mode"] = "train"
    if cfg['data'] not in ALLOWED_BASE_DATASETS:
        # Custom dir, build full path
        if not os.path.isabs(cfg['data']):
            cfg['data'] = os.path.join(os.path.dirname(__file__), cfg['data'])

    entrypoint(cfg)
