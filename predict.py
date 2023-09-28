import argparse
import os

from src import entrypoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Encoder trained model')
    parser.add_argument('--centroids', type=str, required=True, help='Centroids directory path.')
    parser.add_argument('--data', type=str, required=True, help='Image/s to test.')
    parser.add_argument('--imgsz', type=int, default=None, help='Convert in a different square size the image from dataset.')
    parser.add_argument('--device', type=str, default=None,  help='Device to use')
    args = parser.parse_args()

    cfg = vars(args)
    cfg["mode"] = "predict"

    entrypoint(cfg)
