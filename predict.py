import argparse
import os

from src.core import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Encoder trained model.')
    parser.add_argument('--centroids', type=str, required=True, help='Centroids directory path.')
    parser.add_argument('--images-path', type=str, required=True, help='Image to test.')
    parser.add_argument('--image-size', type=int, required=True, help='Image size from pretrained model.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for faster results.')
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    args = parser.parse_args()

    path = args.images_path
    images = [path] if os.path.isfile(path) else [os.path.join(path, f) for f in os.listdir(path)]

    predict(args.model, args.centroids, images, args.image_size, args.batch_size, args.gpu)
