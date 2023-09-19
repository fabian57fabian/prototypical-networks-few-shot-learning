import argparse
from src.core import learn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Encoder trained model.')
    parser.add_argument('--data', type=str, required=True, help='Data from where to extract centroids.')
    parser.add_argument('--image-size', type=int, required=True, help='Image size from pretrained model')
    parser.add_argument('--image-ch', type=int, default=None, help='Image channels from pretrained model')
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    args = parser.parse_args()

    learn(args.model, args.data, args.image_size, args.image_ch, args.gpu)
