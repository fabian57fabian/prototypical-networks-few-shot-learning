import argparse
from src.core import meta_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Encoder trained model')
    parser.add_argument('--episodes-per-epoch', type=int, default=100, help='Number of episodes (iterations).')
    parser.add_argument('--dataset', type=str, default='mini_imagenet', help='Dataset to test in ["mini_imagenet", "omniglot", "flowers102"]')
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--test-num-query', '--test-query', type=int, default=15, help='Number of queries samples per class during test.')
    parser.add_argument('--test-num-class', '--test-way', type=int, default=5, help='Number of classes in batch during test.')
    parser.add_argument('--number-support', '--shot', type=int, default=5, help='Number of support samples per class during test.')
    parser.add_argument('--distance-function', type=str, default="euclidean", choices=["eucldean", "cosine"], help='Distance function.')
    parser.add_argument('--image-size', type=int, default=None, help='Convert in a different square size the image from dataset.')
    args = parser.parse_args()

    meta_test(args.model, args.episodes_per_epoch, args.dataset, args.gpu,
         args.test_num_query, args.test_num_class, args.number_support,
         args.distance_function, args.image_size)