import argparse
from src.core import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Base learning rate for Adam.')
    parser.add_argument('--dataset', type=str, default='mini_imagenet', help='Dataset to test in ["mini_imagenet"]')
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--train-num-query', type=int, default=15, help='')
    parser.add_argument('--train-num-class', type=int, default=30, help='')
    parser.add_argument('--number-support', type=int, default=5, help='Number of samples/class to use as support')
    args = parser.parse_args()

    train(args.dataset, args.epochs, args.gpu, args.learning_rate,
          args.train_num_classes, args.train_num_query, args.number_support)