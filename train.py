import argparse
from src.core import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Epochs to train.')
    parser.add_argument('--dataset', type=str, default='mini_imagenet', help='Dataset to test in ["mini_imagenet", "omniglot", "flowers102"]')
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--train-num-query', type=int, default=15, help='')
    parser.add_argument('--train-num-class', type=int, default=30, help='')
    parser.add_argument('--test-num-class', type=int, default=5, help='')
    parser.add_argument('--episodes-per-epoch', type=int, default=50, help='')
    parser.add_argument('--number-support', type=int, default=5, help='Number of samples/class to use as support')
    parser.add_argument('--adam-lr', type=float, default=0.01, help='Base learning rate for Adam.')
    parser.add_argument('--opt-step-size', type=int, default=20, help='Optimization scheduler step size (defualt 20)')
    parser.add_argument('--opt-gamma', type=float, default=0.55, help='Optimization scheduler gamma (defualt 0.5)')
    parser.add_argument('--save-each', type=int, default=5, help='Save model each N epochs, default 5')
    args = parser.parse_args()

    train(args.dataset, args.epochs, args.gpu, args.adam_lr,
          args.train_num_class, args.test_num_class, args.train_num_query, args.number_support,
          args.episodes_per_epoch, args.opt_step_size, args.opt_gamma, args.save_each)