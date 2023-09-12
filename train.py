import argparse
from src.core import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Epochs to train.')
    parser.add_argument('--dataset', type=str, default='mini_imagenet', help='Dataset to test in ["mini_imagenet", "omniglot", "flowers102"]')
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--train-num-query', '--train-query', type=int, default=15, help='Number of queries samples per class during training.')
    parser.add_argument('--train-num-class', '--train-way', type=int, default=30, help='Number of classes in batch during training.')
    parser.add_argument('--val-num-class', '--test-way', type=int, default=5, help='Number of classes in batch during validation.')
    parser.add_argument('--episodes-per-epoch', type=int, default=100, help='Number of episodes (iterations) per epoch (training/validation).')
    parser.add_argument('--number-support', '--shot', type=int, default=5, help='Number of support samples per class during training.')
    parser.add_argument('--adam-lr', type=float, default=0.001, help='Base learning rate for Adam.')
    parser.add_argument('--opt-step-size', type=int, default=20, help='Optimization scheduler step size (defualt 20)')
    parser.add_argument('--opt-gamma', type=float, default=0.5, help='Optimization scheduler gamma (defualt 0.5)')
    parser.add_argument('--image-size', type=int, default=None, help='Convert in a different square size the image from dataset.')
    parser.add_argument('--distance-function', type=str, default="euclidean", choices=["euclidean", "cosine"], help='Distance function.')
    parser.add_argument('--save-each', type=int, default=5, help='Save model each N epochs, default 5')
    parser.add_argument('--eval-each', type=int, default=1, help='Evaluate each X epochs, default 1.')
    args = parser.parse_args()

    train(args.dataset, args.epochs, args.gpu, args.adam_lr,
          args.train_num_class, args.val_num_class, args.train_num_query, args.number_support,
          args.episodes_per_epoch, args.opt_step_size, args.opt_gamma, args.distance_function, args.image_size, args.save_each, args.eval_each)
