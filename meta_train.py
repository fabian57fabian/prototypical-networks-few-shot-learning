import argparse
import os

from src.core import meta_train, get_allowed_base_datasets_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Epochs to train.')
    parser.add_argument('--dataset', type=str, default='mini_imagenet', help='Dataset to train in ["mini_imagenet", "omniglot", "flowers102", "path/to/my_custom_dataset"]')
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--train-num-query', '--train-query', type=int, default=15, help='Number of queries samples per class during training.')
    parser.add_argument('--train-num-class', '--train-way', type=int, default=30, help='Number of classes in batch during training.')
    parser.add_argument('--val-num-class', '--val-way', type=int, default=5, help='Number of classes in batch during validation.')
    parser.add_argument('--episodes-per-epoch', type=int, default=100, help='Number of episodes (iterations) per epoch (training/validation).')
    parser.add_argument('--number-support', '--shot', type=int, default=5, help='Number of support samples per class during training.')
    parser.add_argument('--adam-lr', type=float, default=0.001, help='Base learning rate for Adam.')
    parser.add_argument('--opt-step-size', type=int, default=20, help='Optimization scheduler step size (defualt 20)')
    parser.add_argument('--opt-gamma', type=float, default=0.5, help='Optimization scheduler gamma (defualt 0.5)')
    parser.add_argument('--image-size', type=int, default=None, help='Convert in a different square size the image from dataset.')
    parser.add_argument('--image-ch', type=int, default=None, help='Convert in a different channel size the image from dataset.')
    parser.add_argument('--distance-function', type=str, default="euclidean", choices=["euclidean", "cosine"], help='Distance function.')
    parser.add_argument('--save-each', type=int, default=5, help='Save model each N epochs, default 5')
    parser.add_argument('--eval-each', type=int, default=1, help='Evaluate each X epochs, default 1.')
    parser.add_argument('--early-stop-count', type=int, default=-1, help='Execute early stopping after COUNT epochs. default -1.')
    parser.add_argument('--early-stop-delta', type=float, default=0.0, help='Early stopping delta value for score change, default 0.')
    parser.add_argument('--load-model', type=str, default=None, help='Start training from existing model')
    args = parser.parse_args()

    dataset = args.dataset
    if dataset not in get_allowed_base_datasets_names():
        assert os.path.exists(dataset), "given custom path does not exist"
        if not os.path.isabs(dataset):
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            dataset = os.path.join(cur_dir, dataset)

    meta_train(dataset, args.epochs, args.gpu, args.adam_lr,
          args.train_num_class, args.val_num_class, args.train_num_query, args.number_support,
          args.episodes_per_epoch, args.opt_step_size, args.opt_gamma, args.distance_function, args.image_size, args.image_ch, args.save_each, args.eval_each,
          args.early_stop_count, args.early_stop_delta, args.load_model)
