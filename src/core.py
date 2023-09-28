import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import yaml
from tqdm import tqdm

from src.EarlyStopping import EarlyStopping
from src.prototypical_net import PrototypicalNetwork
from src.prototypical_loss import prototypical_loss, euclidean_dist, cosine_dist
from src.data.MiniImagenetDataset import MiniImagenetDataset
from src.data.OmniglotDataset import OmniglotDataset
from src.data.Flowers102Dataset import Flowers102Dataset
from src.data.StanfordCarsDataset import StanfordCarsDataset
from src.data.CustomDataset import CustomDataset
from src.data.AbstractClassificationDataset import load_class_images, load_image

from src.data.centroids import load_centroids, save_centroids
from src import __version__

print(f"***** Few-shot Learning with proto nets. v{__version__} *****")

# example train algo from https://github.com/pytorch/examples/blob/main/mnist/main.py
# Loading datasets from https://github.com/learnables/learn2learn/tree/master#learning-domains


def get_allowed_base_datasets_names() -> list:
    return ["mini_imagenet", "omniglot", "flowers102", "stanford_cars"]


def build_dataloaders_test(dataset='mini_imagenet', size=None, channels=None):
    if dataset == 'mini_imagenet':
        test_loader = MiniImagenetDataset(mode='test', load_on_ram=True, download=True, images_size=size, tmp_dir="datasets")
        return test_loader
    elif dataset == 'omniglot':
        test_loader = OmniglotDataset(mode='test', load_on_ram=True, download=True, images_size=size, tmp_dir="datasets")
        return test_loader
    elif dataset == 'flowers102':
        test_loader = Flowers102Dataset(mode='test', load_on_ram=True, download=True, images_size=size, tmp_dir="datasets")
        return test_loader
    elif os.path.exists(dataset):
        test_loader = CustomDataset(mode='test', load_on_ram=True, images_size=size, image_ch=channels, dataset_path=dataset)
        return test_loader
    elif dataset == 'stanford_cars':
        test_loader = StanfordCarsDataset(mode='test', load_on_ram=True, download=True, images_size=size, tmp_dir="datasets")
        return test_loader
    raise Exception("dataset unknown")


def build_dataloaders(dataset='mini_imagenet', size=None, channels=None):
    if dataset == 'mini_imagenet':
        train_loader = MiniImagenetDataset(mode='train', load_on_ram=True, download=True, images_size=size, tmp_dir="datasets")
        valid_loader = MiniImagenetDataset(mode='val', load_on_ram=True, download=False, images_size=size, tmp_dir="datasets")
        return train_loader, valid_loader
    elif dataset == 'omniglot':
        train_loader = OmniglotDataset(mode='train', load_on_ram=True, download=True, images_size=size, tmp_dir="datasets")
        valid_loader = OmniglotDataset(mode='val', load_on_ram=True, download=False, images_size=size, tmp_dir="datasets")
        return train_loader, valid_loader
    elif dataset == 'flowers102':
        train_loader = Flowers102Dataset(mode='train', load_on_ram=True, download=True, images_size=size, tmp_dir="datasets")
        valid_loader = Flowers102Dataset(mode='val', load_on_ram=True, download=False, images_size=size, tmp_dir="datasets")
        return train_loader, valid_loader
    elif os.path.exists(dataset):
        train_loader = CustomDataset(mode='train', load_on_ram=True, images_size=size, image_ch=channels, dataset_path=dataset)
        valid_loader = CustomDataset(mode='val', load_on_ram=True, images_size=size, image_ch=channels, dataset_path=dataset)
        return train_loader, valid_loader
    elif dataset == 'stanford_cars':
        train_loader = StanfordCarsDataset(mode='train', load_on_ram=True, download=True, images_size=size, tmp_dir="datasets")
        valid_loader = StanfordCarsDataset(mode='val', load_on_ram=True, download=False, images_size=size, tmp_dir="datasets")
        return train_loader, valid_loader
    raise Exception("dataset unknown")


def build_device(use_gpu=False):
    device = torch.device("cpu")
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("WWARN: Unable to set device to GPU because not available. Fallback to 'cpu'")
    return device


def build_distance_function(distance_function: str):
    if distance_function not in ["euclidean", "cosine"]:
        raise Exception("Wrong distance function supplied")

    if distance_function == "euclidean":
        return euclidean_dist
    elif distance_function == "cosine":
        return cosine_dist

def save_yaml_config(training_dir, config):
    with open(os.path.join(training_dir, "config.yaml"), 'w') as file:
        file.write(yaml.dump(config))


def init_savemodel(prefix="train") -> str:
    main_dir = "runs"
    if not os.path.exists(main_dir): os.mkdir(main_dir)
    i = 0
    build_dir = lambda idx: f"{main_dir}/{prefix}_{idx}"
    out_dir = build_dir(i)
    while os.path.exists(out_dir):
        out_dir = build_dir(i)
        i += 1
    os.mkdir(out_dir)
    return out_dir


def save_model(model, training_dir, name):
    torch.save(model.state_dict(), os.path.join(training_dir, name))

def load_model(model, path:str):
    model.load_state_dict(torch.load(path))
    model.eval()

def meta_train(dataset='mini_imagenet', epochs=300, use_gpu=False, lr=0.001,
          train_num_classes=30,
          test_num_class=5,
          train_num_query=15,
          number_support=5,
          episodes_per_epoch=50,
          optim_step_size=20,
          optim_gamma = 0.5,
          distance_function="euclidean",
          images_size=None,
          images_ch=None,
          save_each=5,
          eval_each=1,
          es_count=100,
          es_delta=.001,
          model_to_load=None) -> str:
    training_dir = init_savemodel()
    print(f"Writing to {training_dir}")
    writer = SummaryWriter(log_dir=training_dir)

    print("Building DataLoaders")
    train_loader, valid_loader = build_dataloaders(dataset, images_size, images_ch)
    device = build_device(use_gpu)

    print(f"Creating Prototype model on {device}")
    model = PrototypicalNetwork()
    if model_to_load is not None:
        if not os.path.exists(model_to_load):
            raise Exception(f"Model path to load does not exist: {model_to_load}")
        print(f"Loading model {model_to_load}")
        load_model(model, model_to_load)
    model = model.to(device)

    # Optimizer, lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step_size, gamma=optim_gamma)
    early_stopping = EarlyStopping(patience=es_count, delta=es_delta)

    distance_fn = build_distance_function(distance_function)

    # Save config
    config = {
        "dataset": dataset,
        "epochs": epochs,
        "gpu": use_gpu,
        "adam_lr": lr,
        "NC_train": train_num_classes,
        "NQ_train": train_num_query,
        "NC_valid": test_num_class,
        "NS": number_support,
        "ep_per_epoch": episodes_per_epoch,
        "opt_step_size": optim_step_size,
        "opt_gamma": optim_gamma,
        "distance_function": distance_function,
        "images_size": images_size,
        "images_channels": images_ch,
        "save_each": save_each,
        "eval_each": eval_each,
        "early_stopping_count": es_count,
        "early_stopping_delta": es_delta,
        "train_from": model_to_load if model_to_load is not None else ""
    }
    save_yaml_config(training_dir, config)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = -1
    best_acc_ep = -1
    start_time = datetime.datetime.now()

    print(f"Startring training at {str(start_time)}")
    for epoch in range(epochs + 1):
        model.train()
        # Train
        for i in tqdm(range(episodes_per_epoch), total=episodes_per_epoch): # should be enough to cover batch*100 >= dataset_size
            batch = train_loader.GetSample(train_num_classes, number_support, train_num_query)
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x = model(x)
            loss, acc = prototypical_loss(x, y, number_support, train_num_classes, distance_fn)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        if epoch % save_each == 0:
            save_model(model, training_dir, f"model_{epoch}.pt")
        loss_mean, acc_mean = np.mean(train_loss[-episodes_per_epoch:]),np.mean(train_acc[-episodes_per_epoch:])
        writer.add_scalar("Loss/train", loss_mean, epoch)
        writer.add_scalar("Acc/train", acc_mean, epoch)
        print(f'Ep {epoch}: Avg Train loss: {loss_mean}, Avg Train acc: {acc_mean}')
        curr_lr = lr_scheduler.get_last_lr()[-1]
        writer.add_scalar("lr", curr_lr, epoch)
        lr_scheduler.step()

        # Val
        if eval_each > 0 and epoch % eval_each == 0:
            model.eval()
            with torch.no_grad():
                for i in tqdm(range(episodes_per_epoch), total=episodes_per_epoch):
                    batch = valid_loader.GetSample(test_num_class, number_support, train_num_query)
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    x = model(x)
                    loss, acc = prototypical_loss(x, y, number_support, test_num_class, distance_fn)
                    val_loss.append(loss.item())
                    val_acc.append(acc.item())
                avg_loss = np.mean(val_loss[-episodes_per_epoch:])
                avg_acc = np.mean(val_acc[-episodes_per_epoch:])
            writer.add_scalar("Loss/val", avg_loss, epoch)
            writer.add_scalar("Acc/val", avg_acc, epoch)
            print(f"Avg Val Loss: {avg_loss}, Avg Val Acc: {avg_acc}")
            if avg_acc > best_acc:
                save_model(model, training_dir, f"model_best.pt")
                best_acc = avg_acc
                best_acc_ep = epoch
            early_stopping(avg_loss)
        if early_stopping.early_stop:
            print("Stopping due to EarlyStopping")
            break

    writer.flush()
    writer.close()
    duration = (datetime.datetime.now() - start_time)
    print(f"Training duration: {str(duration)}")
    print(f"Best val/acc {best_acc*100:.2f} on epoch {best_acc_ep}")
    return training_dir

def meta_test(model_path, episodes_per_epoch=100, dataset='mini_imagenet', use_gpu=False,
         test_num_query=15,
          test_num_class=5,
          number_support=5,
          distance_function="euclidean",
          images_size=None,
          images_ch=None) -> float:
    print("Building DataLoaders")
    test_loader = build_dataloaders_test(dataset, images_size, images_ch)
    device = build_device(use_gpu)

    print(f"Creating Prototype model on {device} from {model_path}")
    model = PrototypicalNetwork().to(device)
    model.load_state_dict(torch.load(model_path))

    distance_fn = build_distance_function(distance_function)

    val_acc = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(episodes_per_epoch), total=episodes_per_epoch):
            batch = test_loader.GetSample(test_num_class, number_support, test_num_query)
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x = model(x)
            _, acc = prototypical_loss(x, y, number_support, test_num_class, distance_fn)
            val_acc.append(acc.item())
        avg_acc = np.mean(val_acc[-episodes_per_epoch:])
    print(f"Avg Test Acc: {avg_acc}")
    return float(avg_acc)


def learn(model_path: str, data_path: str, images_size=None, images_ch=None, use_gpu=False):
    device = build_device(use_gpu)
    print(f"Creating Prototype model on {device} from {model_path}")
    model = PrototypicalNetwork().to(device)
    model.load_state_dict(torch.load(model_path))

    out_dir = init_savemodel("centroids")

    model.eval()
    with torch.no_grad():
        classes = list(os.listdir(data_path))
        for cl in tqdm(classes, total=len(classes)):
            class_samples = load_class_images(os.path.join(data_path, cl), (images_size, images_size), images_ch)
            class_samples = class_samples.to(device)
            embeddings = model(class_samples)
            centroids = embeddings.mean(dim=0)
            save_centroids(os.path.join(out_dir, cl), centroids.to('cpu'))
    print(f"Deployed to {out_dir}")

def predict(model_path: str, centroids_path: str, images_path: list, images_size=None, batch_size=4, use_gpu=False) -> list:
    device = build_device(use_gpu)
    print(f"Creating Prototype model on {device} from {model_path}")
    model = PrototypicalNetwork().to(device)
    model.load_state_dict(torch.load(model_path))

    print("Loading data")
    prototypes, classes = load_centroids(centroids_path)
    size = (images_size, images_size)

    model.eval()
    results = []
    with torch.no_grad():
        i = 0
        while i < len(images_path):
            batch = []
            images = []
            for j in range(batch_size):
                if i+j >= len(images_path): break
                images.append(images_path[i + j])
                batch.append(load_image(images_path[i + j], size).float())
            i += len(batch)
            if len(batch) == 0: break
            sample = torch.stack(batch)
            sample = sample.to(device)
            sample = model(sample)
            sample = sample.to("cpu")
            distances = euclidean_dist(prototypes, sample)
            for k, imp in zip(range(distances.shape[1]), images):
                classification = classes[torch.argmin(distances[:,k])]
                print(f"{imp}: {classification}")
                results.append((imp, classification))
    return results
