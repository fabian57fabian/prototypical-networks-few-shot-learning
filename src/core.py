import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import yaml
from tqdm import tqdm

from src.prototypical_net import PrototypicalNetwork
from src.prototypical_loss import prototypical_loss, euclidean_dist, cosine_dist
from src.data.MiniImagenetDataset import MiniImagenetDataset
from src.data.OmniglotDataset import OmniglotDataset
from src.data.Flowers102Dataset import Flowers102Dataset
from src.data.CustomDataset import CustomDataset
from src.data.AbstractClassificationDataset import load_class_images, load_image

from src.data.centroids import load_centroids, save_centroids

# example train algo from https://github.com/pytorch/examples/blob/main/mnist/main.py
# Loading datasets from https://github.com/learnables/learn2learn/tree/master#learning-domains


def get_allowed_base_datasets_names() -> list:
    return ["mini_imagenet", "omniglot", "flowers102"]


def build_dataloaders(dataset='mini_imagenet', size=None, channels=None, only_test=False):
    if dataset == 'mini_imagenet':
        # Loading datasets
        test_loader = MiniImagenetDataset(mode='test', load_on_ram=True, download=True, images_size=size, tmp_dir="datasets")
        if only_test: return None, None, test_loader
        train_loader = MiniImagenetDataset(mode='train', load_on_ram=True, download=False, images_size=size, tmp_dir="datasets")
        valid_loader = MiniImagenetDataset(mode='val', load_on_ram=True, download=False, images_size=size, tmp_dir="datasets")
        return train_loader, valid_loader, test_loader
    elif dataset == 'omniglot':
        test_loader = OmniglotDataset(mode='test', load_on_ram=True, download=True, images_size=size, tmp_dir="datasets")
        if only_test: return None, None, test_loader
        train_loader = OmniglotDataset(mode='train', load_on_ram=True, download=False, images_size=size, tmp_dir="datasets")
        valid_loader = OmniglotDataset(mode='val', load_on_ram=True, download=False, images_size=size, tmp_dir="datasets")
        return train_loader, valid_loader, test_loader
    elif dataset == 'flowers102':
        test_loader = Flowers102Dataset(mode='test', load_on_ram=True, download=True, images_size=size, tmp_dir="datasets")
        if only_test: return None, None, test_loader
        train_loader = Flowers102Dataset(mode='train', load_on_ram=True, download=False, images_size=size, tmp_dir="datasets")
        valid_loader = Flowers102Dataset(mode='val', load_on_ram=True, download=False, images_size=size, tmp_dir="datasets")
        return train_loader, valid_loader, test_loader
    elif os.path.exists(dataset):
        test_loader = CustomDataset(mode='test', load_on_ram=True, images_size=size, image_ch=channels, dataset_path=dataset)
        if only_test: return None, None, test_loader
        train_loader = CustomDataset(mode='train', load_on_ram=True, images_size=size, image_ch=channels, dataset_path=dataset)
        valid_loader = CustomDataset(mode='val', load_on_ram=True, images_size=size, image_ch=channels, dataset_path=dataset)
        return train_loader, valid_loader, test_loader
    assert False, "dataset unknown"


def build_device(use_gpu=False):
    device = torch.device("cpu")
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("WWARN: Unable to set device to GPU because not available. Fallback to 'cpu'")
    return device


def build_distance_function(distance_function: str):
    assert distance_function in ["euclidean", "cosine"]
    if distance_function == "euclidean":
        return euclidean_dist
    elif distance_function == "cosine":
        return cosine_dist
    assert False, "Wrong distance function supplied"

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

def load_model(path):
    model = torch.load(path)

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
          eval_each=1):
    training_dir = init_savemodel()
    print(f"Writing to {training_dir}")
    writer = SummaryWriter(log_dir=training_dir)
    loaders = build_dataloaders(dataset, images_size, images_ch)
    train_loader, valid_loader, test_loader = loaders
    device = build_device(use_gpu)
    print(f"Creating Prototype model on {device}")
    model = PrototypicalNetwork().to(device)
    #print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step_size, gamma=optim_gamma)

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
        "save_each": save_each
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
    for epoch in range(epochs):
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

    writer.flush()
    writer.close()
    duration = (datetime.datetime.now() - start_time)
    print(f"Training duration: {str(duration)}")
    print(f"Best val/acc {best_acc*100:.2f} on epoch {best_acc_ep}")

def meta_test(model_path, episodes_per_epoch=100, dataset='mini_imagenet', use_gpu=False,
         test_num_query=15,
          test_num_class=5,
          number_support=5,
          distance_function="euclidean",
          images_size=None,
          images_ch=None):
    _, _, test_loader = build_dataloaders(dataset, images_size, images_ch, only_test=True)
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

def predict(model_path: str, centroids_path: str, images_path: list, images_size=None, batch_size=4, use_gpu=False):
    device = build_device(use_gpu)
    print(f"Creating Prototype model on {device} from {model_path}")
    model = PrototypicalNetwork().to(device)
    model.load_state_dict(torch.load(model_path))

    prototypes, classes = load_centroids(centroids_path)

    # TODO: optimize with batch size

    size = (images_size, images_size)

    model.eval()
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
