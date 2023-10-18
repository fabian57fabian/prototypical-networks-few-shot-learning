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
from src.utils import get_torch_device
from src import __version__

print(f"***** Few-shot Learning with proto nets. v{__version__} *****")

# example train algo from https://github.com/pytorch/examples/blob/main/mnist/main.py
# Loading datasets from https://github.com/learnables/learn2learn/tree/master#learning-domains


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

def meta_train(cfg: dict) -> str:
    training_dir = init_savemodel()
    print(f"Writing to {training_dir}")
    writer = SummaryWriter(log_dir=training_dir)

    print("Building DataLoaders")
    train_loader, valid_loader = build_dataloaders(cfg["data"], cfg["imgsz"], cfg["channels"])
    device = get_torch_device(cfg["device"])

    print(f"Creating Prototype model on {device}")
    model = PrototypicalNetwork()
    if cfg["model"] is not None:
        if not os.path.exists(cfg["model"]):
            raise Exception(f"Model path to load does not exist: {cfg['model']}")
        print(f"Loading model {cfg['model']}")
        load_model(model, cfg["model"])
    model = model.to(device)

    # Optimizer, lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['adam_lr'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['adam_step'], gamma=cfg['adam_gamma'])
    # Callbacks
    early_stopping = EarlyStopping(patience=cfg['patience'], delta=cfg['patience_delta'])

    distance_fn = build_distance_function(cfg['metric'])

    # Save config
    save_yaml_config(training_dir, cfg)

    train_loss = []
    train_acc = []
    best_acc = -1
    best_acc_ep = -1
    start_time = datetime.datetime.now()

    print(f"Startring training at {str(start_time)}")
    NC, NS, NQ = cfg['num_way'], cfg['shot'], cfg['query']
    for epoch in range(cfg['episodes'] + 1):
        model.train()
        # Train
        for i in tqdm(range(cfg['iterations']), total=cfg['iterations']): # should be enough to cover batch*100 >= dataset_size
            batch = train_loader.GetSample(NC, NS, NQ)
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x = model(x)
            loss, acc = prototypical_loss(x, y, NS, NC, distance_fn)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        if cfg['save_period'] > 0 and epoch % cfg['save_period'] == 0:
            save_model(model, training_dir, f"model_{epoch}.pt")
        loss_mean, acc_mean = np.mean(train_loss[-cfg['iterations']:]),np.mean(train_acc[-cfg['iterations']:])
        writer.add_scalar("Loss/train", loss_mean, epoch)
        writer.add_scalar("Acc/train", acc_mean, epoch)
        print(f'Ep {epoch}: Avg Train loss: {loss_mean}, Avg Train acc: {acc_mean}')
        curr_lr = lr_scheduler.get_last_lr()[-1]
        writer.add_scalar("lr", curr_lr, epoch)
        lr_scheduler.step()

        # Val
        if cfg['eval_each'] > 0 and epoch % cfg['eval_each'] == 0:
            avg_loss, avg_acc = evaluate(cfg, model, valid_loader, device, distance_fn)
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


def evaluate(cfg, model, data_loader, device, distance_fn):
    NC, NS, NQ = cfg['val_num_way'], cfg['shot'], cfg['query']
    losses, accs = [], []
    print("Starting validation")
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(cfg['iterations']), total=cfg['iterations']):
            batch = data_loader.GetSample(NC, NS, NQ)
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x = model(x)
            loss, acc = prototypical_loss(x, y, NS, NC, distance_fn)
            losses.append(loss.item())
            accs.append(acc.item())
        avg_loss = np.mean(losses)
        avg_acc = np.mean(accs)
    return avg_loss, avg_acc

def meta_test(cfg) -> float:
    print("Building DataLoaders")
    test_loader = build_dataloaders_test(cfg["data"], cfg["imgsz"], cfg["channels"])
    device = get_torch_device(cfg["device"])

    print(f"Creating Prototype model on {device} from {cfg['model']}")
    model = PrototypicalNetwork().to(device)
    model.load_state_dict(torch.load(cfg['model']))

    distance_fn = build_distance_function(cfg['metric'])

    _, avg_acc = evaluate(cfg, model, test_loader, device, distance_fn)
    print(f"Avg Test Acc: {avg_acc}")
    return float(avg_acc)


def learn(cfg):
    device = get_torch_device(cfg["device"])
    print(f"Creating Prototype model on {device} from {cfg['model']}")
    model = PrototypicalNetwork().to(device)
    model.load_state_dict(torch.load(cfg['model']))

    out_dir = init_savemodel("centroids")

    model.eval()
    with torch.no_grad():
        classes = list(os.listdir(cfg['data']))
        for cl in tqdm(classes, total=len(classes)):
            class_samples = load_class_images(os.path.join(cfg['data'], cl), (cfg['imgsz'], cfg['imgsz']), cfg['channels'])
            class_samples = class_samples.to(device)
            embeddings = model(class_samples)
            centroids = embeddings.mean(dim=0)
            save_centroids(os.path.join(out_dir, cl), centroids.to('cpu'))
    print(f"Deployed to {out_dir}")

def predict(cfg) -> list:
    device = get_torch_device(cfg["device"])
    print(f"Creating Prototype model on {device} from {cfg['model']}")
    model = PrototypicalNetwork().to(device)
    model.load_state_dict(torch.load(cfg['model']))

    print("Loading data")
    prototypes, classes = load_centroids(cfg['centroids'])
    size = (cfg['imgsz'], cfg['imgsz'])
    batch_size = 4  # set to speed up
    images_path = [cfg['data']] if os.path.isfile(cfg['data']) else [os.path.join(cfg['data'], f) for f in sorted(os.listdir(cfg['data']))]

    model.eval()
    results = []
    with torch.no_grad():
        i = 0
        while i < len(images_path):
            batch = []
            images = []
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
