import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import yaml
from tqdm import tqdm

from src.prototypical_net import PrototypicalNetwork
from src.prototypical_loss import prototypical_loss
from src.data.MiniImagenetDataset import MiniImagenetDataset
from src.data.OmniglotDataset import OmniglotDataset
from src.data.Flowers102Dataset import Flowers102Dataset

# example train algo from https://github.com/pytorch/examples/blob/main/mnist/main.py
# Loading datasets from https://github.com/learnables/learn2learn/tree/master#learning-domains


def build_dataloaders(dataset='mini_imagenet'):
    if dataset == 'mini_imagenet':
        # Loading datasets
        train_loader = MiniImagenetDataset(mode='train', load_on_ram=True, download=True, tmp_dir="datasets")
        valid_loader = MiniImagenetDataset(mode='val', load_on_ram=True, download=False, tmp_dir="datasets")
        test_loader = MiniImagenetDataset(mode='test', load_on_ram=True, download=False, tmp_dir="datasets")
        return train_loader, valid_loader, test_loader
    elif dataset == 'omniglot':
        train_loader = OmniglotDataset(mode='train', load_on_ram=True, download=True, tmp_dir="datasets")
        valid_loader = OmniglotDataset(mode='val', load_on_ram=True, download=False, tmp_dir="datasets")
        test_loader = OmniglotDataset(mode='test', load_on_ram=True, download=False, tmp_dir="datasets")
        return train_loader, valid_loader, test_loader
    elif dataset == 'flowers102':
        train_loader = Flowers102Dataset(mode='train', load_on_ram=True, download=True, tmp_dir="datasets")
        valid_loader = Flowers102Dataset(mode='val', load_on_ram=True, download=False, tmp_dir="datasets")
        test_loader = Flowers102Dataset(mode='test', load_on_ram=True, download=False, tmp_dir="datasets")
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


def save_yaml_config(training_dir, config):
    with open(os.path.join(training_dir, "config.yaml"), 'w') as file:
        file.write(yaml.dump(config))


def init_savemodel() -> str:
    main_dir = "runs"
    if not os.path.exists(main_dir): os.mkdir(main_dir)
    i = 0
    build_dir = lambda idx: f"{main_dir}/train_{idx}"
    out_dir = build_dir(i)
    while os.path.exists(out_dir):
        out_dir = build_dir(i)
        i += 1
    os.mkdir(out_dir)
    return out_dir


def save_model(model, training_dir, name):
    torch.save(model.state_dict(), os.path.join(training_dir, name))


def train(dataset='mini_imagenet', epochs=300, use_gpu=False, lr=0.001,
          train_num_classes=30,
          test_num_class=5,
          train_num_query=15,
          number_support=5,
          episodes_per_epoch=50,
          optim_step_size=20,
          optim_gamma = 0.5,
          save_each=5):
    training_dir = init_savemodel()
    print(f"Writing to {training_dir}")
    writer = SummaryWriter(log_dir=training_dir)
    loaders = build_dataloaders(dataset)
    train_loader, valid_loader, test_loader = loaders
    device = build_device(use_gpu)
    print(f"Creating Prototype model on {device}")
    model = PrototypicalNetwork().to(device)
    #print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step_size, gamma=optim_gamma)

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
        "save_each": save_each
    }
    save_yaml_config(training_dir, config)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = -1
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
            loss, acc = prototypical_loss(x, y, number_support, train_num_classes)
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
        model.eval()
        for i in tqdm(range(episodes_per_epoch), total=episodes_per_epoch):
            batch = valid_loader.GetSample(test_num_class, number_support, train_num_query)
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x = model(x)
            loss, acc = prototypical_loss(x, y, number_support, test_num_class)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-episodes_per_epoch:])
        avg_acc = np.mean(val_acc[-episodes_per_epoch:])
        writer.add_scalar("Loss/val", avg_loss, epoch)
        writer.add_scalar("Acc/val", avg_acc, epoch)
        print(f"Avg Val Loss: {avg_loss}, Avg Val Acc: {avg_acc}")
        if avg_acc > best_acc:
            save_model(model, training_dir, f"model_best.pt")

    writer.flush()
    writer.close()
    duration = (datetime.datetime.now() - start_time)
    print(f"Training duration: {str(duration)}")

