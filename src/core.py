import os

import learn2learn.vision.models
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.prototypical_net import PrototypicalNetwork
from src.prototypical_loss import prototypical_loss
from src.MiniImagenetDataset import MiniImagenetDataset

from learn2learn.vision.datasets import MiniImagenet
from learn2learn.data import MetaDataset, Taskset
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels


# example train algo from https://github.com/pytorch/examples/blob/main/mnist/main.py
# Loading datasets from https://github.com/learnables/learn2learn/tree/master#learning-domains


def build_dataloaders(dataset='mini_imagenet'):
    if dataset == 'mini_imagenet':
        train_loader = MiniImagenetDataset(mode='train', batch_size=16, load_on_ram=True, download=True, tmp_dir="datasets")
        valid_loader = MiniImagenetDataset(mode='val', batch_size=16, load_on_ram=True, download=False, tmp_dir="datasets")
        test_loader = MiniImagenetDataset(mode='test', batch_size=16, load_on_ram=True, download=False, tmp_dir="datasets")
        return train_loader, valid_loader, test_loader


def build_device(use_gpu=False):
    device = torch.device("cpu")
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("WWARN: Unable to set device to GPU because not available. Fallback to 'cpu'")
    return device


def train(dataset='mini_imagenet', epochs=300, use_gpu=False, lr=0.001,
          train_num_classes=30,
          train_num_query=15,
          number_support=5):
    loaders = build_dataloaders(dataset)
    train_loader, valid_loader, test_loader = loaders
    device = build_device(use_gpu)
    model = PrototypicalNetwork().to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_loss = []
    train_acc = []
    for epoch in range(epochs):
        model.train()
        # Train
        for i in range(100): # should be enough to cover batch*100 >= dataset_size
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
        print(f'Ep {epoch}, Avg Train loss: {np.mean(train_loss)}, Avg Train acc: {np.mean(train_acc)}')
        lr_scheduler.step()

        # TODO: validate

        # TODO: test

