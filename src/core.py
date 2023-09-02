import os

import learn2learn.vision.models
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.prototypical_net import PrototypicalNetwork
from src.prototypical_loss import prototypical_loss

from learn2learn.vision.datasets import MiniImagenet
from learn2learn.data import MetaDataset, Taskset
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels


# example train algo from https://github.com/pytorch/examples/blob/main/mnist/main.py
# Loading datasets from https://github.com/learnables/learn2learn/tree/master#learning-domains


def build_dataloaders(dataset='mini_imagenet',
                      train_numway=30,
                      train_kquery=15,
                      test_numway=30,
                      test_kquery=15):
    if not os.path.exists('tmp'): os.mkdir('tmp')
    if dataset == 'mini_imagenet':
        tmp_dir = 'tmp/mini_imagenet'
        if not os.path.exists(tmp_dir): os.mkdir(tmp_dir)
        dts_train = MiniImagenet(root=tmp_dir, mode='train', download=True)
        dts_valid = MiniImagenet(root=tmp_dir, mode='validation', download=True)
        dts_test = MiniImagenet(root=tmp_dir, mode='test', download=True)

        # Train
        train_dataset = MetaDataset(dts_train)
        train_transforms = [
            NWays(train_dataset, train_numway),
            KShots(train_dataset, train_kquery),
            LoadData(train_dataset),
            RemapLabels(train_dataset),
        ]
        train_tasks = Taskset(train_dataset, task_transforms=train_transforms)
        train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

        # Valid
        valid_dataset = MetaDataset(dts_valid)
        valid_transforms = [
            NWays(valid_dataset, test_numway),
            KShots(valid_dataset, test_kquery),
            LoadData(valid_dataset),
            RemapLabels(valid_dataset),
        ]
        valid_tasks = Taskset(
            valid_dataset,
            task_transforms=valid_transforms,
            num_tasks=200,
        )
        valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

        # Test
        test_dataset = MetaDataset(dts_test)
        test_transforms = [
            NWays(test_dataset, test_numway),
            KShots(test_dataset, test_kquery),
            LoadData(test_dataset),
            RemapLabels(test_dataset),
        ]
        test_tasks = Taskset(
            test_dataset,
            task_transforms=test_transforms,
            num_tasks=2000,
        )
        test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)

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
          train_numway=30,
          train_kquery=15,
          test_numway=30,
          test_kquery=15,
          number_support=5):
    loaders = build_dataloaders(dataset, train_numway, train_kquery, test_numway, test_kquery)
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
            batch = next(iter(train_loader))
            optimizer.zero_grad()
            x, y = batch
            x = x.squeeze(0).to(device)
            y = y.to(device)
            model_out = model(x)
            loss, acc = prototypical_loss(model_out, y, number_support)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        print(f'Ep {epoch}, Avg Train loss: {np.mean(train_loss)}, Avg Train acc: {np.mean(train_acc)}')
        lr_scheduler.step()

        # TODO: validate

        # TODO: test

