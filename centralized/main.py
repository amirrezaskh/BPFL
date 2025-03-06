import os
import json
import torch
import random
import requests
import numpy as np
import torch.utils
import torchvision
from torch import nn
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2, ToTensor
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    

def train():
    losses = []
    model.train()
    model.to(device)
    for _, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return sum(losses) / len(losses)

def validate():
    model.eval()
    with torch.no_grad():
        X, y = next(iter(val_dataloader))
        pred = model(X)
        val_loss = loss_fn(pred, y).item()
    return val_loss

learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 32
epochs = 100

transforms = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToTensor()
        ])

train_dataset = datasets.CIFAR10(
    root="../nodes/data",
    train=True,
    transform=transforms
)

test = datasets.CIFAR10(
    root="../nodes/data",
    train=False,
    transform=ToTensor()
)

test_indexes = [i for i in range(len(test))]

val_dataset = torch.utils.data.Subset(test, test_indexes[:len(test_indexes)//2])
test_dataset = torch.utils.data.Subset(test, test_indexes[len(test_indexes)//2:])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet18Classifier(num_classes=10).to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)

torch.backends.cudnn.benchmark = True

data = []

for e in range(epochs):
    print(f"Epoch: {e+1}")
    train_loss = train()
    val_loss = validate()
    data.append({
        "epoch" : e + 1,
        "train_loss" : train_loss,
        "val_loss" : val_loss
    })
    scheduler.step(val_loss, epoch=e)
    torch.cuda.empty_cache()

with open("./res.json", "w") as f:
    f.write(json.dumps(data, indent=2))
