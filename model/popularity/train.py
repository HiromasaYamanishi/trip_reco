# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
from re import I
from tkinter import image_names
from tkinter.ttk import LabeledScale

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pathlib import Path
import sys

dir = sys.argv[1]
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(dir, x),data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle = True, num_workers= 4)
                for x in ['train','valid']}
dataset_sizes = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle = True, num_workers = 4)
                for x in ["train","valid"]}
device = torch.device("cuda:0" if torch.cuda.is_avalilable() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs-1))
        print('-'*10)
        for phase in ['train','val']:
            if phase =="train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = input.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enables(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss +=loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss/ dataset_sizes[phase]
            epoch_acc = running_corrects.double()/ dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase =="val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elasped = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s".format(
            time_elasped//60, time_elasped% 60))

        print("Best val Acc: {:4f}".format(best_acc))

        model.load_state_dict(best_model_wts)
        return model









