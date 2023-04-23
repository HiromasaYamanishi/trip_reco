from __future__ import print_function, division

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from typing import OrderedDict
from tqdm import tqdm
import os
import cv2
from PIL import Image
import copy
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import sys
sys.path.append('..')
from data.jalan.preprocessing import Path

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(device)

class PlaceDataset(Dataset):
    '''
    観光地の画像データセット
    '''
    def __init__(self, transform, mode) -> None:
        super().__init__()
        self.path = Path()
        self.df = pd.read_csv(self.path.df_experience_light_path)
        self.mode = mode
        if mode=='train':
            self.df = self.df[self.df['valid'] >1].reset_index(drop=True)
        elif mode=='val':
            self.df = self.df[self.df['valid'] == 0].reset_index(drop=True)
        else:
            self.df = self.df[self.df['valid']==1].reset_index(drop=True)
        self.spots = self.df['spot_name'].values
        #self.page_view = self.df['page_view'].values/self.df['page_view'].max()
        self.page_view = np.log10(self.df['review_count'])
        self.transform = transform

    def __getitem__(self, index):
        spot_name = self.spots[index].replace('/','')
        img_file_path = os.path.join(self.path.jalan_image_dir , spot_name + '_0.jpg')
        img = Image.open(img_file_path)
        img = img.convert(mode='RGB')
        if self.transform:
            #print(self.transform)
            img = self.transform(img)

        page_view = self.page_view[index]
        page_view = torch.tensor(page_view)
        return img, page_view.to(torch.float64)

    def __len__(self):
        return len(self.spots)

def get_transforms():
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
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def get_datasets(transforms):
    train_dataset = PlaceDataset(transform=transforms['train'], mode = 'train')
    valid_dataset = PlaceDataset(transform=transforms['val'], mode = "val")
    test_dataset = PlaceDataset(transform=transforms['test'], mode = "test")
    datasets = {'train': train_dataset,
                'val': valid_dataset,
                'test': test_dataset}

    dataset_sizes = {'train': len(train_dataset),
                    'val': len(valid_dataset),
                    'test': len(test_dataset)}

    return datasets, dataset_sizes

def get_dataloaders(datasets):
    train_loader = torch.utils.data.DataLoader(datasets['train'], batch_size = 16, shuffle = True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(datasets['val'], batch_size = 16, shuffle = True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size = 16, shuffle = True, num_workers = 4)
    dataloaders = {'train': train_loader,
                    'val': valid_loader,
                    'test': test_loader}
    return dataloaders

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    writer = SummaryWriter(log_dir= "log")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e5

    for epoch in range(num_epochs):
        print(f'Epoch{epoch}/{num_epochs}')
        print('-' * 10)

        for phase in ["train", "val"]:
            with tqdm(dataloaders[phase]) as pbar:
                pbar.set_description(f'[Phase: {phase} Epoch: {epoch+1}/{num_epochs}]')
                if phase == 'train':
                    model.train()

                else:
                    model.eval()

                running_loss = 0.0

                for inputs, labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)


                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase=="train"):
                        outputs = model(inputs).to(torch.float64)
                        labels = labels.reshape(-1,1)
                        #print('outputs',outputs, 'labels',labels)
                        loss = criterion(outputs,labels )

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                if phase == "train":
                    scheduler.step()

                pbar.set_postfix(
                    OrderedDict(
                        Loss=loss.item()
                    )
                )
                epoch_loss = running_loss/dataset_sizes[phase]
                if phase == 'train':
                    writer.add_scalar('../data/log/train/loss', epoch_loss, epoch)
                else:
                    writer.add_scalar('../data/log/valid/loss', epoch_loss, epoch)
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                if phase == "val" and epoch_loss < best_loss:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_loss = epoch_loss

                print()

    time_elasped = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elasped //60 , time_elasped //60
    ))
    print("Best Loss: {:.4f}".format(best_loss))
    torch.save(best_model_wts, '../data/best_model_image.bin')
    print('saved')
    model.load_state_dict(best_model_wts)
    return model

def test_model(model):
    with torch.no_grad():
        gt_all, pred_all = [], []
        for phase in ['val','test']:
            dataloader = dataloaders[phase]
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                out = model(inputs)
                gt_all.append(labels.flatten().cpu().detach().numpy().copy())
                pred_all.append(out.flatten().cpu().detach().numpy().copy())
        gt_all = np.concatenate(gt_all)
        pred_all = np.concatenate(pred_all)
        plt.scatter(gt_all, pred_all)
        plt.savefig('image_cor.png')
        print(np.corrcoef(gt_all, pred_all))

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if __name__ == "__main__":
    transforms = get_transforms()
    datasets, dataset_sizes = get_datasets(transforms)
    dataloaders = get_dataloaders(datasets)

    model_ft = models.resnet18(pretrained = True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, 1)

    model_ft = model_ft.to(device)
    criterion = nn.MSELoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, dataloaders, dataset_sizes,criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

    test_model(model_ft)