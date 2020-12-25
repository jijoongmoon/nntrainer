#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# @file	main.py
# @date	08 Dec 2020
# @brief	This is VGG16 Example using PyTorch
# @see		https://github.com/nnstreamer/nntrainer
# @author	Parichay Kapoor <pk.kapoor@samsung.com>
# @bug		No known bugs except for NYI items
#
# This is based on official pytorch examples

'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

device = 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)

indices = np.arange(len(trainset))
train_indices  = list(range(0,len(trainset), 100))
test_indices = list(range(0,len(trainset), 101))

# Warp into Subsets and DataLoaders
train_dataset = torch.utils.data.Subset(trainset, train_indices)
test_dataset = torch.utils.data.Subset(trainset, test_indices)

trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=1)

testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=1)

# Model

cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Linear(256, 100))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

print('Building model..')
net = VGG('VGG16')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def inference():
    net.eval()
    net.load_state_dict(torch.load("./vgg_cnn.pt"))
    correct = 0
    count =0
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        count = count+1
        break;
    print(count)
    output = net(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

    print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(testloader.dataset)))
>>>>>>> 9dc5f2e... [ MNIST ] Add inference with float

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    state = {'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch VGG Example')    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--inference', action='store_true', default=False,
                        help='inference?')
    args = parser.parse_args()

    if (args.inference):
        inference()
    else:
        for epoch in range(1):
            train(epoch)
            if args.save_model:
                torch.save(net.state_dict(), "vgg_cnn.pt")
        
    # test(epoch)
