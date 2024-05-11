
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc

from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights, vgg16
from torch.utils.data import Dataset
import copy
import deeplake
from tqdm import tqdm
import numpy as np
import argparse 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("SETTING DEVICE AS", device)

imsize = 256
loader = transforms.Compose([
    transforms.ToTensor(),  # transform it into a torch tensor
]) 

batch_size=32
train_dataset = torchvision.datasets.ImageFolder("decrypt_dataset/", transform=loader)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.ImageFolder("decrypt_dataset/val")
trainloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

classes = ('banksy', 'cezanne', 'dali', 'haring', 'kahlo', 'miyazaki', 'monet', 'picasso', 'rembrandt', 'toriyama', 'vanagogh', 'warhol', 'wei')

'''
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=loader)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
'''
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        vgg16_model = vgg16(pretrained=True)
        vgg16_model.classifier = vgg16_model.classifier[:-1]
        vgg16_model.eval()
        vgg16_model.requires_grad_(False)
        self.fx = vgg16_model
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 13)

    def forward(self, x):
        x = self.fx(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Model().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    train_acc = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        #inputs = inputs.to(device)
        #data = data.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        outputs = outputs.softmax(dim=1)
        top1 = torch.argmax(outputs, dim=1)
        train_acc += torch.sum(top1 == labels)
        if i % 50 == 0 and i > 0:    # print every 2000 mini-batches
            print(f'[{epoch}, {i:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0
            print("accuracy", train_acc/(i * batch_size))
print('Finished Training')
with torch.no_grad():
    
