
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
torch.set_default_device(device)

imsize = 256
loader = transforms.Compose([
    transforms.ToTensor(),  # transform it into a torch tensor
]) 

batch_size=32
train_dataset = torchvision.datasets.ImageFolder("decrypt_dataset/")
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

#test_dataset = torchvision.datasets.ImageFolder("decrypt_dataset/test")
#trainloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
#                                          shuffle=True, num_workers=2)

classes = ('banksy', 'cezanne', 'dali', 'haring', 'kahlo', 'miyazaki', 'monet', 'picasso', 'rembrandt', 'toriyama', 'vanagogh', 'warhol', 'wei')


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 13)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        data = data.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0

print('Finished Training')