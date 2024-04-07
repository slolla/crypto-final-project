import torch.nn as nn

MNIST_model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=(3, 3),),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Dropout(0.2),

    nn.Conv2d(32, 64, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(64),

    nn.Conv2d(64, 128, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.Dropout(0.2),

    nn.Conv2d(128, 64, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    
    nn.Flatten(),

    nn.Linear(25600, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)