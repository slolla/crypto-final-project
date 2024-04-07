import torch
import torchvision
import torchvision.transforms as T
from models import *
from trainers import *

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=T.Compose([T.ToTensor(), ]))
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=T.Compose([T.ToTensor(), ])) 
BATCH_SIZE = 64
# DEVICE = torch.device('cuda') use if on GPU
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)    
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True) 

def logging_fn(train_loss, test_loss, test_acc):
    print("train_loss", train_loss, "test_loss", test_loss, "test_acc", test_acc)

config = Config(
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        logging_fn=logging_fn,
        save_path = f"models/mnist_test.pt",
        epochs=5,
        lr=1e-2,
        
    )

trainer = Trainer(MNIST_model, config)
trainer.run_training_loop()