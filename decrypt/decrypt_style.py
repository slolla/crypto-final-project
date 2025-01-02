import deeplake
from PIL import Image
import numpy as np
import os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

# developing training pipeline from 
# https://docs.activeloop.ai/examples/dl/tutorials/training-models/splitting-datasets-training

'''
This function should take images and determine which
style was used to mask each image. Should print T, such 
that T is the masking style
'''
def decrypt_img_styles(model, input_path):
    enc_imgs = os.listdir(input_path)
    for i in enc_imgs:
        input_img = image_loader(input_path+i)
        with torch.no_grad():
            outputs = model(input_img)
        T = torch.argmax(outputs, dim=1)
        print(f'Predicted T = {T} for input img {i}')

'''
Train a model to, given a series of masked images and the style (T)
that was used to create the mask, recognize when a style T has been used.
Should print the accuracy after each epoch, then return the trained model.
'''
def train_style_recognition(train_view, test_view, tform, num_epochs):
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    loader = train_view.pytorch(num_workers = 0, shuffle = True, transform = {'images': tform, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.squeeze(1).to(device)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # i'm not sure if the * inputs.size(0) is needed
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    
    model.eval()
    test_loader = test_view.pytorch(num_workers = 0, transform = {'images': tform, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).sum().item() / len(labels)
            print(f'Accuracy on test set: {accuracy}')
    return model

def image_loader(image_name, imsize, device):
    loader = transforms.Compose([
        transforms.ToTensor(),  # transform it into a torch tensor
        transforms.Resize(imsize),  # scale imported image
        transforms.CenterCrop(imsize),
    ]) 
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU
   
    # read inputs from our encrypted ciphertexts (path name is temp)
    # split 0.8 training/0.2 testing
    path = "./enc_outputs/"
    enc_imgs = os.listdir(path)
    ds = []
    for i in enc_imgs:
        ds.append(image_loader(path+i, imsize, device))

    images = torch.stack([i for i in ds])

    len_ds = len(images)
    train_frac = 0.8
    x = list(range(len_ds))
    random.shuffle(x)
    x_lim = round(train_frac*len(images))
    train_indices, test_indices = x[:x_lim], x[x_lim:]

    train_view = images[train_indices]
    test_view = images[test_indices]

    tform = transforms.Compose([
        transforms.CenterCrop(10),
        transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
        transforms.Normalize([0.5], [0.5]),
    ])

    # TRAIN THE MODEL
    model = train_style_recognition(train_view, test_view, tform, 10000)

    # now we know how to find T, so we can use the model to decrypt
    # the labels for other imgs if we need to
    # decrypt_img_styles(model, input_path)



if __name__ == "__main__":
    main()