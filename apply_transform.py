# For tips on running notebooks in Google Colab, see
# https://pytorch.org/tutorials/beginner/colab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights, vgg16

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# desired size of the output image
imsize = 512  # use small size if no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

images_dir = "images/final_images/glazed"
transform_dir = "images/transformed_glaze_images"
imgs = os.listdir(images_dir)
for content_filename in imgs:
    content_img = image_loader(f"{images_dir}/{content_filename}")
    input_img = content_img.clone()

    im = content_img.detach().squeeze().cpu().numpy() * 255
    im = im.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))
    im = Image.fromarray(im)
    im.save(f"{transform_dir}/{content_filename}")