import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def plot_imgs(images):
    im = make_grid(images,nrow=len(images)//2)
    return plt.imshow(np.transpose(im.cpu().numpy(), (1, 2, 0)))
    
