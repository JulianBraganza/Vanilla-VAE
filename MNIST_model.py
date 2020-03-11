import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from VAE import VAE, train ,loss_function_BCE
from nn_utils import Flatten, UnFlatten

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder = nn.Sequential(nn.Conv2d(1,8,3),
                        nn.ReLU(),
                        nn.Conv2d(8,16,3),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(16,32,3),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        Flatten(),
                        nn.Linear(800,256),
                        nn.ReLU(),
                        nn.Linear(256,40))

decoder = nn.Sequential(nn.Linear(20,256),
                        nn.ReLU(),
                        nn.Linear(256,800),
                        UnFlatten(32,5),
                        nn.Upsample(scale_factor=2),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32,16,3),
                        nn.Upsample(scale_factor=2),
                        nn.ConvTranspose2d(16,8,3),
                        nn.ReLU(),
                        nn.ConvTranspose2d(8,1,3),
                        nn.Sigmoid())


transform = transforms.ToTensor()
train_data = datasets.MNIST(root='C:/Users/Julian/Documents/Data', train=True, download=True, transform=transform)


train_loader = DataLoader(train_data, batch_size=128, shuffle=True,pin_memory=True)

mnistVAE_bce = VAE(encoder,decoder).to(device)
train(mnistVAE_bce,epoch=50,l_rate=0.0001,loss_function=loss_function_BCE,train_loader=train_loader)


