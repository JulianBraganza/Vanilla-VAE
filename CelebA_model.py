import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from VAE import VAE, train ,loss_function_BCE, loss_function_MSE
from nn_utils import Flatten, UnFlatten, ResBlock
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




encoder = nn.Sequential(nn.Conv2d(3,16,1),
                        ResBlock(16,16), ResBlock(16,16), ResBlock(16,16),
                        nn.AvgPool2d(2,2),
                        nn.Conv2d(16,32,1),
                        ResBlock(32,32), ResBlock(32,32), ResBlock(32,32),
                        nn.AvgPool2d(2,2),
                        nn.Conv2d(32,64,1),
                        ResBlock(64,64), ResBlock(64,64), ResBlock(64,64),
                        nn.AvgPool2d(2,2),
                        nn.Conv2d(64,128,1),
                        ResBlock(128,128), ResBlock(128,128), ResBlock(128,128),
                        nn.AvgPool2d(2,2),
                        nn.Conv2d(128,256,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(256),
                        Flatten(),
                        nn.Linear(4096,512))

decoder = nn.Sequential(nn.Linear(256,4096),
                        UnFlatten(256,4,4),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,128,1),
                        nn.Upsample(scale_factor=2),
                        ResBlock(128,128), ResBlock(128,128), ResBlock(128,128),
                        nn.Conv2d(128,64,1),
                        nn.Upsample(scale_factor=2),
                        ResBlock(64,64), ResBlock(64,64), ResBlock(64,64),
                        nn.Conv2d(64,32,1),
                        nn.Upsample(scale_factor=2),
                        ResBlock(32,32), ResBlock(32,32), ResBlock(32,32),
                        nn.Conv2d(32,16,1),
                        nn.Upsample(scale_factor=2),
                        ResBlock(16,16), ResBlock(16,16), ResBlock(16,16),
                        nn.Conv2d(16,3,1),
                        nn.Sigmoid())



transform = transforms.Compose([
            transforms.CenterCrop(150),
            transforms.Resize(64),
            transforms.ToTensor()])


train_data = datasets.CelebA(root='../Data', download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, pin_memory=True,num_workers=4)

for images,_ in train_loader: 
    break

im = make_grid(images,nrow=10)
#im = make_grid(images,nrow=2)
plt.figure(figsize=(10,4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.show()


images_re = reconstruct_image(cifar_VAE,images.to(device)) 
im = make_grid(torch.cat((images, images_re.cpu()), 0),nrow=10)
#im = make_grid(images,nrow=2)
plt.figure(figsize=(10,4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.show()



celeba_VAE = VAE(encoder,decoder).to(device)

torch.save(celeba_VAE.state_dict(), 'celeba_VAE.pt')

celeba_VAE.load_state_dict(torch.load('celeba_VAE.pt'));
celeba_VAE.eval() # be sure to run this step!


train(celeba_VAE,epoch=10,l_rate=0.0001,loss_function=loss_function_BCE,train_loader=train_loader)



images_re = reconstruct_image(celeba_VAE,images.to(device)) 
im = make_grid(torch.cat((images, images_re.cpu()), 0),nrow=10)
#im = make_grid(images,nrow=2)
plt.figure(figsize=(10,4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
plt.show()


new_imgs = generate_samples(celeba_VAE,12,256)

plt.figure()
plot_imgs(new_imgs)
#plt.savefig('10epoch2')
plt.show()

plt.figure()
plt.imshow(np.transpose(p.view(3,64,64).cpu().numpy(),(1,2,0)))
plt.show()


plt.figure()
plt.imshow(np.transpose(images[0].cpu().numpy(),(1,2,0)))
plt.show()