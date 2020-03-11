import torch.nn as nn
import torch.nn.functional as F
class Flatten(nn.Module):
    
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    
    def __init__(self,n_channels,size1,size2):
        super().__init__()
        self.n_channels = n_channels
        self.size1 = size1
        self.size2 = size2
        
    def forward(self, input):
        return input.view(input.size(0),self.n_channels,self.size1,self.size2)
    

class ResBlock(nn.Module):

    def __init__(self, outer_dim, inner_dim):
        super().__init__()
        self.net = nn.Sequential(    
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(outer_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(),
            nn.Conv2d(outer_dim, inner_dim, 3, 1, 1))
    def forward(self, input):
        return input + self.net(input)

def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')