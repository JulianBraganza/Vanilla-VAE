import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_utils import Flatten, UnFlatten

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    
    def generate_latent_params(self, x):
        params = self.encoder(x)
        d = params.shape[1]
        mu = params[:,:(d // 2)]
        log_var = params[:,(d // 2):]
        return mu, log_var
    
    def sample_latent(self, mu, log_var):
        eps = torch.randn_like(mu)
        return mu + eps*torch.exp(0.5*log_var)
    
    def forward(self, x):
        mu, log_var = self.generate_latent_params(x)
        z = self.sample_latent(mu,log_var)
        return self.decoder(z), mu, log_var
    
def loss_function_BCE(recon_x, x, mu, logvar):
    #MSE = F.mse_loss(recon_x, x,reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x,reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (BCE + KLD)/x.shape[0]

def loss_function_MSE(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    #MSE = F.binary_cross_entropy(recon_x, x,reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (MSE + KLD)/x.shape[0]

def reconstruct(model,x):
    with torch.no_grad():
        return model.eval()(x)[0]
    
def generate_samples(model,num_images,lat_dim):
    ep = torch.randn(num_images,lat_dim).to(device)
    with torch.no_grad():
        return model.eval().decoder(ep)
    

def train(model,epoch,l_rate,loss_function,train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    for i in range(epoch):
        for batch_idx, (data, _) in enumerate(train_loader):
            #print(batch_idx)
            data = data.to(torch.device('cuda'), non_blocking=True)
            optimizer.zero_grad()
    
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch: {i}, Loss: {loss}')
        print(f'Epoch: {i}, Loss: {loss}')
    print(f'Loss: {loss}')
