
#%%
from torch_snippets import *
from torchvision.datasets import MNIST
from torchvision import transforms

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"


#%% define transformation to use on images
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(lambda x: x.to(device))
])


#%% create train and validation dataset
trn_ds = MNIST("/content/", transform=img_transform, train=True, download=True)
val_ds = MNIST("/content/", transform=img_transform, train=False, download=True)


#%% define dataloaders
batch_size = 256
trn_dl = DataLoader(dataset=trn_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)

#%% defne network architecture
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(len(x), 1, 28, 28)
        return x
    
    
#%% visualize the model
from torchsummary import summary


#%%
model = AutoEncoder(3).to(device)
summary(model, (1,28,28))






# %%
