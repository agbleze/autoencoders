
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


#%% define training func
def train_batch(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss

#%% define validation func
@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss


#%% initialize model, loss criterion, optimizer
model = AutoEncoder(3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)


#%% train the model
num_epochs = 5
log = Report(num_epochs)

for epoch in range(num_epochs):
    N = len(trn_dl)
    for ix, (data, _) in enumerate(trn_dl):
        loss = train_batch(data, model, criterion, optimizer)
        log.record(pos=(epoch + (ix+1)/N), trn_loss=loss, end="\r")
    N = len(val_dl)
    for ix, (data, _) in enumerate(val_dl):
        loss = validate_batch(data, model, criterion)
        log.record(pos=(epoch + (ix+1)/N), val_loss=loss, end="\r")
    log.report_avgs(epoch+1)
    log.plot_epochs(log=True)

#%% visualize the model on val_ds
for _ in range(3):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    _im = model(im[None])[0]
    fig, ax = plt.subplots(1, 2, figsize=(3,3))
    show(im[0], ax=ax[0], title="input")
    show(_im[0], ax=ax[1], title="prediction")
    plt.tight_layout()
    plt.show()

# %%  experiment with varying size of nodes in latent_dim
def train_aec(latent_dim):
    model = AutoEncoder(3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    num_epochs = 5
    log = Report(num_epochs)

    for epoch in range(num_epochs):
        N = len(trn_dl)
        for ix, (data, _) in enumerate(trn_dl):
            loss = train_batch(data, model, criterion, optimizer)
            log.record(pos=(epoch + (ix+1)/N), trn_loss=loss, end="\r")
        N = len(val_dl)
        for ix, (data, _) in enumerate(val_dl):
            loss = validate_batch(data, model, criterion)
            log.record(pos=(epoch + (ix+1)/N), val_loss=loss, end="\r")
        log.report_avgs(epoch+1)
        log.plot_epochs(log=True)
    return model
0246723811
#%%    
aecs = [train_aec(dim) for dim in [50, 2, 3, 5, 10]]
    
#%% visualize the model on val_ds
for _ in range(10):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    fig, ax = plt.subplots(1, len(aecs)+1, figsize=(10,4))
    ax = iter(ax.flat)
    show(im[0], ax=next(ax), title="input") 
    for model in aecs:
        _im = model(im[None])[0] 
        show(_im[0], ax=next(ax), title=f"prediction\nlatent-dim:{model.latent_dim}")
        plt.tight_layout()
        plt.show()
    
# %%  ####### autoencoders with convolutional layer  #####
class ConvAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
#%%
model = ConvAutoEncoder().to(device)
summary(model, (1,28,28))

#%%
# %%  train conv for encoder / decoder
def train_convaec(model, criterion, optimizer, trn_dl, val_dl, num_epochs):
    #model = AutoEncoder(3).to(device)
    #criterion = nn.MSELoss()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    #num_epochs = 5
    log = Report(num_epochs)

    for epoch in range(num_epochs):
        N = len(trn_dl)
        for ix, (data, _) in enumerate(trn_dl):
            loss = train_batch(data, model, criterion, optimizer)
            log.record(pos=(epoch + (ix+1)/N), trn_loss=loss, end="\r")
        N = len(val_dl)
        for ix, (data, _) in enumerate(val_dl):
            loss = validate_batch(data, model, criterion)
            log.record(pos=(epoch + (ix+1)/N), val_loss=loss, end="\r")
        log.report_avgs(epoch+1)
    log.plot_epochs(log=True)
    return model
# %%
model = ConvAutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


# %%
model = train_convaec(model=model, criterion=criterion, optimizer=optimizer,
              trn_dl=trn_dl, val_dl=val_dl, num_epochs=10
              )
#%% visualize the model on val_ds
def viz_pred(model, val_ds):
    for _ in range(3):
        ix = np.random.randint(len(val_ds))
        im, _ = val_ds[ix]
        _im = model(im[None])[0]
        fig, ax = plt.subplots(1, 2, figsize=(3,3))
        show(im[0], ax=ax[0], title="input")
        show(_im[0], ax=ax[1], title="prediction")
        plt.tight_layout()
        plt.show()
        
#%%
viz_pred(model=model, val_ds=val_ds)

        
# %% visualizaing clusters of similar images using autoencoders  ##
latent_vectors = []
classes = []

#%%
for im, clss in val_dl:
    latent_vectors.append((model.encoder(im).view(len(im), -1)))
    classes.extend(clss)
latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy()

#%%
from sklearn.manifold import TSNE
tsne = TSNE(2)

#%%
clustered = tsne.fit_transform(latent_vectors)

#%%
fig = plt.figure(figsize=(12, 10))
cmap = plt.get_cmap("Spectral", 10)
plt.scatter(*zip(*clustered), c=classes, cmap=cmap)
plt.colorbar(drawedges=True)




# %%   #### understanding the limitation of embeddings that fall between clsuters
## instead of belonging to one.
latent_vectors = []
classes = []
for im, clss in val_dl:
    latent_vectors.append(model.encoder(im))
    classes.extend(clss)
latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy().reshape(10000, -1)

## generate random vectors with mean stadard deviation of and 1 with noise applied 
rand_vectors = []
for col in latent_vectors.transpose(1,0):
    mu, sigma = col.mean(), col.std()
    rand_vectors.append(sigma*torch.randn(1, 100) + mu)

rand_vectors = torch.cat(rand_vectors).transpose(1,0).to(device)
fig, ax = plt.subplots(10, 10, figsize=(7,7))
ax = iter(ax.flat)
for p in rand_vectors:
    img = model.decoder(p.reshape(1,64,2,2)).view(28, 28)
    show(img, ax=next(ax))



# %%
