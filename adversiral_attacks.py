
#%%
from torch_snippets import inspect, show, np, torch, nn
from torchvision.models import resnet50

#%%  freeze parameters
model = resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
#%%
model = model.eval()
    


