
#%%
from torch_snippets import *
from torchvision import transforms as T
from torch.nn import functional as F
from torchvision.models import vgg19

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = T.Compose([T.ToTensor(), 
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                    ),
                        T.Lambda(lambda x: x.mul_(255))
                        ]
                       )

postprocess = T.Compose([T.Lambda(lambda x: x.mul_(1./255)),
                         T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                         ]
                        )

#%%
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        feat = input.view(b, c, h*w)
        G = feat@feat.transpose(1, 2)
        G.div_(h*w)
        return G
    
class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = F.mse_loss(GramMatrix()(input), target)
        return (out)
    
    
class vgg19_modified(nn.Module):
    def __init__(self):
        super().__init__()
        features = list(vgg19(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()    
    def forward(self, x, layers=[]):
        order = np.argsort(layers)
        _results, results = [], []
        for ix, model in enumerate(self.features):
            x = model(x)
            if ix in layers: _results.append(x)
        for o in order: 
            results.append(_results[o])
        return results if layers is not [] else x
        

#%%
vgg = vgg19_modified().to(device)


#%%
imgs = [Image.open(path).resize((512, 512)).convert("RGB") for path in ["style_image.png", "60.jpg"]]
style_image, content_image = [preprocess(img).to(device)[None] for img in imgs]

opt_img = content_image.data.clone()
opt_img.requires_grad = True

# layers for style and content loss
style_layers = [0, 5, 10, 19, 28]
content_layers = [21]
loss_layers = style_layers + content_layers

#%% # define loss func for content and style loss
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

#%% define weightage for content and style loss
style_weights = [1000/n**2 for n in [64,128,256,512,512]]
content_weights = [1]
weights = style_weights + content_weights

#%% 
style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
content_targets = [A.detach() for A in vgg(content_image, content_layers)]
targets = style_targets + content_targets


#%% define optimizer - LBFGS is used because it is performs better in deterministic scenarios like this
max_iters = 500
optimizer = optim.LBFGS([opt_img])
log = Report(max_iters)


#%%  ########## perform optimization  ############
iters = 0
while iters < max_iters:
    def closure():
        global iters
        iters += 1
        optimizer.zero_grad()
        out = vgg(opt_img,loss_layers)
        layer_losses = [weights[a]*loss_fns[a](A,targets[a]) for a,A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        log.record(pos=iters, loss=loss, end="\r")
        return loss
    optimizer.step(closure)
    
#%%
log.plot(log=True)

#%% plot img with content and style images
with torch.no_grad():
    out_img = postprocess(opt_img[0]).permute(1,2,0)
show(out_img)



#%%
!wget https://www.dropbox.com/s/z1y0fy2r6z6m6py/60.jpg
!wget https://www.dropbox.com/s/1svdliljyo0a98v/style_image.png


        
        

# %%
