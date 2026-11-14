"""Small, trainable reference architectures; no pretrained model downloads."""

import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs,outputs,3,padding=1),nn.GELU(),
                         nn.Conv2d(outputs,outputs,3,padding=1),nn.GELU())


class CNN(nn.Module):
    def __init__(self,width,layers):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(7,width,3,padding=1),nn.GELU(),
            *[block(width,width) for _ in range(layers)],nn.Conv2d(width,2,1))

    def forward(self,x):
        return self.body(x)


class UNet(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.encoder=block(7,width)
        self.middle=block(width,2*width)
        self.decoder=block(3*width,width)
        self.output=nn.Conv2d(width,2,1)

    def forward(self,x):
        first=self.encoder(x)
        middle=self.middle(F.avg_pool2d(first,2,ceil_mode=True))
        expanded=F.interpolate(middle,size=first.shape[-2:],mode='bilinear',align_corners=False)
        return self.output(self.decoder(torch.cat((first,expanded),dim=1)))


class SpectralConv(nn.Module):
    def __init__(self,width,modes):
        super().__init__()
        self.modes=modes
        shape=(width,width,modes,modes)
        self.positive=nn.Parameter(torch.randn(shape,dtype=torch.complex64)/width)
        self.negative=nn.Parameter(torch.randn(shape,dtype=torch.complex64)/width)

    def forward(self,x):
        transformed=torch.fft.rfft2(x,norm='ortho')
        height=min(self.modes,x.shape[-2]//2)
        width=min(self.modes,transformed.shape[-1])
        output=torch.zeros_like(transformed)
        output[:,:,:height,:width]=torch.einsum('bihw,iohw->bohw',
            transformed[:,:,:height,:width],self.positive[:,:,:height,:width])
        output[:,:,-height:,:width]=torch.einsum('bihw,iohw->bohw',
            transformed[:,:,-height:,:width],self.negative[:,:,:height,:width])
        return torch.fft.irfft2(output,s=x.shape[-2:],norm='ortho')


class FNO(nn.Module):
    def __init__(self,width,modes,layers):
        super().__init__()
        self.lift=nn.Conv2d(7,width,1)
        self.spectral=nn.ModuleList([SpectralConv(width,modes) for _ in range(layers)])
        self.local=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(layers)])
        self.output=nn.Sequential(nn.Conv2d(width,2*width,1),nn.GELU(),nn.Conv2d(2*width,2,1))

    def forward(self,x):
        x=self.lift(x)
        for spectral,local in zip(self.spectral,self.local):
            x=F.gelu(spectral(x)+local(x))
        return self.output(x)


class DeepONet(nn.Module):
    def __init__(self,width,latent):
        super().__init__()
        self.latent=latent
        self.branch=nn.Sequential(block(7,width),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),
            nn.Linear(16*width,width*2),nn.GELU(),nn.Linear(width*2,2*latent))
        self.trunk=nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.Linear(width,latent))
        self.bias=nn.Parameter(torch.zeros(2))

    def forward(self,x):
        coordinates=x[0,-2:].permute(1,2,0).reshape(-1,2)
        trunk=self.trunk(coordinates)
        branch=self.branch(x).reshape(len(x),2,self.latent)
        return (torch.einsum('bck,nk->bcn',branch,trunk)+self.bias[None,:,None]).reshape(len(x),2,*x.shape[-2:])


class MeshTransformer(nn.Module):
    """Pooled field tokens plus node-to-token attention; not Transolver weights."""
    def __init__(self,width,layers):
        super().__init__()
        self.lift=nn.Conv2d(7,width,1)
        encoder=nn.TransformerEncoderLayer(width,4,2*width,dropout=0.,activation='gelu',batch_first=True)
        self.encoder=nn.TransformerEncoder(encoder,layers,enable_nested_tensor=False)
        self.attention=nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.output=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,2))

    def forward(self,x):
        embedded=self.lift(x)
        tokens=F.adaptive_avg_pool2d(embedded,(4,4)).flatten(2).transpose(1,2)
        tokens=self.encoder(tokens)
        queries=embedded.flatten(2).transpose(1,2)
        attended,_=self.attention(queries,tokens,tokens,need_weights=False)
        result=self.output(queries+attended)
        return result.transpose(1,2).reshape(len(x),2,*x.shape[-2:])


class PODMLP(nn.Module):
    def __init__(self,width,nodes,rank,basis=None):
        super().__init__()
        self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(7*nodes,width*2),nn.GELU(),
            nn.Linear(width*2,width*2),nn.GELU(),nn.Linear(width*2,rank))
        self.register_buffer('basis',torch.zeros((2*nodes,rank)) if basis is None else torch.as_tensor(basis,dtype=torch.float32))

    def forward(self,x):
        flattened=self.encoder(x) @ self.basis.T
        return flattened.reshape(len(x),*x.shape[-2:],2).permute(0,3,1,2)


def make_network(options,config,*,rank=None,basis=None):
    family=options.family
    if family=='pod_mlp': return PODMLP(options.width,(config.nx+1)*(config.ny+1),rank,basis)
    if family=='cnn': return CNN(options.width,options.layers)
    if family=='unet': return UNet(options.width)
    if family=='fno': return FNO(options.width,options.fourier_modes,options.layers)
    if family=='deeponet': return DeepONet(options.width,options.latent)
    if family=='transformer': return MeshTransformer(options.width,options.layers)
    raise ValueError('This family does not use a neural network')


def predict_network(network,inputs,scale):
    # Clamp is exact at every inference, rather than learned from a loss term.
    return network(inputs)*(1-inputs[:,4:5])*scale
