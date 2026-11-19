"""Matched conditional/fixed feature bases and a conventional pixel-DCT control."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Innovation(nn.Module):
    def __init__(self,template,rank=8,grid=2,mode='conditional',step=1/64,amplitude=.25):
        super().__init__()
        self.rank,self.grid,self.mode,self.step,self.amplitude=rank,grid,mode,step,amplitude
        self.register_buffer('template',template.clone())
        self.maps=nn.Sequential(nn.Conv2d(192,32,1),nn.GELU(),nn.Conv2d(32,rank,3,padding=1),nn.Tanh())
        self.channels=nn.Conv2d(rank,192,1,bias=False)
        self.feature_projection=nn.Conv2d(192,16,1)
        self.encoder=nn.Sequential(nn.Conv2d(6,32,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(32,32,3,stride=2,padding=1),nn.GELU())
        self.code=nn.Sequential(nn.Conv2d(48,32,3,padding=1),nn.GELU(),nn.Conv2d(32,rank,1))
        nn.init.zeros_(self.code[-1].weight);nn.init.zeros_(self.code[-1].bias)

    def propose(self,feature,source,base):
        image=self.encoder(torch.cat((source,base),1))
        value=self.code(torch.cat((image,self.feature_projection(feature)),1))
        h,w=value.shape[-2:]
        assert h%self.grid==0 and w%self.grid==0
        return 1.984375*F.avg_pool2d(value,(h//self.grid,w//self.grid)).tanh()

    def quantize(self,c,ste=False):
        q=(c/self.step).round().clamp(-127,127)
        return c+(q*self.step-c).detach() if ste else q*self.step

    def render(self,feature,base,tail,c):
        if self.mode=='pixel':
            h,w=base.shape[-2:];th,tw=h//self.grid,w//self.grid
            yy=(torch.arange(th,device=base.device,dtype=base.dtype)+.5)/th
            xx=(torch.arange(tw,device=base.device,dtype=base.dtype)+.5)/tw
            spec=[(0,0,0),(1,0,0),(2,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(2,1,0)]
            assert self.rank==8
            tiles=[]
            for channel,fy,fx in spec:
                spatial=torch.cos(math.pi*fy*yy[:,None])*torch.cos(math.pi*fx*xx[None,:])
                color=base.new_zeros(3);color[channel]=1
                tiles.append(color[:,None,None]*spatial[None])
            basis=torch.stack(tiles)
            correction=base.new_zeros(base.shape)
            for gy in range(self.grid):
                for gx in range(self.grid):
                    correction[:,:,gy*th:(gy+1)*th,gx*tw:(gx+1)*tw]=torch.einsum('bk,kchw->bchw',c[:,:,gy,gx],basis)
            return (base+self.amplitude*correction).clamp(0,1)
        value=feature if self.mode=='conditional' else self.template.expand(feature.shape[0],-1,-1,-1)
        maps=self.maps(value)
        weights=F.interpolate(c,size=feature.shape[-2:],mode='nearest')
        correction=self.amplitude*self.channels(maps*weights)
        return tail(feature+correction).clamp(0,1)
