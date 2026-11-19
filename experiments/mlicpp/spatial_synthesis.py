"""Decoder-side latent residual controlled by local or image-wide activity."""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialSynthesis(nn.Module):
    def __init__(self, synthesis, channels, mode):
        super().__init__()
        if mode not in {'local', 'global'} or channels < 1:
            raise ValueError('Invalid spatial synthesis configuration')
        self.synthesis = synthesis
        self.channels = channels
        self.mode = mode
        width = 16
        self.reduce = nn.Conv2d(channels, width, 1)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.gate = nn.Conv2d(1, width, 1)
        self.expand = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, latent):
        mean = F.avg_pool2d(latent, 3, stride=1, padding=1, count_include_pad=False)
        activity = (latent - mean).abs().mean(dim=1, keepdim=True).log1p()
        if self.mode == 'global':
            activity = activity.mean(dim=(-2, -1), keepdim=True)
        features = self.spatial(F.gelu(self.reduce(latent)))
        correction = self.expand(features * (2 * self.gate(activity).sigmoid()))
        return self.synthesis(latent + correction)


def load_synthesis(model, checkpoint):
    architecture = checkpoint.get('architecture')
    if architecture is not None:
        if architecture['mode'] in {'entropy', 'entropy-blind'}:
            from entropy_synthesis import EntropySynthesis
            wrapper = EntropySynthesis
        else:
            wrapper = SpatialSynthesis
        model.g_s = wrapper(model.g_s, **architecture).to(next(model.parameters()).device)
    model.g_s.load_state_dict(checkpoint['synthesis'])
    model.g_s.eval()
