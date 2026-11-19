"""Post-entropy feature adapter and regional edge-risk surrogate for the pilot."""
import torch
from torch import nn
from torch.nn import functional as F


def regional_pool(value):
    h, w = value.shape[-2:]
    if h % 2 or w % 2:
        raise ValueError('Regional pooling requires even dimensions')
    # Non-overlapping fixed pooling has deterministic CUDA backward in torch2;
    # adaptive_avg_pool2d does not. Preserve the same 2x2 regional definition.
    return F.avg_pool2d(value, (h//2, w//2), (h//2, w//2))


class Adapter(nn.Module):
    def __init__(self, channels=192, hidden=32, amplitude=.25):
        super().__init__()
        self.amplitude = amplitude
        self.delta = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(hidden, channels, 1))
        self.harm = nn.Sequential(nn.Conv2d(2*channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, feature, hide_change=False):
        delta = self.amplitude * self.delta(feature).tanh()
        # Detach the proposal from head inputs: its supervised target must not
        # teach the proposal to manipulate the error estimator.
        change = torch.zeros_like(delta) if hide_change else delta.detach()
        score = regional_pool(self.harm(torch.cat((feature, change), 1)))
        gate = 1 - F.interpolate(score.sigmoid(), size=feature.shape[-2:], mode='nearest')
        return feature + gate*delta, delta, score


def edge_regions(output, target, scale=1.0):
    """Per-image 2x2 regional normalized Sobel error; image-balanced sampling."""
    kernel = output.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])/8
    kernels = torch.stack((kernel, kernel.t())).unsqueeze(1)
    def grad(image):
        # Compare intensity edges, not RGB-independent hallucination guarantees.
        gray = image.mean(1, keepdim=True)
        return F.conv2d(gray, kernels, padding=1)
    difference = (grad(output)-grad(target)).abs().mean(1, keepdim=True)
    return (regional_pool(difference)/scale).clamp(0, 1).flatten(1)


def cvar_objective(values, tau, alpha=.9):
    return tau + (values-tau).relu().mean()/(1-alpha)


def empirical_cvar(values, alpha=.9):
    """Exact finite-sample upper-tail mean with fractional boundary mass."""
    ordered = torch.sort(values.flatten(), descending=True).values
    mass = (1-alpha)*len(ordered)
    whole = int(mass)
    fraction = mass-whole
    total = ordered[:whole].sum()
    if fraction > 1e-8:
        total = total + fraction*ordered[min(whole, len(ordered)-1)]
    return total/mass


def self_check():
    torch.manual_seed(1984)
    adapter = Adapter(channels=4, hidden=3)
    feature = torch.randn(2, 4, 8, 8)
    corrected, delta, score = adapter(feature)
    assert torch.equal(corrected, feature) and torch.count_nonzero(delta) == 0
    corrected.square().mean().backward()
    assert adapter.delta[-1].weight.grad.abs().sum() > 0
    image = torch.rand(2, 3, 16, 16)
    assert torch.equal(edge_regions(image, image), torch.zeros(2, 4))
    assert abs(empirical_cvar(torch.arange(1., 6.), .5).item()-4.2) < 1e-6
    tau = torch.tensor(.5, requires_grad=True)
    cvar_objective(torch.tensor([.1, .9]), tau, .5).backward()
    assert torch.isfinite(tau.grad)
    print('adapter identity/gradient, edge-zero, fractional-tail, CVaR gradient: passed')


if __name__ == '__main__':
    self_check()
