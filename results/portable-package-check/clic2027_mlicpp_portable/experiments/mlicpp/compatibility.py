"""Optional entropy-only cuDNN control; keep analysis/synthesis on their original backend."""
from functools import wraps


def entropy_no_cudnn(model):
    import torch
    modules = [model.h_s]
    for name in ('local_context', 'channel_context', 'global_inter_context', 'global_intra_context',
                 'entropy_parameters_anchor', 'entropy_parameters_nonanchor', 'lrp_anchor', 'lrp_nonanchor'):
        modules.extend(m for m in getattr(model, name) if m is not None)
    for module in modules:
        original = module.forward
        @wraps(original)
        def forward(*args, original=original, **kwargs):
            with torch.backends.cudnn.flags(enabled=False):
                return original(*args, **kwargs)
        module.forward = forward
