from torch.nn.utils import weight_norm

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

# ---------------------------------------------------------------------------
# Gradient checkpointing (shared by the decoders and the diffusion model)
# ---------------------------------------------------------------------------
import torch
import torch.utils.checkpoint as _cp

_CHECKPOINT_TARGETS = set()


def set_gradient_checkpointing(enabled=True, targets=('decoder', 'diffusion')):
    """Enable/disable activation checkpointing for the given module groups.

    ``targets`` may contain 'decoder' (the HiFiGAN / iSTFTNet generator
    residual blocks, by far the largest activation consumer) and 'diffusion'
    (the style-diffusion transformer blocks).
    """
    _CHECKPOINT_TARGETS.clear()
    if enabled:
        _CHECKPOINT_TARGETS.update(targets)


def maybe_checkpoint(target, module, *args):
    """Call ``module(*args)``, under activation checkpointing when enabled for
    ``target`` and autograd is recording.  RNG state is preserved so modules
    that draw noise (e.g. the harmonic source) recompute identically."""
    if target in _CHECKPOINT_TARGETS and torch.is_grad_enabled():
        return _cp.checkpoint(module, *args, use_reentrant=False)
    return module(*args)
