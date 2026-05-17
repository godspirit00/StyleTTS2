from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)


# ---------------------------------------------------------------------------
# Training helpers: mixed precision + gradient checkpointing
# ---------------------------------------------------------------------------

def resolve_amp_dtype(spec):
    """Map a config 'mixed_precision' value to (autocast_enabled, dtype, use_scaler).

    Accepted spec values:
        - "no" / False / None / ""   -> AMP disabled
        - "fp16" / "float16"         -> autocast(fp16) + GradScaler
        - "bf16" / "bfloat16"        -> autocast(bf16), no scaler (cuda 8.0+)
    """
    if spec is None or spec is False:
        return False, torch.float32, False
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("no", "none", "off", "false", ""):
            return False, torch.float32, False
        if s in ("fp16", "float16", "half"):
            return True, torch.float16, True
        if s in ("bf16", "bfloat16"):
            return True, torch.bfloat16, False
    raise ValueError(f"Unrecognized mixed_precision value: {spec!r}")


class _NoOpScaler:
    """GradScaler stub used when AMP is disabled or running in bf16.

    Provides the same call surface (scale/step/update/unscale_) as
    torch.cuda.amp.GradScaler so the training loop can be written once.
    """
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        if optimizer is not None:
            optimizer.step()
    def update(self):
        pass
    def unscale_(self, optimizer):
        pass
    def state_dict(self):
        return {}
    def load_state_dict(self, state_dict):
        pass
    def is_enabled(self):
        return False


def make_grad_scaler(use_scaler: bool, enabled: bool):
    """Return a real ``torch.cuda.amp.GradScaler`` for fp16 AMP, else a stub."""
    if use_scaler and enabled:
        return torch.cuda.amp.GradScaler(enabled=True)
    return _NoOpScaler()


def enable_diffusion_gradient_checkpointing(enabled: bool) -> None:
    """Toggle gradient checkpointing inside the diffusion transformer blocks.

    Cuts activation memory at the cost of one extra forward pass per block
    during backward.  Safe to call before training starts.
    """
    try:
        from Modules.diffusion.modules import set_gradient_checkpointing
        set_gradient_checkpointing(enabled)
    except Exception:
        pass


def freeze_module(module):
    """Disable gradient tracking and switch to eval-mode for ``module``."""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()
    return module
